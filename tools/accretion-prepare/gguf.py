"""Minimal streaming GGUF v3 reader/writer (stdlib only).

Reads the header + tensor-info table without touching tensor data; tensor
data is streamed in chunks, never loaded whole. Write side emits a fresh
GGUF v3 header and copies/transforms tensor data tensor-by-tensor.
"""
import struct
from dataclasses import dataclass, field

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

# GGUF metadata value types
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)

_SCALAR_FMT = {
    T_U8: '<B', T_I8: '<b', T_U16: '<H', T_I16: '<h', T_U32: '<I',
    T_I32: '<i', T_F32: '<f', T_U64: '<Q', T_I64: '<q', T_F64: '<d',
}

# ggml tensor types: name, block_elems, block_bytes
GGML_TYPES = {
    0:  ('F32', 1, 4),
    1:  ('F16', 1, 2),
    2:  ('Q4_0', 32, 18),
    3:  ('Q4_1', 32, 20),
    6:  ('Q5_0', 32, 22),
    7:  ('Q5_1', 32, 24),
    8:  ('Q8_0', 32, 34),
    9:  ('Q8_1', 32, 36),
    10: ('Q2_K', 256, 84),
    11: ('Q3_K', 256, 110),
    12: ('Q4_K', 256, 144),
    13: ('Q5_K', 256, 176),
    14: ('Q6_K', 256, 210),
    15: ('Q8_K', 256, 292),
    16: ('IQ2_XXS', 256, 66),
    24: ('I8', 1, 1),
    25: ('I16', 1, 2),
    26: ('I32', 1, 4),
    27: ('I64', 1, 8),
    28: ('F64', 1, 8),
    30: ('BF16', 1, 2),
    39: ('MXFP4', 32, 17),
}
TYPE_BY_NAME = {v[0]: k for k, v in GGML_TYPES.items()}


def type_name(t):
    return GGML_TYPES[t][0] if t in GGML_TYPES else 'UNK%d' % t


def tensor_nbytes(ggml_type, n_elements):
    name, be, bb = GGML_TYPES[ggml_type]
    assert n_elements % be == 0, (name, n_elements)
    return (n_elements // be) * bb


@dataclass
class TensorInfo:
    name: str
    dims: list           # GGUF order: dims[0] is fastest-varying (ne[0])
    ggml_type: int
    offset: int          # relative to data section start
    nbytes: int = 0
    n_elements: int = 0

    def finish(self):
        n = 1
        for d in self.dims:
            n *= d
        self.n_elements = n
        self.nbytes = tensor_nbytes(self.ggml_type, n)
        return self


@dataclass
class GGUFFile:
    path: str
    version: int = GGUF_VERSION
    metadata: dict = field(default_factory=dict)   # key -> (type, value); arrays: (T_ARR, (elem_type, list))
    tensors: list = field(default_factory=list)
    data_start: int = 0
    alignment: int = 32


class Reader:
    def __init__(self, f):
        self.f = f

    def _read(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError('short read')
        return b

    def u32(self):
        return struct.unpack('<I', self._read(4))[0]

    def u64(self):
        return struct.unpack('<Q', self._read(8))[0]

    def string(self):
        n = self.u64()
        return self._read(n).decode('utf-8')

    def value(self, vtype):
        if vtype in _SCALAR_FMT:
            fmt = _SCALAR_FMT[vtype]
            return struct.unpack(fmt, self._read(struct.calcsize(fmt)))[0]
        if vtype == T_BOOL:
            return self._read(1)[0] != 0
        if vtype == T_STR:
            return self.string()
        if vtype == T_ARR:
            etype = self.u32()
            count = self.u64()
            if etype in _SCALAR_FMT:
                fmt = _SCALAR_FMT[etype][1]
                sz = struct.calcsize('<' + fmt)
                raw = self._read(sz * count)
                return (etype, list(struct.unpack('<%d%s' % (count, fmt), raw)))
            if etype == T_STR:
                return (etype, [self.string() for _ in range(count)])
            if etype == T_BOOL:
                raw = self._read(count)
                return (etype, [b != 0 for b in raw])
            raise ValueError('bad array elem type %d' % etype)
        raise ValueError('bad value type %d' % vtype)


def read_header(path):
    g = GGUFFile(path=path)
    with open(path, 'rb') as f:
        r = Reader(f)
        if r.u32() != GGUF_MAGIC:
            raise ValueError('not a GGUF file: %s' % path)
        g.version = r.u32()
        if g.version not in (2, 3):
            raise ValueError('unsupported GGUF version %d' % g.version)
        n_tensors = r.u64()
        n_kv = r.u64()
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            g.metadata[key] = (vtype, r.value(vtype))
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            ggml_type = r.u32()
            offset = r.u64()
            g.tensors.append(TensorInfo(name, dims, ggml_type, offset).finish())
        if 'general.alignment' in g.metadata:
            g.alignment = g.metadata['general.alignment'][1]
        pos = f.tell()
        g.data_start = (pos + g.alignment - 1) // g.alignment * g.alignment
    return g


class Writer:
    def __init__(self, f):
        self.f = f

    def u32(self, v):
        self.f.write(struct.pack('<I', v))

    def u64(self, v):
        self.f.write(struct.pack('<Q', v))

    def string(self, s):
        b = s.encode('utf-8')
        self.u64(len(b))
        self.f.write(b)

    def value(self, vtype, v):
        if vtype in _SCALAR_FMT:
            self.f.write(struct.pack(_SCALAR_FMT[vtype], v))
        elif vtype == T_BOOL:
            self.f.write(b'\x01' if v else b'\x00')
        elif vtype == T_STR:
            self.string(v)
        elif vtype == T_ARR:
            etype, items = v
            self.u32(etype)
            self.u64(len(items))
            if etype in _SCALAR_FMT:
                fmt = _SCALAR_FMT[etype][1]
                self.f.write(struct.pack('<%d%s' % (len(items), fmt), *items))
            elif etype == T_STR:
                for s in items:
                    self.string(s)
            elif etype == T_BOOL:
                self.f.write(bytes(1 if x else 0 for x in items))
            else:
                raise ValueError('bad array elem type %d' % etype)
        else:
            raise ValueError('bad value type %d' % vtype)


def write_header(f, metadata, tensors, alignment):
    """Write header + tensor info table; tensors must already carry final
    (aligned) relative offsets. Returns data_start (absolute)."""
    w = Writer(f)
    w.u32(GGUF_MAGIC)
    w.u32(GGUF_VERSION)
    w.u64(len(tensors))
    w.u64(len(metadata))
    for key, (vtype, v) in metadata.items():
        w.string(key)
        w.u32(vtype)
        w.value(vtype, v)
    for t in tensors:
        w.string(t.name)
        w.u32(len(t.dims))
        for d in t.dims:
            w.u64(d)
        w.u32(t.ggml_type)
        w.u64(t.offset)
    pos = f.tell()
    data_start = (pos + alignment - 1) // alignment * alignment
    f.write(b'\x00' * (data_start - pos))
    return data_start
