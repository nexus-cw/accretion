"""Dump a GGUF header: metadata keys/values (arrays summarized) and a
tensor summary grouped by family (blk.N -> blk.N folded)."""
import re
import sys
from collections import OrderedDict

import gguf


def main(path):
    g = gguf.read_header(path)
    print('version=%d alignment=%d data_start=%d n_tensors=%d n_kv=%d' %
          (g.version, g.alignment, g.data_start, len(g.tensors), len(g.metadata)))
    for k, (vt, v) in g.metadata.items():
        if vt == gguf.T_ARR:
            et, items = v
            head = items[:4]
            print('KV %s : arr(etype=%d, n=%d) head=%r' % (k, et, len(items), head))
        else:
            vs = repr(v)
            if len(vs) > 80:
                vs = vs[:80] + '...'
            print('KV %s : t=%d %s' % (k, vt, vs))
    fams = OrderedDict()
    for t in g.tensors:
        fam = re.sub(r'^blk\.\d+\.', 'blk.N.', t.name)
        key = (fam, gguf.type_name(t.ggml_type), tuple(t.dims))
        e = fams.setdefault(key, [0, 0, t.offset, t.name])
        e[0] += 1
        e[1] += t.nbytes
    for (fam, tn, dims), (cnt, nb, off0, first) in fams.items():
        print('TF %-45s %-7s dims=%-22s n=%-3d bytes=%d' % (fam, tn, list(dims), cnt, nb))
    total = sum(t.nbytes for t in g.tensors)
    end = g.data_start + max(t.offset + t.nbytes for t in g.tensors)
    print('total_tensor_bytes=%d data_end=%d' % (total, end))
    # offset order vs listed order
    mono = all(g.tensors[i].offset <= g.tensors[i+1].offset for i in range(len(g.tensors)-1))
    print('offsets_monotonic_in_list_order=%s' % mono)


if __name__ == '__main__':
    main(sys.argv[1])
