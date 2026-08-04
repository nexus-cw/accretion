"""Pre-warm a GGUF's header + all non-routed (dense) tensor byte ranges into
the OS page cache with buffered reads. Routed expert tensors are left cold
(they are streamed via O_DIRECT during prefill, which bypasses page cache),
so warming dense only affects boot-time model load, not the measured
expert-streaming phase."""
import re
import sys
import time


import gguf

ROUTED = re.compile(r'^blk\.\d+\.ffn_(gate|up|down)_exps\.weight$')

path = sys.argv[1]
g = gguf.read_header(path)
t0 = time.time()
total = 0
with open(path, 'rb') as f:
    f.seek(0)
    while f.tell() < g.data_start:
        b = f.read(min(1 << 24, g.data_start - f.tell()))
        if not b:
            break
        total += len(b)
    for t in sorted(g.tensors, key=lambda x: x.offset):
        if ROUTED.match(t.name):
            continue
        f.seek(g.data_start + t.offset)
        left = t.nbytes
        while left:
            b = f.read(min(1 << 24, left))
            if not b:
                break
            left -= len(b)
            total += len(b)
print('prewarmed %.2f GiB in %.0fs' % (total / 2**30, time.time() - t0))
