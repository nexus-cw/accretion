"""Per-family ingest descriptors for accretion-prepare.

This package is the primary arch seam of the accretion toolchain
(ratified design, task #26): adding a new MoE family to *prepare*
means adding one descriptor module here -- not forking prepare.py.

Descriptor contract
-------------------
A descriptor is a subclass of `Descriptor` registered in `REGISTRY`.
Fields / methods (all optional beyond `identify`; base class supplies
generic no-op behavior):

  identify(g) -> bool
      Header-only test: does this descriptor own the parsed GGUF `g`
      (a gguf.GGUFHeader-like object with .metadata / .tensors)?
      Selection walks REGISTRY in order; first verified match wins,
      else GenericDescriptor (graceful skip of all normalization).

  verified : bool
      Skeleton descriptors (fields guessed, not tested against a real
      artifact) set this False; selection ignores them and falls to
      the generic path, so a stub can exist as a file without ever
      being on the hot path.

  derive_metadata(g, xlog)   [the "metadata_rules" slot]
      Add canonical <arch>.* metadata keys the source omits, derived
      from tensor shapes/presence with cross-checks; every derivation
      appended to the transform log `xlog`. No-op in the base class.

  tensor_aliases : dict
      alias (community dialect) name -> canonical name. Per-layer
      entries use '{L}'. `canonical_name(name)` applies them.

  dense_type_policy(t) -> bool
      True if dense tensor `t` should be converted to F16 during
      NORMALIZE. Base class: never convert.

  expert_pattern : compiled regex
      Matches routed-expert tensor names; group(1)=layer index,
      group(2)=projection (gate|up|down). Drives both the OPTIMIZE
      expert-major layout and the manifest (layer,expert) enumeration.

  alignment : int
      Output tensor-offset alignment (bytes) for the OPTIMIZE stage.
      4096 everywhere today (O_DIRECT/cuFile-native).

  normalizes() -> bool
      Whether the NORMALIZE stage applies at all for this family.
      GenericDescriptor returns False -> prepare logs the graceful
      'skip_foreign_arch' transform (behavior introduced in ee2f023).

  lineage_hooks : dict
      RESERVED for the wakestone family (task #14 manifest / #31
      rung 3): fields are defined here so the seam already has a slot
      for dynamic expert ids and parent lineage, but nothing reads
      them yet.
        dynamic_expert_ids : bool  -- expert set may grow/shrink over
                                      the artifact's life; manifest
                                      expert ids are then identities,
                                      not dense indices.
        parent_lineage     : str|None -- manifest key naming the
                                      parent artifact an expert was
                                      accreted from.
        manifest_extra     : dict|None -- family-specific manifest
                                      fields to merge into
                                      manifest['model'].
"""
import re


class Descriptor:
    name = 'generic'
    verified = True
    alignment = 4096
    tensor_aliases = {}
    # llama.cpp-convention routed-expert tensors; shared by every family
    # seen so far (deepseek4, laguna, kimi-k3 all use ffn_*_exps).
    expert_pattern = re.compile(r'^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$')
    lineage_hooks = {
        'dynamic_expert_ids': False,   # reserved (wakestone)
        'parent_lineage': None,        # reserved (wakestone)
        'manifest_extra': None,        # reserved (wakestone)
    }

    def identify(self, g):
        raise NotImplementedError

    def normalizes(self):
        return False

    def derive_metadata(self, g, xlog):
        pass

    def canonical_name(self, name):
        m = re.match(r'^blk\.(\d+)\.(.*)$', name)
        if m:
            il, rest = m.group(1), m.group(2)
            key = 'blk.{L}.' + rest
            if key in self.tensor_aliases:
                return self.tensor_aliases[key].replace('{L}', il)
            return name
        return self.tensor_aliases.get(name, name)

    def dense_type_policy(self, t):
        return False


class GenericDescriptor(Descriptor):
    """Fallback: no family knowledge. NORMALIZE is skipped gracefully
    (skip_foreign_arch); OPTIMIZE + MANIFEST run on the llama.cpp
    ffn_*_exps convention."""
    name = 'generic'

    def identify(self, g):
        return True


def _arch(g):
    return g.metadata.get('general.architecture', (None, None))[1]


from . import deepseek4  # noqa: E402

REGISTRY = [
    deepseek4.Deepseek4Descriptor(),
]
GENERIC = GenericDescriptor()


def select(g, log=None):
    """Pick the first verified descriptor that identifies the header;
    fall back to the generic (graceful-skip) descriptor."""
    for d in REGISTRY:
        if d.identify(g):
            if not d.verified:
                if log:
                    log('descriptor %r matched but is an UNVERIFIED skeleton; '
                        'using generic path' % d.name)
                continue
            return d
    return GENERIC
