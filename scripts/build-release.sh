#!/usr/bin/env bash
# build-release.sh — cut a versioned accretion release tarball.
#
# CONTRIBUTOR tool. Consumers never build (docs/PRINCIPLES.md rule 1) —
# they download the tarball this script produces.
#
# Run from the accretion repo root ON the target machine for the target:
#   gb10-arm64-cuda : GB10 / DGX Spark (sm_121a), builds engine with `make cuda-spark`
#
# Usage: scripts/build-release.sh [--target gb10-arm64-cuda] [--serial N | --version X.Y.Z]
#   --serial N     date-serial for the 0.1.0-dev.N version (default: 1)
#   --version X.Y.Z  exact version (milestone releases, e.g. 0.1.0)
# Output: dist/accretion-<version>-<target>.tar.gz (+ .sha256)
set -euo pipefail

TARGET=gb10-arm64-cuda
SERIAL=1
VERSION_OVERRIDE=
while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --serial) SERIAL="$2"; shift 2 ;;
    --version) VERSION_OVERRIDE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ "$TARGET" = gb10-arm64-cuda ] || { echo "unsupported target: $TARGET (only gb10-arm64-cuda for now)" >&2; exit 2; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -f engine/Makefile ] || { echo "engine/ subtree missing — run from accretion repo root" >&2; exit 2; }

VERSION="${VERSION_OVERRIDE:-0.1.0-dev.${SERIAL}}"
ACCRETION_COMMIT="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
# engine subtree commit: last squash commit records "Squashed 'engine/' changes from X..Y" (or "... (Y)" on the seed commit)
ENGINE_COMMIT="$(git log --grep="Squashed 'engine/'" -1 --format=%s 2>/dev/null | grep -oE '[0-9a-f]{7,}$' || true)"
[ -n "$ENGINE_COMMIT" ] || ENGINE_COMMIT=unknown

echo "== build engine ($TARGET)"
make -C engine cuda-spark

# ds4-inkling-server builds from the engine subtree once the inkling-port
# branch (ds4 fork) lands in platform via subtree sync — see
# docs/ARCH_SEAM.md "The inkling row". Until that sync happens the make
# target below won't exist; fail loudly rather than ship a stale/missing
# binary in a tagged release.
# The inkling server MUST be the CUDA build: the CPU engine is the
# correctness reference and runs ~25x slower (task #36 shipped a CPU-linked
# server to production by accident).  Fail the release rather than ship a
# binary that serves at 0.5 t/s -- ds4-inkling-server-cpu exists for hosts
# that deliberately want the reference engine.
# Look where the engine Makefile actually looks, not just on PATH: a
# non-interactive ssh session on the GPU host has CUDA installed but not
# exported, which made this guard reject a perfectly good build host.
if ! command -v nvcc > /dev/null 2>&1; then
  for cand in "${NVCC:-}" /usr/local/cuda/bin/nvcc /opt/cuda/bin/nvcc; do
    if [ -n "$cand" ] && [ -x "$cand" ]; then
      PATH="$(dirname "$cand"):$PATH"; export PATH; break
    fi
  done
fi
if ! command -v nvcc > /dev/null 2>&1; then
  echo "nvcc not found: ds4-inkling-server must be built with CUDA" >&2
  echo "(a CPU-linked server serves ~25x slower; see PORT_NOTES.md M10)." >&2
  echo "Searched: PATH, \$NVCC, /usr/local/cuda/bin, /opt/cuda/bin." >&2
  echo "Install CUDA or build on the GPU host; refusing to cut a release." >&2
  exit 1
fi

echo "== build engine ($TARGET) - ds4-inkling-server (CUDA)"
make -C engine ds4-inkling-server || {
  echo "missing engine target ds4-inkling-server — the inkling-port branch" >&2
  echo "has not been subtree-synced into engine/ yet; sync it (see" >&2
  echo "docs/ARCH_SEAM.md 'The inkling row') before cutting this release" >&2
  exit 1
}

BINARIES="ds4 ds4-server ds4-eval ds4-inkling-server"
STAGE="$(mktemp -d)"
PKG="accretion-${VERSION}-${TARGET}"
mkdir -p "$STAGE/$PKG"/{bin,tools/accretion-prepare}

for b in $BINARIES; do
  [ -x "engine/$b" ] || { echo "missing binary engine/$b" >&2; exit 1; }
  cp "engine/$b" "$STAGE/$PKG/bin/"
done

cp tools/accretion-prepare/*.py tools/accretion-prepare/README.md "$STAGE/$PKG/tools/accretion-prepare/"
cp build/requirements-accretion-prepare.txt "$STAGE/$PKG/tools/accretion-prepare/requirements.txt"
cp build/install.sh build/uninstall.sh build/accretion-serve "$STAGE/$PKG/"
chmod +x "$STAGE/$PKG/accretion-serve"
cp build/ds4-server.service.template "$STAGE/$PKG/"
cp LICENSE README.md "$STAGE/$PKG/"
chmod +x "$STAGE/$PKG"/install.sh "$STAGE/$PKG"/uninstall.sh

{
  echo "version: $VERSION"
  echo "target: $TARGET"
  echo "accretion-commit: $ACCRETION_COMMIT"
  echo "engine-commit: $ENGINE_COMMIT"
  echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "builder: $(uname -srm)"
} > "$STAGE/$PKG/VERSION"

( cd "$STAGE/$PKG" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )

mkdir -p "$ROOT/dist"
TARBALL="$ROOT/dist/${PKG}.tar.gz"
tar -C "$STAGE" -czf "$TARBALL" "$PKG"
( cd "$ROOT/dist" && sha256sum "${PKG}.tar.gz" > "${PKG}.tar.gz.sha256" )
rm -rf "$STAGE"

echo "== built $TARBALL"
ls -lh "$TARBALL"
cat "$ROOT/dist/${PKG}.tar.gz.sha256"
