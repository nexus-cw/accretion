#!/usr/bin/env bash
# accretion installer — idempotent. Installs to /opt/accretion.
# Run from inside an extracted release tarball, as root (sudo ./install.sh).
#
# - Binaries -> /opt/accretion/bin (always refreshed)
# - Tools    -> /opt/accretion/tools (always refreshed)
# - Config   -> /opt/accretion/etc/ds4-server.env (created ONLY if missing;
#               a reinstall/upgrade never overwrites your config)
# - Systemd  -> /etc/systemd/system/ds4-server.service (created ONLY if missing)
set -euo pipefail

PREFIX=/opt/accretion
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HERE/VERSION" ] || { echo "run from inside an extracted accretion release" >&2; exit 2; }
[ "$(id -u)" = 0 ] || { echo "run as root: sudo ./install.sh" >&2; exit 2; }

echo "== verifying manifest"
( cd "$HERE" && sha256sum --quiet -c MANIFEST.sha256 )

mkdir -p "$PREFIX"/{bin,tools,etc}
cp "$HERE"/bin/* "$PREFIX/bin/"
rm -rf "$PREFIX/tools/accretion-prepare"
cp -r "$HERE/tools/accretion-prepare" "$PREFIX/tools/"
cp "$HERE/VERSION" "$PREFIX/VERSION"

if [ ! -f "$PREFIX/etc/ds4-server.env" ]; then
  cat > "$PREFIX/etc/ds4-server.env" <<'EOF'
# accretion ds4-server configuration. Edit, then: systemctl restart ds4-server
# Values below mirror the production-proven GB10 setup; MODEL is mandatory.
DS4_MODEL=/path/to/model.gguf
DS4_HOST=0.0.0.0
DS4_PORT=8000
DS4_CTX=131072
DS4_EXPERT_CACHE=70GB
DS4_SESSIONS=2
DS4_KV_DIR=/var/lib/accretion/kv
DS4_KV_SPACE_MB=16384
# Model picker (web console): the server rewrites DS4_MODEL here on
# POST /v1/models/select, then restarts. DS4_ENV_FILE tells it which file.
DS4_ENV_FILE=/opt/accretion/etc/ds4-server.env
# Extra directories to scan for switchable *.gguf models (colon-separated).
#DS4_MODEL_DIRS=/data/models
# Set a token to ENABLE model switching from the console/API. Unset =
# switching disabled (the console shows a read-only picker).
#ACCRETION_ADMIN_TOKEN=change-me
EOF
  echo "== wrote default config $PREFIX/etc/ds4-server.env — EDIT DS4_MODEL before starting"
else
  echo "== keeping existing config $PREFIX/etc/ds4-server.env"
fi

UNIT=/etc/systemd/system/ds4-server.service
if [ ! -f "$UNIT" ]; then
  cp "$HERE/ds4-server.service.template" "$UNIT"
  systemctl daemon-reload
  echo "== installed $UNIT (not enabled; enable with: systemctl enable --now ds4-server)"
else
  echo "== keeping existing $UNIT"
fi

echo "== installed accretion $(head -1 "$PREFIX/VERSION" | cut -d' ' -f2) to $PREFIX"
echo "   binaries: $(ls "$PREFIX/bin" | tr '\n' ' ')"
