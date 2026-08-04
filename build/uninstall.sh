#!/usr/bin/env bash
# accretion uninstaller. Removes /opt/accretion and the systemd unit.
# Keeps /opt/accretion/etc (your config) unless --purge is given.
set -euo pipefail
PREFIX=/opt/accretion
[ "$(id -u)" = 0 ] || { echo "run as root: sudo ./uninstall.sh [--purge]" >&2; exit 2; }

if systemctl list-unit-files ds4-server.service >/dev/null 2>&1 && [ -f /etc/systemd/system/ds4-server.service ]; then
  systemctl disable --now ds4-server.service 2>/dev/null || true
  rm -f /etc/systemd/system/ds4-server.service
  systemctl daemon-reload
fi

rm -rf "$PREFIX/bin" "$PREFIX/tools" "$PREFIX/VERSION"
if [ "${1:-}" = "--purge" ]; then
  rm -rf "$PREFIX"
  echo "== removed $PREFIX entirely (config included)"
else
  rmdir "$PREFIX" 2>/dev/null || echo "== kept $PREFIX/etc (config); use --purge to remove"
fi
