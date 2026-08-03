#!/usr/bin/env bash
# Sync the engine/ subtree from the nexus-cw/ds4 `platform` branch.
#
# The `platform` branch itself tracks antirez/ds4 main via the existing
# 6-hourly sync plus manual rebases; this script only pulls its current
# state into the subtree. Run from the repo root on a clean tree.
set -euo pipefail
git subtree pull --prefix=engine https://github.com/nexus-cw/ds4 platform --squash
