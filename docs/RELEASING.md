# Releasing accretion

Consumers never build code (docs/PRINCIPLES.md, rule 1). A release is the moment
contributor work becomes a downloadable artifact; `make` and this document are
for contributors only.

## Version scheme

- Pre-milestone: `0.1.0-dev.N`, where N is a date-serial (bump per cut release,
  not per day). Tag: `v0.1.0-dev.N`.
- The `VERSION` file inside each tarball embeds the accretion commit and the
  engine subtree commit the artifact was built from, plus target and build time.
- At the v0.1 milestone (one install command + browser console + pick model +
  point harness) the scheme moves to plain semver `0.1.0`.

## Target matrix

| Target | Status | Notes |
| --- | --- | --- |
| `gb10-arm64-cuda` | current | GB10 / DGX Spark, sm_121a; built with `make cuda-spark` on a GB10 machine |
| generic arm64 CUDA | planned | `make cuda-generic` |
| x86_64 CUDA | planned | |
| Apple / Metal | planned | rides upstream ds4 Metal support |

## How a release is cut

1. Sync the engine subtree to the current `platform` head of nexus-cw/ds4
   (`scripts/sync-engine.sh` / `git subtree pull --prefix=engine ... platform --squash`)
   and push accretion `main`.
2. On the target machine (robo-dog for gb10-arm64-cuda), clone the accretion
   repo into a scratch dir and run from the repo root:

       scripts/build-release.sh --target gb10-arm64-cuda --serial N

   This builds the engine (`make cuda-spark`), stages `bin/` (ds4, ds4-server,
   ds4-eval), `tools/accretion-prepare` with pinned requirements, `install.sh` /
   `uninstall.sh`, the systemd unit template, `VERSION`, and a `MANIFEST.sha256`,
   then emits `dist/accretion-<version>-<target>.tar.gz` (+ `.sha256`).
3. Smoke the artifact without touching any production server: extract to a
   scratch prefix, `sha256sum -c MANIFEST.sha256`, run each binary `--help`,
   check `ldd` resolves. (A server boot test needs the GPU, which production
   may hold — binary-level checks are the gate.)
4. Tag and publish:

       git tag v<version> && git push origin v<version>
       gh release create v<version> dist/accretion-<version>-<target>.tar.gz \
         dist/accretion-<version>-<target>.tar.gz.sha256 --title ... --notes ...

## Install layout (what install.sh does)

Prefix is `/opt/accretion`: `bin/` (refreshed on every install), `tools/`
(refreshed), `etc/ds4-server.env` (created only if missing — never overwritten
on reinstall), plus `/etc/systemd/system/ds4-server.service` (installed only if
missing). `uninstall.sh` removes everything except config unless `--purge`.
