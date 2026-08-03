# Accretion design principles

Operator-set ground rules. Everything in the platform layer is measured against these.

## 1. Never expect consumers to build code

Consumers install Accretion; they do not compile it. Releases ship prebuilt
binaries per supported target (GB10/arm64 CUDA first; others as they are proven).
The setup downloader fetches and prepares models — download, repack, optimize —
as one command. `make` is for contributors, never a setup step.

## 2. Never require complex custom loading of agents

A consumer's existing agent harness must work by pointing it at the box — one
base URL, one token. Claude CLI, OpenAI-compatible clients, and anything speaking
the standard APIs connect without bespoke adapters, custom wrappers, or config
ceremony. If a feature needs harness-side changes to be usable, it is not done:
the server side must carry the complexity (examples: KV pinning is transparent
prefix recognition, not a client API; the capability endpoint describes the model
honestly to standard clients rather than requiring a custom probe).
