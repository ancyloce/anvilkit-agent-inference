# anvilkit-agent-inference

The Inference service of the AnvilKit Agent platform (architecture V4.0, [DD-07 §1](https://github.com/ancyloce/anvilkit-services/blob/main/docs/architecture/knowledge.md), delivery.md P15): Python 3.12, FastAPI and FlagEmbedding over [`openapi/inference.yaml`](https://github.com/ancyloce/anvilkit-agent-contracts/blob/main/openapi/inference.yaml) of the contracts repository. It computes, and only computes: fixed BGE-M3 dense (1024, normalized) and sparse vectors and fixed bge-reranker-v2-m3 scores. It accepts no model path, download URL, code, Qdrant connection or business identity, reads no database, holds no secret beyond its own TLS key, and Knowledge is its only caller.

## Contract

- `POST /api/v1/embeddings` (profile `bge-m3-v1`) and `POST /api/v1/rerankings` (profile `bge-reranker-v2-m3-v1`). A request names its task compute identity and the contract's `inputDigest`: SHA-256 over the length-prefixed UTF-8 parts (profile id, input kind or query, every input or candidate id and text). The service recomputes it and refuses a mismatch; the answer echoes it with the locked `modelRevision`, so Knowledge refuses a stale or substituted answer.
- Bodies are read within `max_body_bytes`, parsed strictly (duplicate keys, NaN and trailing data refused) and validated by the generated pydantic models in strict JSON mode (unknown fields refused). Size bounds (inputs, candidates, characters) answer 413 `CAPACITY_EXHAUSTED`; a full compute queue answers 503 `CAPACITY_EXHAUSTED` (retryable); another profile answers 400 `PROFILE_UNQUALIFIED`; every error is the contract's envelope.
- `GET /healthz` (liveness), `GET /readyz` (200 only after both models loaded and a probe embedding and rerank succeeded), `GET /metrics` (requests by route/code, items and micro-batch sizes/time by model, readiness, the active configuration generation; no text).

## Transport

- `tls.mode tls` (the default) serves HTTPS with the workload certificate and authenticates no client: the NetworkPolicy decides who may call (Knowledge only). It is DEVELOPMENT_ONLY (RB-A17): the loader and the chart accept it only with `development.enabled`, which the image's default `config.yaml` sets for the development foundation. Outside development the mode is `mtls`, and the chart's default fails until the environment sets it.
- `tls.mode mtls` (RB-A17) also requires a client certificate issued by `ANVILKIT_INFERENCE_TLS_CA_FILE`. It admits only the SPIFFE IDs of `tls.allowed_peers`, which comes from the reviewed file only, never from Apollo; the ID is the certificate's single `spiffe://` URI SAN. A peer outside the list is closed before any of its bytes reach the application.
- The kubelet's probes, which present no certificate, then go to the plaintext `probe.listen` (`ANVILKIT_INFERENCE_PROBE_LISTEN`; `/healthz` and `/readyz` only), which `mtls` requires. The chart sets it with `listen.probePort`.
- A renewed client CA is added to the listener's trust without a restart; a removed CA stays trusted until the next restart.
- `tls.mode development` is plaintext and needs `development.enabled`.

## Weights and dependencies

`models.lock` pins `BAAI/bge-m3` at `5617a9f61b028005a4858fdac845db406aefb181` (MIT) and `BAAI/bge-reranker-v2-m3` at `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` (Apache-2.0) with the SHA-256 of every file loaded; the image build fetches exactly these (`bin/fetch-models.py`) and fails on any difference, and the service runs with `HF_HUB_OFFLINE=1`. `requirements.lock` is the hash-pinned dependency set (`uv pip compile --generate-hashes`, CPU torch 2.14.0, FlagEmbedding 1.4.2, FastAPI 0.141.1, transformers 5.17.0), installed with `--require-hashes`; the generated contract models come from the contracts repository (named build context).

## Configuration

`config.yaml` (bounds and fixed profiles) < a validated Apollo snapshot (`apollo.mode: snapshot`) < the allowlisted environment `ANVILKIT_INFERENCE_{LISTEN,MODELS_DIR,APOLLO_SNAPSHOT_FILE,CONFIG,TLS_CERT_FILE,TLS_KEY_FILE,TLS_CA_FILE,PROBE_LISTEN}`; any other variable or unknown key rejects the generation (the pattern of the parent's `packages/profile-schemas/python/config_generation.py`). One generation is built at start; a changed input takes effect through a rolling restart.

## Checks

`python -m unittest discover -s tests` (with `PYTHONPATH=src:<contracts checkout>/python` and the locked environment, or inside the image: `docker run --rm --network none --entrypoint /opt/venv/bin/python anvilkit-agent-inference:dev -m unittest discover -s tests`); `ANVILKIT_INFERENCE_MODELS_DIR` names the locked weights for the real-model test. `docker build --build-context contracts=<contracts checkout> -t anvilkit-agent-inference:dev .` (`../../../contracts` in the parent checkout); the chart with the settings of the CI `chart` job: `helm lint deploy/chart --set tls.certificate.issuerRef.name=ci-issuer --set image.digest=sha256:0000000000000000000000000000000000000000000000000000000000000001 --set tls.mode=mtls --set 'tls.allowedPeers[0]=spiffe://ci.example/ns/anvilkit-apps/sa/anvilkit-agent-knowledge' --set listen.probePort=9129` (outside development the chart needs a certificate issuer, `image.digest`, `tls.mode mtls`, `tls.allowedPeers` and `listen.probePort`; `--set development.enabled=true` also renders the `tls` default and an image tag).

## Limits

CPU only; no GPU profile, throughput or two-replica residency measurement (ENV-04). The micro-batcher runs one model thread per model per replica; measured batch sizes and latency are development observations, not capacity. The development placements (Compose, the development foundation) run `tls.mode tls` under the development guard. The parent's qualification values run `mtls`, admitting Knowledge's SPIFFE ID, and its NetworkPolicy admits the nodes to the probe port only (RB-A17 Phase B). The production CA and trust domain are ENV-03.

## Repository

This service is the repository `anvilkit-agent-inference`, mounted in the parent `anvilkit-services` as the submodule `services/agent/inference`. It keeps no contract definitions or generated code of its own: the generated pydantic models come from the contracts repository as the image's named `contracts` build context. CI (`.github/workflows/ci.yml`) checks out that repository at the commit its checkout step pins (`b3fe869`, tag `go/v0.1.18`; not necessarily the contracts commit the parent's gitlink records), builds the image, runs the tests inside it with no network (the real-model tests use the image's locked weights) and lints and renders the chart.
