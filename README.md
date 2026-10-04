# anvilkit-agent-inference

The Inference service of the AnvilKit Agent platform (architecture V4.0, [DD-07 §1](https://github.com/ancyloce/anvilkit-services/blob/main/docs/architecture/knowledge.md), delivery.md P15): Python 3.12, FastAPI and FlagEmbedding over [`openapi/inference.yaml`](https://github.com/ancyloce/anvilkit-agent-contracts/blob/main/openapi/inference.yaml) of the contracts repository. It computes, and only computes: fixed BGE-M3 dense (1024, normalized) and sparse vectors and fixed bge-reranker-v2-m3 scores. It accepts no model path, download URL, code, Qdrant connection or business identity, reads no database, holds no secret, and Knowledge is its only caller.

## Contract

- `POST /api/v1/embeddings` (profile `bge-m3-v1`) and `POST /api/v1/rerankings` (profile `bge-reranker-v2-m3-v1`). A request names its task compute identity and the contract's `inputDigest`: SHA-256 over the length-prefixed UTF-8 parts (profile id, input kind or query, every input or candidate id and text). The service recomputes it and refuses a mismatch; the answer echoes it with the locked `modelRevision`, so Knowledge refuses a stale or substituted answer.
- Bodies are read within `max_body_bytes`, parsed strictly (duplicate keys, NaN and trailing data refused) and validated by the generated pydantic models in strict JSON mode (unknown fields refused). Size bounds (inputs, candidates, characters) answer 413 `CAPACITY_EXHAUSTED`; a full compute queue answers 503 `CAPACITY_EXHAUSTED` (retryable); another profile answers 400 `PROFILE_UNQUALIFIED`; every error is the contract's envelope.
- `GET /healthz` (liveness), `GET /readyz` (200 only after both models loaded and a probe embedding and rerank succeeded), `GET /metrics` (requests by route/code, items and micro-batch sizes/time by model, readiness; no text).

## Weights and dependencies

`models.lock` pins `BAAI/bge-m3` at `5617a9f61b028005a4858fdac845db406aefb181` (MIT) and `BAAI/bge-reranker-v2-m3` at `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` (Apache-2.0) with the SHA-256 of every file loaded; the image build fetches exactly these (`bin/fetch-models.py`) and fails on any difference, and the service runs with `HF_HUB_OFFLINE=1`. `requirements.lock` is the hash-pinned dependency set (`uv pip compile --generate-hashes`, CPU torch 2.14.0, FlagEmbedding 1.4.2, FastAPI 0.141.1, transformers 5.17.0), installed with `--require-hashes`; the generated contract models come from the contracts repository (named build context).

## Configuration

`config.yaml` (bounds and fixed profiles) < a validated Apollo snapshot (`apollo.mode: snapshot`) < the allowlisted environment `ANVILKIT_INFERENCE_{LISTEN,MODELS_DIR,APOLLO_SNAPSHOT_FILE,CONFIG}`; any other variable or unknown key rejects the generation (the pattern of the parent's `packages/profile-schemas/python/config_generation.py`). One generation is built at start; a changed input takes effect through a rolling restart.

## Checks

`python -m unittest discover -s tests` (with `PYTHONPATH=src:<contracts checkout>/python` and the locked environment, or inside the image: `docker run --rm --network none --entrypoint /opt/venv/bin/python anvilkit-agent-inference:dev -m unittest discover -s tests`); `ANVILKIT_INFERENCE_MODELS_DIR` names the locked weights for the real-model test. `docker build --build-context contracts=<contracts checkout> -t anvilkit-agent-inference:dev .` (`../../../contracts` in the parent checkout); `helm lint deploy/chart`.

## Limits

CPU only; no GPU profile, throughput or two-replica residency measurement (ENV-04). The micro-batcher runs one model thread per model per replica; measured batch sizes and latency are development observations, not capacity. Workload mTLS is ENV-03 (the listener is plaintext like the other new services).

## Repository

This service is the repository `anvilkit-agent-inference`, mounted in the parent `anvilkit-services` as the submodule `services/agent/inference`. It keeps no contract definitions or generated code of its own: the generated pydantic models come from the contracts repository as the image's named `contracts` build context. CI (`.github/workflows/ci.yml`) checks out that repository at the commit the parent pins (`063f91f`, tag `go/v0.1.4`), builds the image, runs the tests inside it with no network (the real-model tests use the image's locked weights) and lints and renders the chart.
