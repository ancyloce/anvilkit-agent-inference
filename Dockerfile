# syntax=docker/dockerfile:1
# anvilkit-agent-inference (DD-07 §1, P15). Build from this directory with
# the contracts repository as a named context (the generated pydantic
# models of openapi/inference.yaml):
#   docker build --build-context contracts=../../../contracts -t anvilkit-agent-inference:dev .
# Dependencies come from the hash-locked requirements.lock; the weights of
# models.lock are fetched once here at their revisions and SHA-256 sums. At
# run time the service downloads nothing and runs as UID 10001 on a
# read-only root; readiness follows weight load and a probe computation.
ARG PYTHON_IMAGE=python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c

FROM ${PYTHON_IMAGE} AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
COPY requirements.lock /tmp/requirements.lock
RUN /opt/venv/bin/pip install --require-hashes --no-deps -r /tmp/requirements.lock \
      --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
# The weights depend only on the locked dependencies and models.lock, so a
# contracts or source change reuses the cached download.
COPY models.lock bin/fetch-models.py /tmp/
# Plain HTTPS downloads (the Xet client broke mid-transfer on this link
# twice); every file is still verified against its locked SHA-256.
RUN HF_HUB_DISABLE_XET=1 /opt/venv/bin/python /tmp/fetch-models.py /tmp/models.lock /opt/models
COPY --from=contracts python/anvilkit_generated_clients /opt/anvilkit/src/anvilkit_generated_clients
COPY src /opt/anvilkit/src
RUN /opt/venv/bin/python -m compileall -q /opt/anvilkit/src \
 && { /opt/venv/bin/python -m compileall -q /opt/venv >/dev/null 2>&1 || true; }

FROM ${PYTHON_IMAGE}
# Debian security fixes the base digest predates (CRITICAL with a fixed
# version: CVE-2026-31789 OpenSSL, CVE-2026-33845/CVE-2026-42010 GnuTLS),
# pinned to the fixed versions; the build stage and its weights are unaffected.
RUN apt-get update \
 && apt-get install -y --no-install-recommends --only-upgrade libssl3=3.0.22-1~deb12u1 openssl=3.0.22-1~deb12u1 libgnutls30=3.7.9-2+deb12u7 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/models /opt/models
COPY --from=build /opt/anvilkit/src /opt/anvilkit/src
COPY config.yaml /opt/anvilkit/config.yaml
COPY tests /opt/anvilkit/tests
ENV PYTHONPATH=/opt/anvilkit/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TQDM_DISABLE=1 \
    ANVILKIT_INFERENCE_CONFIG=/opt/anvilkit/config.yaml \
    ANVILKIT_INFERENCE_MODELS_DIR=/opt/models \
    HOME=/tmp
USER 10001:10001
WORKDIR /opt/anvilkit
EXPOSE 9108
ENTRYPOINT ["/opt/venv/bin/python", "-m", "anvilkit_inference.main"]
