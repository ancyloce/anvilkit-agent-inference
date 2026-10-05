"""The compute HTTP surface of anvilkit-agent-inference (contracts/openapi/inference.yaml).

POST /api/v1/embeddings and POST /api/v1/rerankings only. Bodies are read
within max_body_bytes, parsed strictly (duplicate keys, NaN and trailing data
refused) and validated by the generated contract models in strict JSON mode
(unknown fields refused: no model path, URL or connection can be named). The
compute profile must be the one this service serves, the input digest is
recomputed and bound into the answer with the locked model revision, and
the size bounds of the active generation apply before any model work. The
service is ready only after both models are loaded and a probe computation
succeeded. Logs carry request ids, routes, codes and counts; never text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from anvilkit_generated_clients.inference import (
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorEnvelope,
    RerankRequest,
    RerankResponse,
)
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .compute import Batcher, Full, input_digest
from .config import Inference

log = logging.getLogger("anvilkit.inference")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.retryable = status, code, message, retryable


@dataclass
class Engine:
    """What the routes compute with; set once loading and the probe succeeded."""

    embed: Batcher | None = None
    rerank: Batcher | None = None
    embed_revision: str = ""
    rerank_revision: str = ""
    dimensions: int = 0
    ready: bool = False


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "anvilkit_inference_requests_total", "Compute requests by route and outcome code.", ["route", "code"],
            registry=self.registry,
        )
        self.items = Counter(
            "anvilkit_inference_items_total", "Inputs or candidates computed by model.", ["model"], registry=self.registry
        )
        self.batch = Histogram(
            "anvilkit_inference_batch_items", "Items per executed micro-batch.", ["model"],
            buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256), registry=self.registry,
        )
        self.batch_seconds = Histogram(
            "anvilkit_inference_batch_seconds", "Model time per executed micro-batch.", ["model"], registry=self.registry
        )
        self.ready = Gauge("anvilkit_inference_ready", "1 after the weights loaded and the probe passed.", registry=self.registry)

    def observe(self, model: str, items: int, seconds: float) -> None:
        self.items.labels(model).inc(items)
        self.batch.labels(model).observe(items)
        self.batch_seconds.labels(model).observe(seconds)


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _strict_object(body: bytes) -> None:
    """Duplicate keys and non-finite numbers are refused before the model validation."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: dict[str, Any] = {}
        for k, v in items:
            if k in seen:
                raise ValueError(f"duplicate key {k!r}")
            seen[k] = v
        return seen

    try:
        value = json.loads(body.decode("utf-8"), object_pairs_hook=pairs, parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError) as err:
        raise ApiError(400, "INVALID_ARGUMENT", f"body is not strict JSON: {str(err)[:200]}") from None
    if not isinstance(value, dict):
        raise ApiError(400, "INVALID_ARGUMENT", "body is not a JSON object")


def create_app(cfg: Inference, engine: Engine, metrics: Metrics | None = None) -> FastAPI:
    metrics = metrics or Metrics()
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

    def envelope(err: ApiError, request_id: str) -> Response:
        body = ErrorEnvelope.model_validate(
            {"error": {"code": err.code, "message": err.message[:512], "requestId": request_id, "retryable": err.retryable}}
        )
        return Response(body.model_dump_json(), status_code=err.status, media_type="application/json")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        code = "INVALID_ARGUMENT"
        return envelope(ApiError(exc.status_code, code, "no such operation"), f"req-{uuid.uuid4().hex}")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> Response:
        return envelope(ApiError(400, "INVALID_ARGUMENT", "invalid request"), f"req-{uuid.uuid4().hex}")

    async def body_of(request: Request) -> bytes:
        if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
            raise ApiError(400, "INVALID_ARGUMENT", "content type must be application/json")
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > cfg.max_body_bytes:
            raise ApiError(413, "CAPACITY_EXHAUSTED", f"body over {cfg.max_body_bytes} bytes")
        parts = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > cfg.max_body_bytes:
                raise ApiError(413, "CAPACITY_EXHAUSTED", f"body over {cfg.max_body_bytes} bytes")
            parts.append(chunk)
        return b"".join(parts)

    async def compute(batcher: Batcher, items: list[Any]) -> list[Any]:
        try:
            fut = batcher.submit(items)
        except Full:
            raise ApiError(503, "CAPACITY_EXHAUSTED", "compute queue is full", retryable=True) from None
        try:
            return await asyncio.wait_for(asyncio.wrap_future(fut), cfg.queue.request_timeout_ms / 1000)
        except asyncio.TimeoutError:
            raise ApiError(503, "CAPACITY_EXHAUSTED", "computation exceeded the request timeout", retryable=True) from None

    async def handle(route: str, request: Request, fn: Callable[[bytes], Any]) -> Response:
        request_id = f"req-{uuid.uuid4().hex}"
        started = time.monotonic()
        try:
            if not engine.ready:
                raise ApiError(503, "DEPENDENCY_UNAVAILABLE", "models are not loaded", retryable=True)
            body = await body_of(request)
            _strict_object(body)
            out = await fn(body)
            metrics.requests.labels(route, "OK").inc()
            log.info("computed", extra={"requestId": request_id, "route": route, "seconds": round(time.monotonic() - started, 3)})
            return Response(out.model_dump_json(), media_type="application/json")
        except ApiError as err:
            metrics.requests.labels(route, err.code).inc()
            log.info("refused", extra={"requestId": request_id, "route": route, "code": err.code})
            return envelope(err, request_id)
        except Exception:  # noqa: BLE001 — the model failed; nothing about the input is echoed
            metrics.requests.labels(route, "DEPENDENCY_UNAVAILABLE").inc()
            log.exception("computation failed", extra={"requestId": request_id, "route": route})
            return envelope(ApiError(503, "DEPENDENCY_UNAVAILABLE", "computation failed", retryable=True), request_id)

    def validated(model: Any, body: bytes) -> Any:
        try:
            return model.model_validate_json(body, strict=True)
        except ValidationError as err:
            first = err.errors()[0]
            where = ".".join(str(x) for x in first.get("loc", ()))
            raise ApiError(400, "INVALID_ARGUMENT", f"{where}: {first.get('type', 'invalid')}") from None

    async def embeddings(body: bytes) -> EmbeddingResponse:
        req: EmbeddingRequest = validated(EmbeddingRequest, body)
        e = cfg.embedding
        if req.compute.profileId != e.profile_id:
            raise ApiError(400, "PROFILE_UNQUALIFIED", f"this service computes profile {e.profile_id}")
        texts = [i.root for i in req.inputs]
        kind = req.inputKind.value
        if input_digest([req.compute.profileId, kind, *texts]) != req.inputDigest:
            raise ApiError(400, "INVALID_ARGUMENT", "inputDigest does not match the inputs")
        if len(texts) > e.max_inputs:
            raise ApiError(413, "CAPACITY_EXHAUSTED", f"more than {e.max_inputs} inputs")
        if sum(len(t) for t in texts) > e.max_request_chars:
            raise ApiError(413, "CAPACITY_EXHAUSTED", f"more than {e.max_request_chars} characters")
        assert engine.embed is not None
        results = await compute(engine.embed, [(kind, t) for t in texts])
        return EmbeddingResponse.model_validate(
            {
                "modelId": "bge-m3",
                "modelRevision": engine.embed_revision,
                "dimensions": engine.dimensions,
                "inputDigest": req.inputDigest,
                "dense": [r[0] for r in results],
                "sparse": [{"indices": r[1][0], "values": r[1][1]} for r in results],
            }
        )

    async def rerankings(body: bytes) -> RerankResponse:
        req: RerankRequest = validated(RerankRequest, body)
        r = cfg.rerank
        if req.compute.profileId != r.profile_id:
            raise ApiError(400, "PROFILE_UNQUALIFIED", f"this service computes profile {r.profile_id}")
        parts = [req.compute.profileId, req.query]
        for c in req.candidates:
            parts.extend([c.candidateId, c.text])
        if input_digest(parts) != req.inputDigest:
            raise ApiError(400, "INVALID_ARGUMENT", "inputDigest does not match the query and candidates")
        ids = [c.candidateId for c in req.candidates]
        if len(set(ids)) != len(ids):
            raise ApiError(400, "INVALID_ARGUMENT", "candidate ids repeat")
        if len(ids) > r.max_candidates:
            raise ApiError(413, "CAPACITY_EXHAUSTED", f"more than {r.max_candidates} candidates")
        if len(req.query) + sum(len(c.text) for c in req.candidates) > r.max_request_chars:
            raise ApiError(413, "CAPACITY_EXHAUSTED", f"more than {r.max_request_chars} characters")
        assert engine.rerank is not None
        scores = await compute(engine.rerank, [(req.query, c.text) for c in req.candidates])
        return RerankResponse.model_validate(
            {
                "modelId": "bge-reranker-v2-m3",
                "modelRevision": engine.rerank_revision,
                "inputDigest": req.inputDigest,
                "scores": [{"candidateId": cid, "score": s} for cid, s in zip(ids, scores, strict=True)],
            }
        )

    @app.post("/api/v1/embeddings")
    async def post_embeddings(request: Request) -> Response:
        return await handle("embeddings", request, embeddings)

    @app.post("/api/v1/rerankings")
    async def post_rerankings(request: Request) -> Response:
        return await handle("rerankings", request, rerankings)

    @app.get("/healthz")
    async def healthz() -> Response:
        return Response('{"status":"ok"}', media_type="application/json")

    @app.get("/readyz")
    async def readyz() -> Response:
        return Response(
            '{"status":"ready"}' if engine.ready else '{"status":"loading"}',
            status_code=200 if engine.ready else 503,
            media_type="application/json",
        )

    @app.get("/metrics")
    async def metrics_route() -> Response:
        metrics.ready.set(1 if engine.ready else 0)
        return Response(generate_latest(metrics.registry), media_type="text/plain; version=0.0.4")

    app.state.metrics = metrics
    return app
