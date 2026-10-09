"""Bootstrap of anvilkit-agent-inference (DD-09 §3, platform.md readiness).

The first configuration generation is validated (a rejected one starts
nothing), the HTTP listener binds and answers liveness at once, the locked
weights load off the event loop and a probe embedding and rerank must
succeed before readiness is reported. SIGTERM withdraws readiness, drains
in-flight requests within the configured bound and stops the batchers. Under
tls.mode tls the listener serves HTTPS and a watcher reloads the certificate
and key into the listener's context when cert-manager renews them; new
handshakes use the new pair, a half-written pair keeps the old one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import ssl
import sys
import threading
from pathlib import Path

import uvicorn

from . import compute
from .app import Engine, Metrics, create_app
from .config import ConfigError, load

log = logging.getLogger("anvilkit.inference")


class JsonFormatter(logging.Formatter):
    FIELDS = ("requestId", "route", "code", "seconds", "generation", "digest", "model", "revision")

    def format(self, record: logging.LogRecord) -> str:
        out = {"level": record.levelname.lower(), "msg": record.getMessage(), "service": "anvilkit-agent-inference"}
        for f in self.FIELDS:
            if hasattr(record, f):
                out[f] = getattr(record, f)
        if record.exc_info and record.exc_info[0] is not None:
            out["error"] = record.exc_info[0].__name__
        return json.dumps(out)


def load_engine(engine: Engine, cfg, metrics: Metrics) -> None:
    """Loads both locked models, probes them and only then marks the engine ready."""
    import torch

    torch.set_num_threads(cfg.torch_threads)
    root = Path(cfg.models_dir)
    locked = compute.read_models_lock(root / "models.lock")
    m3, rr = locked["BAAI/bge-m3"], locked["BAAI/bge-reranker-v2-m3"]
    embed = compute.Bge3(root / m3.directory, cfg.embedding.query_max_tokens, cfg.embedding.passage_max_tokens)
    rerank = compute.Reranker(root / rr.directory, cfg.rerank.max_tokens)
    q = cfg.queue
    engine.embed = compute.Batcher("bge-m3", embed, cfg.embedding.batch_size, q.max_wait_ms, q.max_pending_items, metrics.observe)
    engine.rerank = compute.Batcher("bge-reranker-v2-m3", rerank, cfg.rerank.batch_size, q.max_wait_ms, q.max_pending_items, metrics.observe)
    dense, (indices, _values) = engine.embed.submit([("query", "readiness probe")]).result(timeout=120)[0]
    scores = engine.rerank.submit([("readiness probe", "readiness probe")]).result(timeout=120)
    if len(dense) != 1024 or not indices or len(scores) != 1:
        raise RuntimeError("probe computation returned an unexpected shape")
    engine.dimensions = len(dense)
    engine.embed_revision, engine.rerank_revision = m3.revision, rr.revision
    engine.ready = True
    log.info("models ready", extra={"model": "bge-m3,bge-reranker-v2-m3", "revision": f"{m3.revision},{rr.revision}"})


class CertificateReloader:
    """Polls the certificate placements and loads a changed pair into the
    listener's SSL context (the context uvicorn built at start). A pair that
    does not load (the key written before the certificate) is retried on the
    next tick; the context keeps serving the previous pair meanwhile."""

    def __init__(self, server: uvicorn.Server, cert_file: str, key_file: str, interval_s: float = 30.0) -> None:
        self.server, self.cert_file, self.key_file, self.interval_s = server, cert_file, key_file, interval_s
        self.stop = threading.Event()
        self.loaded = self.fingerprint()
        self.reloads = 0

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for f in (self.cert_file, self.key_file):
            try:
                h.update(Path(f).read_bytes())
            except OSError:
                h.update(b"unreadable")
        return h.hexdigest()

    def check(self) -> bool:
        ctx = getattr(self.server.config, "ssl", None)
        cur = self.fingerprint()
        if ctx is None or cur == self.loaded:
            return False
        try:
            ctx.load_cert_chain(self.cert_file, self.key_file)
        except (OSError, ssl.SSLError):
            log.warning("the renewed certificate pair does not load yet; keeping the previous one")
            return False
        self.loaded = cur
        self.reloads += 1
        log.info("listener certificate reloaded")
        return True

    def run(self) -> None:
        while not self.stop.wait(self.interval_s):
            self.check()


def main() -> int:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    try:
        gen = load(os.environ.get("ANVILKIT_INFERENCE_CONFIG", "config.yaml"))
    except ConfigError as err:
        sys.stderr.write(f"configuration generation 1 rejected: {err}\n")
        return 1
    cfg = gen.config
    log.info("configuration generation active", extra={"generation": gen.number, "digest": gen.digest})
    engine = Engine()
    metrics = Metrics()
    app = create_app(cfg, engine, metrics)
    failed = threading.Event()

    def loader() -> None:
        try:
            load_engine(engine, cfg, metrics)
        except Exception:  # noqa: BLE001
            log.exception("model loading failed; the service stays unready")
            failed.set()
            os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=loader, name="model-loader", daemon=True).start()
    host, port = cfg.listen.rsplit(":", 1)
    tls = {"ssl_certfile": cfg.tls.cert_file, "ssl_keyfile": cfg.tls.key_file} if cfg.tls.mode == "tls" else {}
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=host,
            port=int(port),
            log_config=None,
            access_log=False,
            timeout_graceful_shutdown=cfg.shutdown_timeout_s,
            limit_concurrency=512,
            **tls,
        )
    )
    reloader = None
    if tls:
        reloader = CertificateReloader(server, cfg.tls.cert_file, cfg.tls.key_file)
        threading.Thread(target=reloader.run, name="certificate-reloader", daemon=True).start()
    log.info("listener transport", extra={"code": cfg.tls.mode})
    original = server.handle_exit

    def withdraw(sig: int, frame: object) -> None:
        engine.ready = False
        original(sig, frame)

    server.handle_exit = withdraw  # type: ignore[method-assign]
    server.run()
    if reloader is not None:
        reloader.stop.set()
    for b in (engine.embed, engine.rerank):
        if b is not None:
            b.close()
    log.info("inference stopped")
    return 1 if failed.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
