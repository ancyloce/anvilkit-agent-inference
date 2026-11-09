"""Bootstrap of anvilkit-agent-inference (DD-09 §3, platform.md readiness).

The first configuration generation is validated (a rejected one starts
nothing), the HTTP listener binds and answers liveness at once, the locked
weights load off the event loop and a probe embedding and rerank must
succeed before readiness is reported. SIGTERM withdraws readiness, drains
in-flight requests within the configured bound and stops the batchers. Under
tls.mode tls the listener serves HTTPS and a watcher reloads the certificate
and key into the listener's context when cert-manager renews them; new
handshakes use the new pair, a half-written pair keeps the old one. Under
tls.mode mtls (RB-A17) the handshake also requires a client certificate from
the configured CA, and a connection whose SPIFFE ID is not allowed is closed
before any of its bytes reach the application; the kubelet's probes are then
served by a separate plaintext probe listener (probe.listen).
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
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import uvicorn
from uvicorn.protocols.http.h11_impl import H11Protocol

from . import compute
from .app import Engine, Metrics, create_app
from .config import ConfigError, Inference, load

log = logging.getLogger("anvilkit.inference")


class JsonFormatter(logging.Formatter):
    FIELDS = ("requestId", "route", "code", "seconds", "generation", "digest", "model", "revision", "drainSeconds", "forced")

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
    next tick; the context keeps serving the previous pair meanwhile. Under
    mtls a changed client CA file is added to the context's trust: a CA that
    is removed stays trusted until the process restarts (SSLContext adds
    verify locations and never drops them)."""

    def __init__(self, server: uvicorn.Server, cert_file: str, key_file: str, interval_s: float = 30.0, ca_file: str = "") -> None:
        self.server, self.cert_file, self.key_file, self.interval_s, self.ca_file = server, cert_file, key_file, interval_s, ca_file
        self.stop = threading.Event()
        self.loaded = self.fingerprint()
        self.reloads = 0

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for f in (self.cert_file, self.key_file, *([self.ca_file] if self.ca_file else [])):
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
            if self.ca_file:
                ctx.load_verify_locations(self.ca_file)
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


def spiffe_id(peercert: dict | None) -> str:
    """The SPIFFE ID of a verified peer certificate: its single spiffe:// URI
    SAN (an X.509-SVID carries exactly one), or "" for none or several."""
    uris = [v for k, v in (peercert or {}).get("subjectAltName", ()) if k == "URI"]
    return uris[0] if len(uris) == 1 and uris[0].startswith("spiffe://") else ""


def peer_checked(allowed: frozenset[str]) -> type[H11Protocol]:
    """uvicorn's h11 protocol admitting only the allowed SPIFFE IDs (RB-A17).
    The TLS handshake has already verified the client certificate against the
    CA when the connection is made; a peer outside the allowlist is closed, and
    any bytes it sent before the close are dropped, never parsed."""

    class PeerChecked(H11Protocol):
        refused = False

        def connection_made(self, transport) -> None:  # type: ignore[override]
            super().connection_made(transport)
            if spiffe_id(transport.get_extra_info("peercert")) not in allowed:
                self.refused = True
                log.warning("mTLS peer refused: not an allowed SPIFFE ID", extra={"code": "PEER_NOT_ALLOWED"})
                transport.close()

        def data_received(self, data: bytes) -> None:
            if not self.refused:
                super().data_received(data)

    return PeerChecked


class ProbeServer:
    """The plaintext probe listener (probe.listen): /healthz and /readyz from
    the engine's state for the kubelet, which presents no client certificate;
    nothing else is served on it."""

    def __init__(self, listen: str, engine: Engine) -> None:
        host, port = listen.rsplit(":", 1)

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/healthz":
                    code, body = 200, b'{"status":"ok"}'
                elif self.path == "/readyz":
                    code, body = (200, b'{"status":"ready"}') if engine.ready else (503, b'{"status":"loading"}')
                else:
                    code, body = 404, b'{"status":"not found"}'
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

        self.httpd = ThreadingHTTPServer((host, int(port)), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="probe-listener", daemon=True)

    def start(self) -> "ProbeServer":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def listener_config(cfg: Inference, app, **extra) -> uvicorn.Config:
    """The main listener for the configured transport: HTTPS (tls), HTTPS
    with a required client certificate and the SPIFFE allowlist (mtls), or
    plaintext (development)."""
    tls: dict = {}
    if cfg.tls.mode in ("tls", "mtls"):
        tls = {"ssl_certfile": cfg.tls.cert_file, "ssl_keyfile": cfg.tls.key_file}
    if cfg.tls.mode == "mtls":
        tls |= {"ssl_ca_certs": cfg.tls.ca_file, "ssl_cert_reqs": ssl.CERT_REQUIRED, "http": peer_checked(frozenset(cfg.tls.allowed_peers))}
    host, port = cfg.listen.rsplit(":", 1)
    return uvicorn.Config(app, host=host, port=int(port), log_config=None, access_log=False, **tls, **extra)


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
    metrics.generation.set(gen.number)
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
    server = uvicorn.Server(listener_config(cfg, app, timeout_graceful_shutdown=cfg.shutdown_timeout_s, limit_concurrency=512))
    reloader = None
    if cfg.tls.mode in ("tls", "mtls"):
        reloader = CertificateReloader(server, cfg.tls.cert_file, cfg.tls.key_file, ca_file=cfg.tls.ca_file if cfg.tls.mode == "mtls" else "")
        threading.Thread(target=reloader.run, name="certificate-reloader", daemon=True).start()
    probes = ProbeServer(cfg.probe.listen, engine).start() if cfg.probe.listen else None
    log.info("listener transport", extra={"code": cfg.tls.mode})
    original = server.handle_exit

    stopping: list[float] = []

    def withdraw(sig: int, frame: object) -> None:
        engine.ready = False
        if not stopping:
            stopping.append(time.monotonic())
        original(sig, frame)

    server.handle_exit = withdraw  # type: ignore[method-assign]
    server.run()
    # RB-A25: the drain time and whether uvicorn's graceful bound ended it
    # (it cancels the requests still running). Logged, not exported: /metrics
    # is served only on the business listener this drain has just closed.
    if stopping:
        drain = time.monotonic() - stopping[0]
        log.info("inference drained", extra={"drainSeconds": round(drain, 3), "forced": drain >= cfg.shutdown_timeout_s})
    if reloader is not None:
        reloader.stop.set()
    if probes is not None:
        probes.stop()
    for b in (engine.embed, engine.rerank):
        if b is not None:
            b.close()
    log.info("inference stopped")
    return 1 if failed.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
