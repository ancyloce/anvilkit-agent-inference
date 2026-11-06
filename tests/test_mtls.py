"""mTLS listener tests (RB-A17): the configuration rules of tls.mode mtls and,
on real sockets with the fixture certificates (fixtures/mtls), the client
certificate requirement, the SPIFFE allowlist, the plaintext probe listener and
a client CA reload.

    python -m unittest discover -s tests
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import socket
import ssl
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import uvicorn

from anvilkit_inference.app import Engine, Metrics, create_app
from anvilkit_inference.config import ConfigError, load
from anvilkit_inference.main import CertificateReloader, ProbeServer, listener_config, spiffe_id

FIX = Path(__file__).parent / "fixtures" / "mtls"
KNOWLEDGE = "spiffe://anvilkit.test/ns/anvilkit-apps/sa/anvilkit-agent-knowledge"
MCP = "spiffe://anvilkit.test/ns/anvilkit-apps/sa/anvilkit-agent-mcp"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def env(ca: Path = FIX / "ca.crt", cert: Path = FIX / "server.crt", key: Path = FIX / "server.key", **extra: str) -> dict[str, str]:
    return {
        "ANVILKIT_INFERENCE_TLS_CERT_FILE": str(cert),
        "ANVILKIT_INFERENCE_TLS_KEY_FILE": str(key),
        "ANVILKIT_INFERENCE_TLS_CA_FILE": str(ca),
        "ANVILKIT_INFERENCE_PROBE_LISTEN": "127.0.0.1:9128",
        **extra,
    }


class Files:
    """Reviewed configuration files in a temporary directory."""

    def __init__(self) -> None:
        self.dir = tempfile.mkdtemp()

    def __call__(self, text: str) -> str:
        f = Path(self.dir) / f"c{abs(hash(text))}.yaml"
        f.write_text(text)
        return str(f)

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


MTLS_FILE = f"tls:\n  mode: mtls\n  allowed_peers:\n    - {KNOWLEDGE}\n"


class MtlsConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.file = Files()

    def tearDown(self) -> None:
        self.file.close()

    def test_mtls_needs_its_ca_peers_and_probe_listener(self) -> None:
        g = load(self.file(MTLS_FILE), env())
        self.assertEqual(g.config.tls.mode, "mtls")
        self.assertEqual(g.config.tls.allowed_peers, [KNOWLEDGE])
        self.assertEqual(g.config.probe.listen, "127.0.0.1:9128")
        cases = [
            (MTLS_FILE, {k: v for k, v in env().items() if k != "ANVILKIT_INFERENCE_TLS_CA_FILE"}, "ANVILKIT_INFERENCE_TLS_CA_FILE"),
            (MTLS_FILE, {k: v for k, v in env().items() if k != "ANVILKIT_INFERENCE_PROBE_LISTEN"}, "ANVILKIT_INFERENCE_PROBE_LISTEN"),
            (MTLS_FILE, {k: v for k, v in env().items() if k != "ANVILKIT_INFERENCE_TLS_KEY_FILE"}, "ANVILKIT_INFERENCE_TLS_KEY_FILE"),
            ("tls:\n  mode: mtls\n", env(), "tls.allowed_peers"),
            ("tls:\n  mode: mtls\n  allowed_peers: [anvilkit-agent-knowledge]\n", env(), "not a SPIFFE ID"),
            ("tls:\n  mode: mtls\n  allowed_peers: ['spiffe://anvilkit.test']\n", env(), "not a SPIFFE ID"),
            (f"tls:\n  mode: tls\n  allowed_peers: [{KNOWLEDGE}]\n", env(), "only under tls.mode mtls"),
            (MTLS_FILE + "  ca_file: /etc/x/ca.crt\n", env(), "placement"),
            (MTLS_FILE + "probe:\n  listen: 0.0.0.0:9128\n", env(), "placement"),
            (MTLS_FILE, env(ANVILKIT_INFERENCE_LISTEN="127.0.0.1:9128"), "must differ"),
        ]
        for text, environ, want in cases:
            with self.subTest(want=want, text=text):
                with self.assertRaisesRegex(ConfigError, want):
                    load(self.file(text), environ)

    def test_the_peers_are_part_of_the_digest_and_the_placements_are_not(self) -> None:
        g = load(self.file(MTLS_FILE), env())
        self.assertEqual(
            g.digest, load(self.file(MTLS_FILE), env(ca=Path("/other/ca.crt"), ANVILKIT_INFERENCE_PROBE_LISTEN="0.0.0.0:9200")).digest
        )
        two = load(self.file(f"tls:\n  mode: mtls\n  allowed_peers: [{KNOWLEDGE}, {MCP}]\n"), env())
        self.assertNotEqual(g.digest, two.digest, "a widened allowlist is a new generation")
        plain_tls = {k: v for k, v in env().items() if k in ("ANVILKIT_INFERENCE_TLS_CERT_FILE", "ANVILKIT_INFERENCE_TLS_KEY_FILE")}
        self.assertNotEqual(g.digest, load(None, plain_tls).digest, "the transport mode is part of the digest")

    def test_an_apollo_snapshot_never_sets_the_allowlist(self) -> None:
        snap = Path(self.file.dir) / "snapshot.json"
        snap.write_text(
            json.dumps(
                {
                    "appId": "anvilkit-agent-inference",
                    "releaseKey": "r1",
                    "expiresAt": "2999-01-01T00:00:00Z",
                    "configurations": {"tls.allowed_peers": MCP},
                }
            )
        )
        with self.assertRaisesRegex(ConfigError, "reviewed configuration file only"):
            load(self.file(MTLS_FILE + "apollo:\n  mode: snapshot\n"), env(ANVILKIT_INFERENCE_APOLLO_SNAPSHOT_FILE=str(snap)))

    def test_tls_and_development_keep_their_rules(self) -> None:
        plain_tls = {k: v for k, v in env().items() if k in ("ANVILKIT_INFERENCE_TLS_CERT_FILE", "ANVILKIT_INFERENCE_TLS_KEY_FILE")}
        self.assertEqual(load(None, plain_tls).config.tls.mode, "tls")
        self.assertEqual(
            load(None, {**plain_tls, "ANVILKIT_INFERENCE_PROBE_LISTEN": "127.0.0.1:9128"}).config.probe.listen, "127.0.0.1:9128"
        )
        dev = load(self.file("tls:\n  mode: development\ndevelopment:\n  enabled: true\n"), {})
        self.assertEqual(dev.config.probe.listen, "")


class SpiffeIdTest(unittest.TestCase):
    def test_exactly_one_spiffe_uri(self) -> None:
        self.assertEqual(spiffe_id({"subjectAltName": (("DNS", "knowledge"), ("URI", KNOWLEDGE))}), KNOWLEDGE)
        for cert in (
            None,
            {},
            {"subjectAltName": (("DNS", "knowledge"),)},
            {"subjectAltName": (("URI", KNOWLEDGE), ("URI", MCP))},
            {"subjectAltName": (("URI", "https://knowledge.example/"),)},
        ):
            with self.subTest(cert=cert):
                self.assertEqual(spiffe_id(cert), "")


class Counting:
    """An ASGI wrapper counting the HTTP requests that reach the application."""

    def __init__(self, app) -> None:
        self.app, self.calls = app, 0

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            self.calls += 1
        await self.app(scope, receive, send)


class MtlsListenerTest(unittest.TestCase):
    """tls.mode mtls on a real socket: only Knowledge's certificate from the
    configured CA is served; every other client is refused before the
    application sees a request, and the listener keeps serving."""

    def setUp(self) -> None:
        self.file = Files()
        self.dir = Path(self.file.dir)
        for f in ("server.crt", "server.key", "ca.crt"):
            shutil.copy(FIX / f, self.dir / f)
        self.port = free_port()
        cfg = load(
            self.file(MTLS_FILE),
            env(
                ca=self.dir / "ca.crt",
                cert=self.dir / "server.crt",
                key=self.dir / "server.key",
                ANVILKIT_INFERENCE_LISTEN=f"127.0.0.1:{self.port}",
            ),
        ).config
        self.app = Counting(create_app(cfg, Engine(), Metrics()))
        self.server = uvicorn.Server(listener_config(cfg, self.app, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("server did not start")
        self.cfg = cfg

    def tearDown(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)
        self.file.close()

    def get(self, client: str | None) -> int:
        ctx = ssl.create_default_context(cafile=str(FIX / "ca.crt"))
        if client:
            ctx.load_cert_chain(str(FIX / f"{client}.crt"), str(FIX / f"{client}.key"))
        conn = http.client.HTTPSConnection("127.0.0.1", self.port, context=ctx, timeout=10)
        try:
            conn.request("GET", "/healthz")
            return conn.getresponse().status
        finally:
            conn.close()

    def test_only_the_allowed_peer_is_served(self) -> None:
        self.assertEqual(self.get("knowledge"), 200)
        self.assertEqual(self.app.calls, 1)
        for client in ("mcp", "no-uri", "two-uris", "foreign", None):
            with self.subTest(client=client):
                with self.assertRaises(OSError):  # a closed connection, a reset or the TLS alert (ssl.SSLError)
                    self.get(client)
        with self.assertRaises(OSError):  # plaintext to the mTLS port
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/healthz", timeout=10)
        self.assertEqual(self.app.calls, 1, "no refused client's request reached the application")
        self.assertEqual(self.get("knowledge"), 200, "the listener keeps serving after the refusals")

    def test_a_reloaded_client_ca_is_trusted_for_new_handshakes(self) -> None:
        reloader = CertificateReloader(self.server, self.cfg.tls.cert_file, self.cfg.tls.key_file, ca_file=self.cfg.tls.ca_file)
        with self.assertRaises(OSError):
            self.get("foreign")
        self.assertFalse(reloader.check(), "unchanged files are not reloaded")
        shutil.copy(FIX / "other-ca.crt", self.cfg.tls.ca_file)
        self.assertTrue(reloader.check())
        self.assertEqual(self.get("foreign"), 200, "Knowledge's identity from the newly trusted CA")
        self.assertEqual(self.get("knowledge"), 200, "a removed CA stays trusted until a restart (SSLContext only adds)")


class ProbeServerTest(unittest.TestCase):
    """The plaintext probe listener answers the kubelet's two paths from the
    engine's state and serves nothing else."""

    def setUp(self) -> None:
        self.engine = Engine()
        self.probes = ProbeServer("127.0.0.1:0", self.engine).start()

    def tearDown(self) -> None:
        self.probes.stop()

    def call(self, path: str, method: str = "GET") -> int:
        req = urllib.request.Request(f"http://127.0.0.1:{self.probes.port}{path}", method=method, data=b"{}" if method == "POST" else None)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    def test_probes_only(self) -> None:
        self.assertEqual(self.call("/healthz"), 200)
        self.assertEqual(self.call("/readyz"), 503)
        self.engine.ready = True
        self.assertEqual(self.call("/readyz"), 200)
        self.assertEqual(self.call("/metrics"), 404)
        self.assertEqual(self.call("/api/v1/embeddings", "POST"), 501, "no compute route on the probe listener")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    unittest.main()
