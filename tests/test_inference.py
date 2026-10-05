"""Inference service tests: the compute HTTP contract with fixed fake models on a
real socket, and (when ANVILKIT_INFERENCE_MODELS_DIR names the locked weights)
the real BGE-M3 and bge-reranker-v2-m3 computations.

    python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import uvicorn

from anvilkit_inference import compute
from anvilkit_inference.app import Engine, Metrics, create_app
from anvilkit_inference.config import ConfigError, Inference, load

# The cross-language vectors of the contract's digest rule (Knowledge's
# TypeScript client test asserts the same values).
EMBED_VECTOR = (["bge-m3-v1", "query", "Brand colors?", "Teal — é 雪"], "sha256:ad8549a0ed7828ae7df38176852c34b7354cd584ac6e1df9c67e1817b1ec09f1")
RERANK_VECTOR = (
    ["bge-reranker-v2-m3-v1", "colors?", "c1", "Teal and slate.", "c2", "Inter"],
    "sha256:b0d329006c5de2e239ced3dc64b56387241e3358824504e5e2bbfa98ecaa3ee9",
)


class FakeEmbed:
    def __init__(self, hold: float = 0.0) -> None:
        self.hold = hold
        self.batches: list[int] = []

    def run(self, items):
        self.batches.append(len(items))
        time.sleep(self.hold)
        return [([float(len(t)) / 100.0] * 1024, ([3, 7], [0.5, 0.25])) for _, t in items]


class FakeRerank:
    def run(self, items):
        return [float(len(t)) for _, t in items]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, cfg: Inference, engine: Engine) -> None:
        self.port = free_port()
        self.server = uvicorn.Server(uvicorn.Config(create_app(cfg, engine, Metrics()), host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("server did not start")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)

    def call(self, path: str, body: bytes | str | dict, content_type: str = "application/json", method: str = "POST"):
        data = body if isinstance(body, bytes) else (body.encode() if isinstance(body, str) else json.dumps(body).encode())
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data if method == "POST" else None,
                                     method=method, headers={"Content-Type": content_type})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")


def embed_body(inputs=("Brand colors?",), kind="query", profile="bge-m3-v1", digest=None):
    return {
        "compute": {"taskId": "query-1", "generation": "1", "profileId": profile},
        "inputKind": kind,
        "inputs": list(inputs),
        "inputDigest": digest or compute.input_digest([profile, kind, *inputs]),
    }


def rerank_body(query="colors?", candidates=(("c1", "Teal and slate."), ("c2", "Inter")), profile="bge-reranker-v2-m3-v1"):
    parts = [profile, query]
    for cid, text in candidates:
        parts += [cid, text]
    return {
        "compute": {"taskId": "search-1", "generation": "1", "profileId": profile},
        "query": query,
        "candidates": [{"candidateId": c, "text": t} for c, t in candidates],
        "inputDigest": compute.input_digest(parts),
    }


class ContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Inference.model_validate(
            {"embedding": {"max_inputs": 4, "max_request_chars": 1000}, "rerank": {"max_candidates": 4},
             "queue": {"max_pending_items": 8, "max_wait_ms": 20, "request_timeout_ms": 2000}, "max_body_bytes": 8192}
        )
        self.fake = FakeEmbed()
        self.engine = Engine()
        self.server = Server(self.cfg, self.engine)

    def tearDown(self) -> None:
        self.server.stop()
        for b in (self.engine.embed, self.engine.rerank):
            if b:
                b.close()

    def ready(self, embed=None) -> None:
        self.engine.embed = compute.Batcher("bge-m3", embed or self.fake, 4, 20, 8)
        self.engine.rerank = compute.Batcher("bge-reranker-v2-m3", FakeRerank(), 4, 0, 8)
        self.engine.embed_revision, self.engine.rerank_revision = "5617a9f", "953dc6f"
        self.engine.dimensions = 1024
        self.engine.ready = True

    def test_digest_vectors(self) -> None:
        self.assertEqual(compute.input_digest(EMBED_VECTOR[0]), EMBED_VECTOR[1])
        self.assertEqual(compute.input_digest(RERANK_VECTOR[0]), RERANK_VECTOR[1])

    def test_not_ready_until_loaded(self) -> None:
        status, body = self.server.call("/api/v1/embeddings", embed_body())
        self.assertEqual((status, body["error"]["code"], body["error"]["retryable"]), (503, "DEPENDENCY_UNAVAILABLE", True))
        self.assertEqual(self.server.call("/readyz", b"", method="GET")[0], 503)
        self.assertEqual(self.server.call("/healthz", b"", method="GET")[0], 200)
        self.ready()
        self.assertEqual(self.server.call("/readyz", b"", method="GET")[0], 200)

    def test_embeddings_bind_digest_and_revision(self) -> None:
        self.ready()
        body = embed_body(("a", "bb"), kind="passage")
        status, out = self.server.call("/api/v1/embeddings", body)
        self.assertEqual(status, 200, out)
        self.assertEqual(out["inputDigest"], body["inputDigest"])
        self.assertEqual((out["modelId"], out["modelRevision"], out["dimensions"]), ("bge-m3", "5617a9f", 1024))
        self.assertEqual(len(out["dense"]), 2)
        self.assertEqual(out["sparse"][0], {"indices": [3, 7], "values": [0.5, 0.25]})

    def test_rerankings(self) -> None:
        self.ready()
        status, out = self.server.call("/api/v1/rerankings", rerank_body())
        self.assertEqual(status, 200, out)
        self.assertEqual([s["candidateId"] for s in out["scores"]], ["c1", "c2"])
        self.assertEqual(out["modelRevision"], "953dc6f")
        dup = rerank_body(candidates=(("c1", "a"), ("c1", "b")))
        self.assertEqual(self.server.call("/api/v1/rerankings", dup)[1]["error"]["code"], "INVALID_ARGUMENT")

    def test_refusals(self) -> None:
        self.ready()
        good = embed_body()
        cases = [
            ("digest mismatch", embed_body(digest="sha256:" + "0" * 64), 400, "INVALID_ARGUMENT"),
            ("another profile", embed_body(profile="e5-large-v1"), 400, "PROFILE_UNQUALIFIED"),
            ("model path named", {**good, "modelPath": "/tmp/evil"}, 400, "INVALID_ARGUMENT"),
            ("download URL named", {**good, "compute": {**good["compute"], "url": "https://x"}}, 400, "INVALID_ARGUMENT"),
            ("generation as number", {**good, "compute": {**good["compute"], "generation": 1}}, 400, "INVALID_ARGUMENT"),
            ("unknown input kind", embed_body(kind="document"), 400, "INVALID_ARGUMENT"),
            ("too many inputs", embed_body(("a",) * 5), 413, "CAPACITY_EXHAUSTED"),
            ("too many characters", embed_body(("x" * 600, "y" * 600)), 413, "CAPACITY_EXHAUSTED"),
            ("too many candidates", rerank_body(candidates=tuple((f"c{i}", "t") for i in range(5))), 413, "CAPACITY_EXHAUSTED"),
        ]
        for name, body, status, code in cases:
            with self.subTest(name):
                path = "/api/v1/rerankings" if "candidates" in body else "/api/v1/embeddings"
                got, out = self.server.call(path, body)
                self.assertEqual((got, out["error"]["code"]), (status, code))
                self.assertTrue(out["error"]["requestId"].startswith("req-"))
        raw = json.dumps(good)
        for name, text, ct, status in [
            ("duplicate key", raw[:-1] + ', "inputKind": "query"}', "application/json", 400),
            ("NaN", raw.replace('"generation": "1"', '"generation": NaN'), "application/json", 400),
            ("trailing data", raw + " {}", "application/json", 400),
            ("not JSON content", raw, "text/plain", 400),
            ("body over the bound", json.dumps({**good, "pad": "x" * 9000}), "application/json", 413),
        ]:
            with self.subTest(name):
                self.assertEqual(self.server.call("/api/v1/embeddings", text, ct)[0], status)
        status, out = self.server.call("/api/v1/models", good)
        self.assertEqual((status, out["error"]["code"]), (404, "INVALID_ARGUMENT"))

    def test_a_timed_out_request_leaves_the_service_serving(self) -> None:
        self.cfg = self.cfg.model_copy(update={"queue": self.cfg.queue.model_copy(update={"request_timeout_ms": 200})})
        self.server.stop()
        self.server = Server(self.cfg, self.engine)
        self.ready(FakeEmbed(hold=0.5))
        holder = threading.Thread(target=lambda: self.server.call("/api/v1/embeddings", embed_body(("hold",))))
        holder.start()
        time.sleep(0.05)
        status, out = self.server.call("/api/v1/embeddings", embed_body(("queued",)))
        self.assertEqual((status, out["error"]["code"]), (503, "CAPACITY_EXHAUSTED"))
        holder.join()
        for _ in range(100):  # the held batch finishes after its caller gave up
            if self.engine.embed.pending == 0:
                break
            time.sleep(0.05)
        self.assertEqual(self.engine.embed.pending, 0)
        self.engine.embed.model = FakeEmbed()
        status, out = self.server.call("/api/v1/embeddings", embed_body(("after",)))
        self.assertEqual(status, 200, out)

    def test_micro_batches_and_bounded_queue(self) -> None:
        slow = FakeEmbed(hold=0.4)
        self.ready(slow)
        results: list[int] = []

        def one(text: str) -> None:
            results.append(self.server.call("/api/v1/embeddings", embed_body((text,)))[0])

        threads = [threading.Thread(target=one, args=(f"t{i}",)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results, [200, 200, 200])
        self.assertLess(len(slow.batches), 3, "concurrent requests share micro-batches")
        # Fill the pending bound (8) with held work; the next request is refused, not queued.
        big = embed_body(("q1", "q2", "q3", "q4"))
        holders = [threading.Thread(target=lambda: results.append(self.server.call("/api/v1/embeddings", big)[0])) for _ in range(2)]
        for t in holders:
            t.start()
        time.sleep(0.1)
        status, out = self.server.call("/api/v1/embeddings", embed_body(("late",)))
        for t in holders:
            t.join()
        self.assertEqual((status, out["error"]["code"], out["error"]["retryable"]), (503, "CAPACITY_EXHAUSTED", True))


class BatcherTest(unittest.TestCase):
    def test_a_cancelled_request_never_stops_the_worker_thread(self) -> None:
        slow = FakeEmbed(hold=0.3)
        b = compute.Batcher("bge-m3", slow, 1, 0, 8)
        try:
            first = b.submit([("query", "a")])
            time.sleep(0.05)  # the first batch runs; the second waits in the queue
            second = b.submit([("query", "b")])
            self.assertTrue(second.cancel(), "a queued request can be cancelled (its caller timed out)")
            self.assertEqual(len(first.result(timeout=5)), 1)
            third = b.submit([("query", "c")])
            self.assertEqual(len(third.result(timeout=5)), 1, "the worker thread still serves")
            time.sleep(0.05)
            self.assertEqual(b.pending, 0, "dropped items leave the pending bound")
        finally:
            b.close()


class ConfigTest(unittest.TestCase):
    def test_generation_rules(self) -> None:
        g = load(None, {"ANVILKIT_INFERENCE_LISTEN": "0.0.0.0:9108"})
        self.assertEqual(g.config.listen, "0.0.0.0:9108")
        with self.assertRaises(ConfigError):
            load(None, {"ANVILKIT_INFERENCE_MODEL_URL": "https://x"})
        with self.assertRaises(ConfigError):
            load(None, {"ANVILKIT_INFERENCE_LISTEN": "nowhere"})
        g2 = load(None, {"ANVILKIT_INFERENCE_LISTEN": "127.0.0.2:9108"})
        self.assertEqual(g.digest, g2.digest, "placements are not part of the digest")


@unittest.skipUnless(os.environ.get("ANVILKIT_INFERENCE_MODELS_DIR"), "ANVILKIT_INFERENCE_MODELS_DIR not set (locked weights)")
class RealModelTest(unittest.TestCase):
    """The locked weights: 1024-dim normalized dense, sparse token weights, relevance order."""

    @classmethod
    def setUpClass(cls) -> None:
        from anvilkit_inference.main import load_engine

        cls.cfg = Inference.model_validate({"models_dir": os.environ["ANVILKIT_INFERENCE_MODELS_DIR"]})
        cls.engine = Engine()
        load_engine(cls.engine, cls.cfg, Metrics())
        cls.server = Server(cls.cfg, cls.engine)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.stop()
        cls.engine.embed.close()
        cls.engine.rerank.close()

    def test_embed_and_rerank(self) -> None:
        lock = compute.read_models_lock(Path(self.cfg.models_dir) / "models.lock")
        status, out = self.server.call("/api/v1/embeddings", embed_body(("What are the brand colors?", "Brand colors are teal and slate.", "Headings use Inter."), kind="passage"))
        self.assertEqual(status, 200, out)
        self.assertEqual(out["modelRevision"], lock["BAAI/bge-m3"].revision)
        d = out["dense"]
        self.assertEqual(len(d[0]), 1024)
        self.assertAlmostEqual(sum(x * x for x in d[0]), 1.0, places=3)
        cos = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
        self.assertGreater(cos(d[0], d[1]), cos(d[0], d[2]))
        for s in out["sparse"]:
            self.assertEqual(s["indices"], sorted(s["indices"]))
            self.assertTrue(all(isinstance(i, int) for i in s["indices"]))
        status, out = self.server.call("/api/v1/rerankings", rerank_body("What are the brand colors?", (("c1", "Headings use Inter."), ("c2", "Brand colors are teal and slate."))))
        self.assertEqual(status, 200, out)
        scores = {s["candidateId"]: s["score"] for s in out["scores"]}
        self.assertGreater(scores["c2"], scores["c1"])
        self.assertEqual(out["modelRevision"], lock["BAAI/bge-reranker-v2-m3"].revision)


if __name__ == "__main__":
    unittest.main()
