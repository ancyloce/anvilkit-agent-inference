"""The fixed computation of anvilkit-agent-inference (DD-07 §1/§3).

BGE-M3 dense and sparse embeddings and the bge-reranker-v2-m3 scores from
the locked local weights (models.lock); no model path, download URL, code or
connection is accepted from a request. Items of concurrent requests are
micro-batched per model on one worker thread within the configured batch
size and wait window; the pending items are bounded, and a request that does
not fit is refused (CAPACITY_EXHAUSTED) instead of queueing without limit.
"""

from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Protocol


def input_digest(parts: list[str]) -> str:
    """The contract's input digest: each part as UTF-8 byte length, LF and the bytes."""
    h = hashlib.sha256()
    for p in parts:
        b = p.encode("utf-8")
        h.update(str(len(b)).encode("ascii"))
        h.update(b"\n")
        h.update(b)
    return "sha256:" + h.hexdigest()


@dataclass(frozen=True)
class LockedModel:
    repository: str
    revision: str
    license: str
    files: dict[str, str]

    @property
    def directory(self) -> str:
        return self.repository.replace("/", "--")


def read_models_lock(path: Path) -> dict[str, LockedModel]:
    models: dict[str, LockedModel] = {}
    current: LockedModel | None = None
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "model":
            current = LockedModel(parts[1], parts[2], parts[3], {})
            models[parts[1]] = current
        elif current is not None and len(parts) == 2:
            current.files[parts[1]] = parts[0]
    return models


class Full(Exception):
    """The pending items of a model would exceed the bound."""


class Model(Protocol):
    def run(self, items: list[Any]) -> list[Any]: ...


@dataclass
class _Job:
    items: list[Any]
    future: Future
    queued_at: float = field(default_factory=time.monotonic)


class Batcher:
    """Micro-batches items across requests on one worker thread per model."""

    def __init__(self, name: str, model: Model, batch_size: int, max_wait_ms: int, max_pending: int,
                 observe: Callable[[str, int, float], None] | None = None) -> None:
        self.name = name
        self.model = model
        self.batch_size = batch_size
        self.max_wait = max_wait_ms / 1000
        self.max_pending = max_pending
        self.observe = observe
        self._queue: Queue[_Job | None] = Queue()
        self._pending = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name=f"batcher-{name}", daemon=True)
        self._thread.start()

    @property
    def pending(self) -> int:
        return self._pending

    def submit(self, items: list[Any]) -> Future:
        with self._lock:
            if self._pending + len(items) > self.max_pending:
                raise Full(f"{self.name}: {self._pending} items pending")
            self._pending += len(items)
        job = _Job(items, Future())
        self._queue.put(job)
        return job.future

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5)

    def _loop(self) -> None:
        while True:
            first = self._queue.get()
            if first is None:
                return
            jobs = [first]
            size = len(first.items)
            deadline = time.monotonic() + self.max_wait
            while size < self.batch_size:
                try:
                    nxt = self._queue.get(timeout=max(0.0, deadline - time.monotonic()))
                except Empty:
                    break
                if nxt is None:
                    self._queue.put(None)
                    break
                jobs.append(nxt)
                size += len(nxt.items)
            # A request that timed out cancelled its still-pending future:
            # its items are dropped here, and the others are marked running
            # so a later cancellation cannot race the result.
            live = [j for j in jobs if j.future.set_running_or_notify_cancel()]
            dropped = sum(len(j.items) for j in jobs) - sum(len(j.items) for j in live)
            items = [it for j in live for it in j.items]
            started = time.monotonic()
            try:
                out: list[Any] = []
                for i in range(0, len(items), self.batch_size):
                    out.extend(self.model.run(items[i : i + self.batch_size]))
                pos = 0
                for j in live:
                    j.future.set_result(out[pos : pos + len(j.items)])
                    pos += len(j.items)
            except Exception as err:  # the model's failure fails exactly these requests
                for j in live:
                    if not j.future.done():
                        j.future.set_exception(err)
            finally:
                with self._lock:
                    self._pending -= len(items) + dropped
                if self.observe and items:
                    self.observe(self.name, len(items), time.monotonic() - started)


class Bge3:
    """BGE-M3 dense+sparse under the locked profile; inputs are (kind, text) pairs."""

    def __init__(self, directory: Path, query_max: int, passage_max: int) -> None:
        from FlagEmbedding import BGEM3FlagModel

        self._m = BGEM3FlagModel(str(directory), use_fp16=False, devices=["cpu"], normalize_embeddings=True)
        self._max = {"query": query_max, "passage": passage_max}

    def run(self, items: list[tuple[str, str]]) -> list[tuple[list[float], tuple[list[int], list[float]]]]:
        out: list[Any] = [None] * len(items)
        for kind in ("query", "passage"):
            idx = [i for i, (k, _) in enumerate(items) if k == kind]
            if not idx:
                continue
            enc = self._m.encode(
                [items[i][1] for i in idx],
                batch_size=len(idx),
                max_length=self._max[kind],
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            for pos, i in enumerate(idx):
                weights = enc["lexical_weights"][pos]
                pairs = sorted((int(t), float(w)) for t, w in weights.items() if float(w) > 0.0)
                out[i] = (
                    [float(x) for x in enc["dense_vecs"][pos]],
                    ([t for t, _ in pairs][:8192], [w for _, w in pairs][:8192]),
                )
        return out


class Reranker:
    """bge-reranker-v2-m3 raw relevance scores for (query, text) pairs."""

    def __init__(self, directory: Path, max_tokens: int) -> None:
        from FlagEmbedding import FlagReranker

        self._m = FlagReranker(str(directory), use_fp16=False, devices=["cpu"])
        self._max = max_tokens

    def run(self, items: list[tuple[str, str]]) -> list[float]:
        scores = self._m.compute_score([list(p) for p in items], max_length=self._max, batch_size=len(items))
        if not isinstance(scores, list):
            scores = [scores]
        return [float(s) for s in scores]
