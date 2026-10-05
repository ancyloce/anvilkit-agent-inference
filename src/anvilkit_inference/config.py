"""Immutable configuration generations of anvilkit-agent-inference (DD-09 §4).

defaults < the reviewed config.yaml < a validated, unexpired Apollo snapshot
(non-secret keys) < the allowlisted ANVILKIT_INFERENCE_* environment, validated
as a whole; the pattern of packages/profile-schemas/python/config_generation.py.
The service holds no secret (fixed local weights, no provider key, no
database): placements come from the environment only. A generation is built
once at start; a changed input takes effect through a rolling restart, never
by mutating the running service.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ENV_PREFIX = "ANVILKIT_INFERENCE_"
APP_ID = "anvilkit-agent-inference"
PLACEMENT = re.compile(r"(^|\.)(listen|models_dir|snapshot_file)$")

ENV_OVERRIDES = {
    "ANVILKIT_INFERENCE_LISTEN": "listen",
    "ANVILKIT_INFERENCE_MODELS_DIR": "models_dir",
    "ANVILKIT_INFERENCE_APOLLO_SNAPSHOT_FILE": "apollo.snapshot_file",
}


class Embedding(BaseModel):
    """The locked BGE-M3 profile: dense (normalized, 1024) and sparse vectors under one profile id."""

    model_config = ConfigDict(extra="forbid", strict=True)
    profile_id: str = Field(default="bge-m3-v1", min_length=1, max_length=128)
    query_max_tokens: int = Field(default=512, ge=8, le=8192)
    passage_max_tokens: int = Field(default=2048, ge=8, le=8192)
    max_inputs: int = Field(default=64, ge=1, le=128)
    max_request_chars: int = Field(default=262_144, ge=1, le=4_194_304)
    batch_size: int = Field(default=16, ge=1, le=128)


class Rerank(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    profile_id: str = Field(default="bge-reranker-v2-m3-v1", min_length=1, max_length=128)
    max_tokens: int = Field(default=1024, ge=8, le=8192)
    max_candidates: int = Field(default=64, ge=1, le=256)
    max_request_chars: int = Field(default=262_144, ge=1, le=4_194_304)
    batch_size: int = Field(default=16, ge=1, le=256)


class Queue(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    max_pending_items: int = Field(default=512, ge=1, le=65_536)
    max_wait_ms: int = Field(default=5, ge=0, le=1000)
    request_timeout_ms: int = Field(default=30_000, ge=100, le=600_000)


class Apollo(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    mode: str = Field(default="disabled", pattern="^(disabled|snapshot)$")
    app_id: str = APP_ID
    snapshot_file: str = ""


class Inference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    listen: str = Field(default="127.0.0.1:9108", pattern=r"^[^:\s]+:\d{1,5}$")
    models_dir: str = "/opt/models"
    torch_threads: int = Field(default=4, ge=1, le=256)
    max_body_bytes: int = Field(default=4_194_304, ge=1024, le=67_108_864)
    shutdown_timeout_s: int = Field(default=20, ge=1, le=300)
    embedding: Embedding = Embedding()
    rerank: Rerank = Rerank()
    queue: Queue = Queue()
    apollo: Apollo = Apollo()

    @model_validator(mode="after")
    def cross_fields(self) -> "Inference":
        if self.queue.max_pending_items < max(self.embedding.max_inputs, self.rerank.max_candidates):
            raise ValueError("queue.max_pending_items must admit one full request")
        return self


class Generation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    number: int
    config: Inference
    digest: str
    apollo_release: str = ""
    expires_at: str = ""


class ConfigError(Exception):
    pass


def _digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _set(raw: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = raw
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
        if not isinstance(cur, dict):
            raise ConfigError(f"{dotted}: {p} is not a mapping")
    cur[parts[-1]] = value


def _leaves(raw: dict[str, Any], prefix: str = "") -> list[str]:
    out = []
    for k, v in raw.items():
        path = f"{prefix}.{k}" if prefix else k
        out.extend(_leaves(v, path) if isinstance(v, dict) else [path])
    return out


def _snapshot(path: str, schema_dir: Path | None, now: datetime) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text())
    if schema_dir is not None:
        schema = json.loads((schema_dir / "apollo-snapshot.schema.json").read_text())
        jsonschema.Draft202012Validator(schema).validate(doc)
    if doc.get("appId") != APP_ID:
        raise ConfigError(f"apollo snapshot: appId {doc.get('appId')} is not this service")
    expires = datetime.fromisoformat(str(doc["expiresAt"]).replace("Z", "+00:00"))
    if expires <= now:
        raise ConfigError("apollo snapshot: expired; an expired snapshot never starts a generation")
    return doc


def load(
    config_file: str | None,
    environ: dict[str, str] | None = None,
    number: int = 1,
    now: datetime | None = None,
    schema_dir: Path | None = None,
) -> Generation:
    environ = dict(os.environ if environ is None else environ)
    now = now or datetime.now(timezone.utc)
    raw: dict[str, Any] = {}
    if config_file:
        try:
            loaded = yaml.safe_load(Path(config_file).read_text()) or {}
        except (OSError, yaml.YAMLError) as err:
            raise ConfigError(f"config file {config_file}: {err}") from err
        if not isinstance(loaded, dict):
            raise ConfigError(f"config file {config_file}: not a mapping")
        for key in _leaves(loaded):
            if PLACEMENT.search(key):
                raise ConfigError(f"config file {config_file}: {key} is a placement and is accepted only from the environment")
        raw.update(loaded)
    env: dict[str, str] = {}
    unknown = []
    for name, value in environ.items():
        if not name.startswith(ENV_PREFIX) or name == f"{ENV_PREFIX}CONFIG":
            continue
        if name in ENV_OVERRIDES:
            env[ENV_OVERRIDES[name]] = value
        else:
            unknown.append(name)
    if unknown:
        raise ConfigError("environment variables are not allowed overrides: " + ", ".join(sorted(unknown)))
    release = ""
    expires = ""
    if (raw.get("apollo") or {}).get("mode") == "snapshot":
        snap_file = env.get("apollo.snapshot_file", "")
        if not snap_file:
            raise ConfigError("apollo.snapshot_file is required in snapshot mode (ANVILKIT_INFERENCE_APOLLO_SNAPSHOT_FILE)")
        snap = _snapshot(snap_file, schema_dir, now)
        for key, value in snap["configurations"].items():
            if PLACEMENT.search(key):
                raise ConfigError(f"apollo snapshot: {key} is a placement and never comes from Apollo")
            _set(raw, key, int(value) if isinstance(value, str) and value.isdigit() else value)
        release, expires = snap["releaseKey"], snap["expiresAt"]
    for key, value in env.items():
        _set(raw, key, value)
    try:
        cfg = Inference.model_validate(raw)
    except ValidationError as err:
        raise ConfigError(f"config: {err}") from None
    non_placement = {k: v for k, v in cfg.model_dump().items() if k not in ("listen", "models_dir")}
    return Generation(
        number=number,
        config=cfg,
        digest=_digest(json.dumps(non_placement, sort_keys=True, separators=(",", ":"))),
        apollo_release=release,
        expires_at=expires,
    )
