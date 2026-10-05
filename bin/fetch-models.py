"""Build-time only: downloads every model of models.lock at its revision and verifies each file's SHA-256."""

import hashlib
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

lock, target = Path(sys.argv[1]), Path(sys.argv[2])
models: list[tuple[str, str, dict[str, str]]] = []
for line in lock.read_text().splitlines():
    if not line.strip() or line.startswith("#"):
        continue
    parts = line.split()
    if parts[0] == "model":
        models.append((parts[1], parts[2], {}))
    else:
        models[-1][2][parts[1]] = parts[0]
for repo, revision, files in models:
    out = target / repo.replace("/", "--")
    snapshot_download(repo, revision=revision, local_dir=out, allow_patterns=[*files, "README.md"])
    for name, digest in files.items():
        actual = hashlib.sha256((out / name).read_bytes()).hexdigest()
        if actual != digest:
            sys.exit(f"{repo}/{name}: sha256 {actual} is not the locked {digest}")
    shutil.rmtree(out / ".cache", ignore_errors=True)
    print(f"verified {len(files)} files of {repo}@{revision}")
shutil.copyfile(lock, target / "models.lock")
