#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHONUNBUFFERED=1
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${REPO_ROOT}"

python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_version={torch.version.cuda}")
print(f"gpu_count={torch.cuda.device_count()}")
PY

python - <<'PY'
import importlib

required = [
    "datasets",
    "transformers",
    "trl",
    "yaml",
    "oss2",
    "requests",
]
for name in required:
    try:
        module = importlib.import_module(name)
        print(f"{name}=ok")
    except Exception as exc:
        raise SystemExit(f"{name} import failed: {exc}") from exc
PY

python -m compileall -q trainer model dataset scripts

echo "Smoke test passed."
