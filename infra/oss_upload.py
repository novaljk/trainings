#!/usr/bin/env python3
"""Upload the repository dataset directory to Alibaba Cloud OSS.

Configuration is read from infra/credentials.yaml. No OSS credentials are
accepted on the command line.

Default destination:
    oss://tars-data-platform-software/dataset/minimind/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    print(
        "Missing dependency: PyYAML. Install it with:\n"
        "    pip install PyYAML==6.0.3",
        file=sys.stderr,
    )
    raise

try:
    import oss2
except ImportError as exc:
    print(
        "Missing dependency: oss2. Install it with:\n"
        "    pip install oss2==2.19.1",
        file=sys.stderr,
    )
    raise


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "credentials.yaml"
DEFAULT_SOURCE = REPO_ROOT / "dataset"
DEFAULT_PREFIX = "dataset/minimind"
DEFAULT_PROFILE = "software"
DEFAULT_PROFILE_ENV = "TARS_OSS_DEFAULT_PROFILE"
RESUMABLE_THRESHOLD = 100 * 1024 * 1024
PART_SIZE = 100 * 1024 * 1024
UPLOAD_THREADS = 4


class OssConfigError(RuntimeError):
    """The OSS configuration is missing or invalid."""


def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"https://{endpoint}"
    return endpoint


def normalize_prefix(prefix: str) -> str:
    return prefix.strip("/").replace("\\", "/")


def load_oss_profile(config_path: Path, profile: str) -> dict[str, Any]:
    if not config_path.is_file():
        raise OssConfigError(f"OSS config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    oss = data.get("oss")
    if not isinstance(oss, dict):
        raise OssConfigError(f"The 'oss' section is missing in {config_path}")

    selected = oss.get(profile)
    if not isinstance(selected, dict):
        available = ", ".join(k for k, v in oss.items() if isinstance(v, dict)) or "none"
        raise OssConfigError(
            f"Unknown OSS profile {profile!r}. Available profiles: {available}"
        )

    # Profile-specific values override shared top-level values.
    return {**oss, **selected}


def iter_local_files(source: Path, skip_incomplete: bool) -> list[Path]:
    files: list[Path] = []
    for root, dirs, names in os.walk(source, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(names):
            path = Path(root) / name
            if not path.is_file():
                continue
            if name in {".DS_Store"} or name.endswith(".pyc"):
                continue
            if skip_incomplete and name.endswith(".incomplete"):
                continue
            files.append(path)
    return files


def remote_size(bucket: oss2.Bucket, key: str) -> int | None:
    """Return the OSS object size, or None when the object does not exist."""
    try:
        return int(bucket.head_object(key).content_length)
    except oss2.exceptions.NoSuchKey:
        return None
    except oss2.exceptions.OssError as exc:
        if getattr(exc, "status", None) == 404:
            return None
        raise


def upload_one(
    bucket: oss2.Bucket,
    local_path: Path,
    key: str,
) -> None:
    size = local_path.stat().st_size
    if size > RESUMABLE_THRESHOLD:
        oss2.resumable_upload(
            bucket,
            key,
            str(local_path),
            part_size=PART_SIZE,
            num_threads=UPLOAD_THREADS,
        )
    else:
        bucket.put_object_from_file(key, str(local_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the local dataset directory to Alibaba Cloud OSS."
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help=f"OSS profile in the credentials file. Default: {DEFAULT_PROFILE}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Credentials YAML file. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Local source directory. Default: repository dataset directory",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"Remote OSS prefix. Default: {DEFAULT_PREFIX}",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip an object when its remote size matches the local file size.",
    )
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="Skip local files whose names end with .incomplete.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned uploads without contacting OSS.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = os.environ.get(DEFAULT_PROFILE_ENV, "").strip() or args.profile

    try:
        config = load_oss_profile(args.config, profile)
        bucket_name = config["bucket"]
        endpoint = normalize_endpoint(config["endpoint"])
        access_key_id = config["access_key_id"]
        access_key_secret = config["access_key_secret"]
        security_token = config.get("security_token")
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError, OssConfigError) as exc:
        print(f"Failed to load OSS profile from {args.config}: {exc}", file=sys.stderr)
        return 2

    if not args.source.is_dir():
        print(f"Source directory does not exist: {args.source}", file=sys.stderr)
        return 2

    source_root = args.source.resolve()
    files = iter_local_files(source_root, args.skip_incomplete)
    if not files:
        print(f"No files found under {args.source}", file=sys.stderr)
        return 1

    prefix = normalize_prefix(args.prefix)

    print(f"Profile:    {profile}")
    print(f"Bucket:     {bucket_name}")
    print(f"Endpoint:   {endpoint}")
    print(f"Source:     {source_root}")
    print(f"Prefix:     {prefix}/")
    print(f"Files:      {len(files)}")
    print()

    if args.dry_run:
        for local_path in files:
            relative = local_path.relative_to(source_root).as_posix()
            print(f"{local_path} -> oss://{bucket_name}/{prefix}/{relative}")
        return 0

    if security_token:
        auth = oss2.StsAuth(access_key_id, access_key_secret, security_token)
    else:
        auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    uploaded = 0
    skipped = 0
    uploaded_bytes = 0
    failures: list[tuple[Path, str, Exception]] = []

    for index, local_path in enumerate(files, start=1):
        relative = local_path.relative_to(source_root).as_posix()
        key = f"{prefix}/{relative}"
        size = local_path.stat().st_size

        print(f"[{index}/{len(files)}] oss://{bucket_name}/{key} ({size} bytes)")

        if args.skip_existing:
            try:
                remote = remote_size(bucket, key)
            except oss2.exceptions.OssError as exc:
                print(f"    Failed to inspect remote object: {exc}", file=sys.stderr)
                failures.append((local_path, key, exc))
                continue
            if remote == size:
                skipped += 1
                print("    Skipped: remote object has the same size.")
                continue

        try:
            upload_one(bucket, local_path, key)
            uploaded += 1
            uploaded_bytes += size
            print("    Uploaded.")
        except oss2.exceptions.OssError as exc:
            print(f"    Failed: {exc}", file=sys.stderr)
            failures.append((local_path, key, exc))

    result = {
        "status": "error" if failures else "ok",
        "uploaded_files": uploaded,
        "skipped_files": skipped,
        "uploaded_bytes": uploaded_bytes,
        "failed_files": len(failures),
        "oss_path": f"oss://{bucket_name}/{prefix}/",
    }
    print()
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if failures:
        print("\nFailed objects:", file=sys.stderr)
        for local_path, key, exc in failures:
            print(f"- {local_path} -> oss://{bucket_name}/{key}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
