#!/usr/bin/env python3
"""Submit and inspect TARS PAI distributed training jobs from the minimind repo.

The API contract is adapted from the company-internal PAI tool. This standalone
CLI does not depend on tars_platform; it authenticates directly with TARS and
resolves human-readable names in a job YAML to PAI ids.

Default code source:
    https://github.com/novaljk/trainings.git (main) -> /mnt/run/code

Default dataset/output mounts:
    dataset: oss://tars-data-platform-software/dataset/minimind/ -> /mnt/data
    output:  oss://tars-data-platform-software/minimind/runs/<jobName>/ -> /mnt/run

Common commands:
    python infra/pai_submit.py list-clusters
    python infra/pai_submit.py list-images --cluster-name 集群1
    python infra/pai_submit.py list-storages --cluster-name 集群1
    python infra/pai_submit.py submit --config dlc/pai_job.yaml --dry-run
    python infra/pai_submit.py submit --config dlc/pai_job.yaml
    python infra/pai_submit.py status --job-id 123456
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "credentials.yaml"
DEFAULT_JOB_CONFIG = REPO_ROOT / "dlc" / "pai_job.yaml"
DEFAULT_BASE_URL = "https://open.tars-ai.com"
DEFAULT_TIMEOUT = 30

# minimind's PAI-mountable OSS storage on cluster 1.
DEFAULT_DATA_STORAGE_NAME = "tars-data-platform-software"
DEFAULT_DATA_FILE_SYSTEM_PATH = "/dataset/minimind/"
DEFAULT_DATA_MOUNT_PATH = "/mnt/data"
DEFAULT_RUN_STORAGE_NAME = "tars-data-platform-software"
DEFAULT_RUN_FILE_SYSTEM_PATH_PREFIX = "/minimind/runs/"
DEFAULT_RUN_MOUNT_PATH = "/mnt/run"

# GitHub source defaults. The repository is public, so no credentials are put
# into the generated PAI command.
DEFAULT_SOURCE_TYPE = "git"
DEFAULT_GIT_URL = "https://github.com/novaljk/trainings.git"
DEFAULT_GIT_BRANCH = "main"
DEFAULT_GIT_DEPTH = 1
DEFAULT_CODE_DIR = "/mnt/run/code"
DEFAULT_PYTHON = "python3"
DEFAULT_PIP_INDEX_URL = "https://mirrors.aliyun.com/pypi/simple"


class PAIError(RuntimeError):
    pass


def dump(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PAIError(f"YAML file not found: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise PAIError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PAIError(f"YAML root must be a mapping: {path}")
    return value


def load_credentials(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise PAIError(f"Credentials file not found: {config_path}")
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise PAIError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PAIError(f"Credentials YAML root must be a mapping: {config_path}")
    return value


class PAIClient:
    def __init__(self, config_path: Path = DEFAULT_CONFIG) -> None:
        credentials = load_credentials(config_path)
        pai = credentials.get("pai")
        if not isinstance(pai, dict):
            pai = {}

        self.base_url = (
            os.environ.get("PAI_BASE_URL", "").strip()
            or str(pai.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        )
        token = os.environ.get("PAI_ACCESS_TOKEN", "").strip()
        username = os.environ.get("PAI_USERNAME", "").strip()
        password = os.environ.get("PAI_PASSWORD", "")

        if not username:
            username = str(pai.get("username") or "").strip()
        if not password:
            password = str(pai.get("password") or "")

        if not token and username and password:
            token = self._login(username, password)
        if not token:
            raise PAIError(
                "No PAI access token. Set PAI_ACCESS_TOKEN, or set PAI_USERNAME/PAI_PASSWORD, "
                "or add a pai.username/pai.password section to infra/credentials.yaml."
            )

        self.token = token
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def _login(self, username: str, password: str) -> str:
        # TARS password login requires SHA-256(password), not plaintext.
        response = requests.post(
            f"{self.base_url}/api/v1/auth/login",
            json={
                "loginType": "PASSWORD",
                "clientId": "tarsai",
                "username": username,
                "password": hashlib.sha256(password.encode()).hexdigest(),
            },
            timeout=DEFAULT_TIMEOUT,
        )
        payload = self._json(response, "login")
        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("userToken"), dict):
            data = data["userToken"]
        token = data.get("accessToken") if isinstance(data, dict) else None
        if payload.get("code") not in (0, 200) or not token:
            raise PAIError(f"TARS login failed: {payload.get('msg') or payload}")
        return str(token)

    def _json(self, response: requests.Response, stage: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise PAIError(f"PAI returned non-JSON for {stage}: HTTP {response.status_code} {response.text[:300]}") from exc
        if not isinstance(payload, dict):
            raise PAIError(f"Unexpected PAI response for {stage}: {payload!r}")
        return payload

    def request(self, method: str, path: str, *, json_body: dict | None = None, params: dict | None = None) -> Any:
        response = self.session.request(
            method,
            f"{self.base_url}{path}",
            json=json_body,
            params=params,
            timeout=DEFAULT_TIMEOUT,
        )
        payload = self._json(response, f"{method} {path}")
        if payload.get("code") not in (0, 200):
            raise PAIError(
                f"PAI API failed ({method} {path}): code={payload.get('code')} "
                f"msg={payload.get('msg')} traceId={payload.get('traceId')}"
            )
        return payload.get("data")

    def clusters(self) -> list[dict[str, Any]]:
        value = self.request("GET", "/api/v1/pai/cluster/cluster-option")
        return value if isinstance(value, list) else []

    def gpu_types(self, cluster_id: int) -> list[dict[str, Any]]:
        value = self.request("GET", f"/api/v1/pai/spec/gpu-type-enum/{cluster_id}")
        rows = value if isinstance(value, list) else []
        return [
            {"specId": row.get("key"), "name": row.get("value")}
            for row in rows
            if isinstance(row, dict)
        ]

    def resource_pools(self, cluster_id: int | None = None) -> list[dict[str, Any]]:
        params = {"clusterId": cluster_id} if cluster_id is not None else None
        value = self.request("GET", "/api/v1/pai/resource/allocate-static", params=params)
        return value if isinstance(value, list) else []

    def projects(self) -> list[dict[str, Any]]:
        value = self.request("GET", "/api/v1/pai/project/list/optional")
        return value if isinstance(value, list) else []

    def images(self, cluster_id: int, name: str | None = None) -> list[dict[str, Any]]:
        body: dict[str, Any] = {"page": 1, "pageSize": 100, "clusterId": cluster_id}
        if name:
            # PAI's imageName filter matches the repository name, not name:tag.
            # Keep the full value for local matching, but query by the repository part.
            body["imageName"] = name.split(":", 1)[0]
        value = self.request("POST", "/api/v1/pai/image/list", json_body=body)
        return value if isinstance(value, list) else []

    def datasets(self, cluster_id: int | None = None, name: str | None = None) -> list[dict[str, Any]]:
        body: dict[str, Any] = {}
        if cluster_id is not None:
            body["clusterId"] = cluster_id
        if name:
            body["datasetName"] = name
        value = self.request("POST", "/api/v1/pai/dataset/list", json_body=body)
        return value if isinstance(value, list) else []

    def storages(self, cluster_id: int | None = None, resource_id: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if cluster_id is not None:
            params["clusterId"] = cluster_id
        if resource_id is not None:
            params["resourceId"] = resource_id
        value = self.request("GET", "/api/v1/pai/file-storage/list/optional", params=params)
        return value if isinstance(value, list) else []

    def create_job(self, body: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        response = self.session.post(
            f"{self.base_url}/api/v1/pai/train/create-job",
            json=body,
            timeout=DEFAULT_TIMEOUT,
        )
        payload = self._json(response, "create-job")
        if payload.get("code") not in (0, 200):
            raise PAIError(
                f"Create PAI job failed: code={payload.get('code')} "
                f"msg={payload.get('msg')} traceId={payload.get('traceId')} request_body={body}"
            )
        return payload.get("data"), payload

    def job_status(self, job_id: int) -> Any:
        return self.request("GET", f"/api/v1/pai/train/detail/{job_id}")

    def user_id(self) -> int | None:
        try:
            value = self.request("GET", "/api/v1/auth/user")
        except PAIError:
            return None
        if isinstance(value, dict):
            uid = value.get("id")
            if isinstance(uid, int):
                return uid
            if isinstance(uid, str) and uid.isdigit():
                return int(uid)
        return None


def match_row(rows: list[dict[str, Any]], value: Any, fields: list[str]) -> dict[str, Any]:
    """Match exact values first, then substring values."""
    needle = str(value or "").strip()
    if not needle:
        raise PAIError(f"Cannot match an empty value using fields {fields}")

    for row in rows:
        image_name = str(row.get("imageName") or "").strip()
        tag = str(row.get("tag") or "").strip()
        if image_name and tag and f"{image_name}:{tag}" == needle:
            return row
        for field in fields:
            if row.get(field) is not None and str(row.get(field)).strip() == needle:
                return row

    for row in rows:
        for field in fields:
            field_value = row.get(field)
            if field_value is None:
                continue
            field_text = str(field_value).strip()
            if needle in field_text or field_text in needle:
                return row

    available = [str(row.get(fields[0])) for row in rows[:30] if row.get(fields[0]) is not None]
    raise PAIError(f"Cannot find {needle!r} in {fields}. Available: {available}")


def cloud_type(cluster: dict[str, Any]) -> str:
    company = str(cluster.get("company") or "")
    if "火山" in company or "volc" in company.lower():
        return "volc"
    if "阿里" in company or "aliyun" in company.lower():
        return "aliyun"
    return "volc" if cluster.get("id") in (3, 4) else "aliyun"


def build_train_command(config: dict[str, Any]) -> str:
    if config.get("command"):
        return str(config["command"])

    train = config.get("train")
    if not isinstance(train, dict):
        raise PAIError("Job YAML must contain either command or a train section")

    stage = str(train.get("stage") or "pretrain").strip()
    if stage not in {"pretrain", "sft"}:
        raise PAIError(f"Unsupported train.stage: {stage}. Use pretrain or sft.")

    values: list[str] = []
    for key, default in (
        ("TRAIN_STAGE", stage),
        ("DATA_ROOT", "/mnt/data"),
        ("OUT_DIR", "/mnt/run/out"),
        ("CKPT_DIR", "/mnt/run/checkpoints"),
        ("NNODES", str(config.get("podCount", 1))),
        ("NPROC_PER_NODE", "gpu"),
        ("RESUME", "1"),
    ):
        value = train.get(key, default)
        if value is not None:
            values.append(f"{key}={shlex.quote(str(value))}")

    for key in (
        "DATA_PATH",
        "EPOCHS",
        "BATCH_SIZE",
        "LEARNING_RATE",
        "HIDDEN_SIZE",
        "NUM_HIDDEN_LAYERS",
        "MAX_SEQ_LEN",
        "SAVE_INTERVAL",
        "ACCUMULATION_STEPS",
    ):
        if key in train and train[key] is not None:
            values.append(f"{key}={shlex.quote(str(train[key]))}")

    source = config.get("source")
    if isinstance(source, str):
        try:
            source = yaml.safe_load(source)
        except yaml.YAMLError as exc:
            raise PAIError(f"Invalid source YAML: {exc}") from exc
    if source is None:
        source = {}

    if not isinstance(source, dict):
        raise PAIError("Job YAML source section must be a mapping")

    source_type = str(source.get("type") or DEFAULT_SOURCE_TYPE).strip().lower()
    if source_type not in {"git", "image"}:
        raise PAIError(f"Unsupported source.type: {source_type}. Use git or image")

    if source_type == "image":
        return "env " + " ".join(values) + " /app/dlc/train.sh"

    git_url = str(source.get("url") or DEFAULT_GIT_URL).strip()
    branch = str(source.get("branch") or DEFAULT_GIT_BRANCH).strip()
    depth = int(source.get("depth") or DEFAULT_GIT_DEPTH)
    code_dir = str(source.get("codeDir") or DEFAULT_CODE_DIR).strip()
    python = str(source.get("python") or DEFAULT_PYTHON).strip()
    install_requirements = bool(source.get("installRequirements", True))
    requirements_path = str(source.get("requirementsPath") or "requirements.txt").strip()
    pip_index_url = str(source.get("pipIndexUrl") or DEFAULT_PIP_INDEX_URL).strip()

    if not git_url:
        raise PAIError("Git source requires url")
    if not branch:
        raise PAIError("Git source requires branch")
    if depth < 1:
        raise PAIError("Git source depth must be at least 1")
    if not code_dir.startswith("/"):
        raise PAIError("Git source codeDir must be an absolute path")
    if code_dir == "/":
        raise PAIError("Git source codeDir cannot be /")

    q_git_url = shlex.quote(git_url)
    q_branch = shlex.quote(branch)
    q_code_dir = shlex.quote(code_dir)
    q_python = shlex.quote(python)
    q_requirements = shlex.quote(requirements_path)
    q_pip_index = shlex.quote(pip_index_url)
    q_train_sh = shlex.quote(f"{code_dir.rstrip('/')}/dlc/train.sh")

    q_git_dir = shlex.quote(code_dir.rstrip("/") + "/.git")
    lines = [
        "set -Eeuo pipefail",
        "command -v git >/dev/null",
        f"mkdir -p {q_code_dir}",
        f"if [ -d {q_git_dir} ]; then",
        f"  cd {q_code_dir}",
        f"  git remote set-url origin {q_git_url}",
        f"  git fetch --prune --depth {depth} origin {q_branch}",
        f"  git checkout -B {q_branch} origin/{q_branch}",
        f"  git reset --hard origin/{q_branch}",
        "elif [ -e " + q_code_dir + " ] && [ -n \"$(ls -A " + q_code_dir + " 2>/dev/null)\" ]; then",
        f"  echo 'codeDir exists but is not a Git repository: {code_dir}' >&2",
        "  exit 2",
        "else",
        f"  git clone --depth {depth} --branch {q_branch} {q_git_url} {q_code_dir}",
        "fi",
        f"cd {q_code_dir}",
    ]

    if install_requirements:
        lines.extend(
            [
                f"test -f {q_requirements}",
                f"{q_python} -m pip install -r {q_requirements} -i {q_pip_index}",
            ]
        )

    lines.extend(
        [
            f"chmod +x {q_train_sh}",
            "exec env " + " ".join(values) + " " + q_train_sh,
        ]
    )

    # PAI executes the submitted command with /bin/sh (dash on many images).
    # dash does not support `set -o pipefail`, so force the generated script
    # through bash while keeping the command as a single PAI command string.
    return "/bin/bash <<'PAI_BASH'\n" + "\n".join(lines) + "\nPAI_BASH"

def coerce_mount_config(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def extract_job_id(data: Any) -> int | None:
    if isinstance(data, bool):
        return None
    if isinstance(data, int):
        return data
    if not isinstance(data, dict):
        return None
    for key in ("id", "jobId", "trainJobId", "trainId"):
        value = data.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def infer_phase(data: Any) -> str:
    if not isinstance(data, dict):
        return "unknown"
    completed = {"succeeded", "success", "completed", "finished"}
    failed = {"failed", "error", "cancelled", "canceled", "terminated"}
    for key in ("status", "jobStatus", "state", "phase", "jobState"):
        value = str(data.get(key) or "").strip().lower()
        if not value:
            continue
        if any(token in value for token in completed):
            return "completed"
        if any(token in value for token in failed):
            return "failed"
        return "running"
    return "unknown"


def build_job_body(client: PAIClient, config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    required = ["jobName", "clusterName", "priority", "podCount", "imageName"]
    missing = [key for key in required if config.get(key) in (None, "")]
    if missing:
        raise PAIError(f"Job YAML is missing required fields: {missing}")

    steps: list[str] = []
    clusters = client.clusters()
    cluster = match_row(clusters, config["clusterName"], ["name"])
    cluster_id = int(cluster["id"])
    cloud = cloud_type(cluster)
    steps.append(f"cluster '{config['clusterName']}' -> clusterId={cluster_id}, cloud={cloud}")

    image_rows = client.images(cluster_id, str(config["imageName"]))
    image = match_row(image_rows, config["imageName"], ["imageName", "imageUri", "id"])
    image_id = image.get("imageId") or image.get("id") or image.get("imageName")
    steps.append(f"image '{config['imageName']}' -> imageId={image_id}")

    body: dict[str, Any] = {
        "clusterId": cluster_id,
        "jobName": str(config["jobName"]),
        "imageId": str(image_id),
        "podCount": int(config["podCount"]),
        "priority": int(config["priority"]),
        "command": build_train_command(config),
    }

    if cloud == "aliyun":
        if not config.get("specName"):
            raise PAIError("Aliyun PAI clusters require specName")
        specs = client.gpu_types(cluster_id)
        spec = match_row(specs, config["specName"], ["name"])
        body["specId"] = int(spec["specId"])
        steps.append(f"spec '{config['specName']}' -> specId={body['specId']}")
    else:
        if not config.get("instanceType"):
            raise PAIError("Volcano PAI clusters require instanceType")
        body["instanceTypeInfo"] = {"id": str(config["instanceType"])}
        steps.append(f"instanceType '{config['instanceType']}' -> instanceTypeInfo.id")

    resource_pool_id: int | None = None
    pools = client.resource_pools(cluster_id)
    if config.get("resourcePoolName"):
        pool = match_row(pools, config["resourcePoolName"], ["name"])
        resource_pool_id = int(pool.get("id") or pool.get("resourcePoolId"))
        steps.append(f"resource pool '{config['resourcePoolName']}' -> resourcePoolId={resource_pool_id}")
    elif cloud == "aliyun" and config.get("specName"):
        # The PAI create-job API currently requires resourcePoolId. Match the
        # selected GPU type to a pool automatically when the user omitted it.
        gpu_type = str(config["specName"]).strip()
        matching_pools = [
            row for row in pools
            if str(row.get("gpuType") or "").strip() == gpu_type
        ]
        if len(matching_pools) != 1:
            available = [
                f"{row.get('name')} (gpuType={row.get('gpuType')}, id={row.get('id')})"
                for row in pools
            ]
            raise PAIError(
                f"Cannot infer a unique resource pool for GPU type {gpu_type!r}. "
                "Set resourcePoolName explicitly. Available: " + "; ".join(available)
            )
        pool = matching_pools[0]
        resource_pool_id = int(pool.get("id") or pool.get("resourcePoolId"))
        steps.append(
            f"resource pool inferred from GPU type {gpu_type!r} -> "
            f"resourcePoolId={resource_pool_id} ({pool.get('name')})"
        )

    if resource_pool_id is not None:
        body["resourcePoolId"] = resource_pool_id

    if config.get("projectName"):
        projects = client.projects()
        project = match_row(projects, config["projectName"], ["name"])
        body["projectId"] = int(project.get("id") or project.get("projectId"))
        steps.append(f"project '{config['projectName']}' -> projectId={body['projectId']}")

    for key in ("gpuCount", "cpuCount", "errorMonitor", "enableTensorboard", "maxRunTime", "publicJobGroup"):
        if config.get(key) is not None:
            body[key] = config[key]

    if config.get("dataSet"):
        dataset_rows = client.datasets(cluster_id)
        mounts: list[dict[str, Any]] = []
        for item in config["dataSet"]:
            dataset = match_row(dataset_rows, item.get("datasetName"), ["datasetName", "name"])
            mount_path = str(item["mountPath"]).strip()
            if not mount_path.startswith("/"):
                raise PAIError(f"Dataset mountPath must be an absolute Linux path: {mount_path!r}")
            mount_path = mount_path.rstrip("/") + "/"

            mount = {
                "datasetLocalId": int(dataset.get("id") or dataset.get("datasetLocalId")),
                "mountPath": mount_path,
            }
            if item.get("version") is not None:
                mount["version"] = item["version"]
            if item.get("readOnly") is not None:
                mount["readOnly"] = bool(item["readOnly"])
            if item.get("config") is not None:
                mount["config"] = coerce_mount_config(item["config"])
            mounts.append(mount)
            steps.append(f"dataset '{item.get('datasetName')}' -> datasetLocalId={mount['datasetLocalId']}")
        body["dataSet"] = mounts

    if not config.get("storageMountList"):
        # Defaults are tailored to minimind on cluster 1: dataset is under
        # /dataset/minimind/, while run artifacts use a unique sibling prefix.
        run_id = str(config.get("runId") or config.get("jobName")).strip("/")
        config = {
            **config,
            "storageMountList": [
                {
                    "storageName": DEFAULT_DATA_STORAGE_NAME,
                    "fileSystemPath": DEFAULT_DATA_FILE_SYSTEM_PATH,
                    "mountPath": DEFAULT_DATA_MOUNT_PATH,
                    "readOnly": True,
                },
                {
                    "storageName": DEFAULT_RUN_STORAGE_NAME,
                    "fileSystemPath": f"{DEFAULT_RUN_FILE_SYSTEM_PATH_PREFIX}{run_id}/",
                    "mountPath": DEFAULT_RUN_MOUNT_PATH,
                    "readOnly": False,
                },
            ],
        }

    if config.get("storageMountList"):
        storage_rows = client.storages(cluster_id, resource_pool_id)
        mounts = []
        for item in config["storageMountList"]:
            storage = match_row(storage_rows, item.get("storageName"), ["storageName", "name"])
            mount_path = str(item["mountPath"]).strip()
            if not mount_path.startswith("/"):
                raise PAIError(f"Storage mountPath must be an absolute Linux path: {mount_path!r}")
            # PAI requires mountPath to start and end with "/".
            mount_path = mount_path.rstrip("/") + "/"

            file_system_path = str(item["fileSystemPath"]).strip()
            if not file_system_path.startswith("/"):
                raise PAIError(f"Storage fileSystemPath must be absolute: {file_system_path!r}")
            if not file_system_path.endswith("/"):
                file_system_path += "/"

            mount = {
                "fileStorageId": int(storage.get("id") or storage.get("fileStorageId")),
                "fileSystemPath": file_system_path,
                "mountPath": mount_path,
            }
            if item.get("readOnly") is not None:
                mount["readOnly"] = bool(item["readOnly"])
            if item.get("workspaceSourceFlag") is not None:
                mount["workspaceSourceFlag"] = bool(item["workspaceSourceFlag"])
            if item.get("config") is not None:
                mount["config"] = coerce_mount_config(item["config"])
            mounts.append(mount)
            steps.append(
                f"storage '{item.get('storageName')}' -> fileStorageId={mount['fileStorageId']}"
            )
        body["storageMountList"] = mounts

    return body, steps


def job_url(client: PAIClient, job_name: str) -> str:
    user_id = client.user_id()
    encoded = quote(job_name)
    if user_id is not None:
        return f"{client.base_url}/pai/distributed-training?onlyMe=false&userId={user_id}&jobName={encoded}&page=1&pageSize=20"
    return f"{client.base_url}/pai/distributed-training?onlyMe=true&jobName={encoded}&page=1&pageSize=20"


def require_int(value: str, name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise PAIError(f"{name} must be an integer, got {value!r}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--credentials", type=Path, default=DEFAULT_CONFIG, help="Credentials YAML path")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-clusters", help="List PAI clusters")

    p = sub.add_parser("list-images", help="List PAI images")
    p.add_argument("--cluster-id", type=int)
    p.add_argument("--cluster-name")
    p.add_argument("--name")

    p = sub.add_parser("list-gpu-types", help="List Aliyun GPU machine specs")
    p.add_argument("--cluster-id", type=int, required=True)

    p = sub.add_parser("list-resource-pools", help="List PAI resource pools")
    p.add_argument("--cluster-id", type=int)
    p.add_argument("--cluster-name")

    p = sub.add_parser("list-storages", help="List PAI file storages")
    p.add_argument("--cluster-id", type=int)
    p.add_argument("--cluster-name")
    p.add_argument("--resource-id", type=int)

    p = sub.add_parser("list-datasets", help="List PAI datasets")
    p.add_argument("--cluster-id", type=int)
    p.add_argument("--cluster-name")
    p.add_argument("--name")

    p = sub.add_parser("submit", help="Resolve names in YAML and submit a PAI job")
    p.add_argument("--config", type=Path, default=DEFAULT_JOB_CONFIG)
    p.add_argument("--dry-run", action="store_true", help="Resolve and print the request without submitting")

    p = sub.add_parser("status", help="Get PAI job status by numeric job id")
    p.add_argument("--job-id", required=True)

    return parser.parse_args()


def resolve_cluster_id(args: argparse.Namespace, client: PAIClient) -> int | None:
    if getattr(args, "cluster_id", None) is not None:
        return args.cluster_id
    name = getattr(args, "cluster_name", None)
    if not name:
        return None
    return int(match_row(client.clusters(), name, ["name"])["id"])


def main() -> int:
    args = parse_args()
    try:
        client = PAIClient(args.credentials)

        if args.command == "list-clusters":
            dump({"status": "ok", "clusters": client.clusters()})
        elif args.command == "list-images":
            cluster_id = resolve_cluster_id(args, client)
            if cluster_id is None:
                raise PAIError("list-images requires --cluster-id or --cluster-name")
            dump({"status": "ok", "images": client.images(cluster_id, args.name)})
        elif args.command == "list-gpu-types":
            dump({"status": "ok", "gpu_types": client.gpu_types(args.cluster_id)})
        elif args.command == "list-resource-pools":
            cluster_id = resolve_cluster_id(args, client)
            dump({"status": "ok", "resource_pools": client.resource_pools(cluster_id)})
        elif args.command == "list-storages":
            cluster_id = resolve_cluster_id(args, client)
            dump({"status": "ok", "storages": client.storages(cluster_id, args.resource_id)})
        elif args.command == "list-datasets":
            cluster_id = resolve_cluster_id(args, client)
            dump({"status": "ok", "datasets": client.datasets(cluster_id, args.name)})
        elif args.command == "submit":
            config = load_yaml(args.config)
            body, steps = build_job_body(client, config)
            if args.dry_run:
                dump({"status": "dry_run", "config": str(args.config), "resolved": steps, "request_body": body})
                return 0
            data, response = client.create_job(body)
            result = {
                "status": "ok",
                "jobName": config["jobName"],
                "jobId": extract_job_id(data),
                "job_url": job_url(client, str(config["jobName"])),
                "data": data,
                "msg": response.get("msg", ""),
                "traceId": response.get("traceId", ""),
                "resolved": steps,
            }
            dump(result)
        elif args.command == "status":
            job_id = require_int(args.job_id, "job-id")
            data = client.job_status(job_id)
            dump({"status": "ok", "jobId": job_id, "inferred_phase": infer_phase(data), "data": data})
        return 0
    except (PAIError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        dump({"status": "error", "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
