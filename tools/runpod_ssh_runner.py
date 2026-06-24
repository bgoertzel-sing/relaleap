#!/usr/bin/env python3
"""Small RunPod SSH bridge for RelaLeap command-driven experiments.

This helper intentionally does not create pods by default.  It discovers an
existing pod through the RunPod REST API, derives the public SSH endpoint, and
then uses ordinary ssh/rsync once the pod has SSH enabled.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RUNPOD_API_URL = "https://rest.runpod.io/v1"
KEYCHAIN_SERVICE = "codex-runpod-api-key"
DEFAULT_REPO_URL = "https://github.com/bgoertzel-sing/relaleap.git"
DEFAULT_REMOTE_DIR = "/workspace/relaleap"
DEFAULT_IDENTITY = Path.home() / ".ssh" / "id_rsa"
DEFAULT_PUBLIC_KEY = Path.home() / ".ssh" / "id_rsa.pub"
DEFAULT_RUN_COMMAND = (
    "python3 -m relaleap.experiments.compare "
    "--config configs/char_validation_support_wide_hep_temporal_clipped_objective_gate.yaml "
    "--config configs/char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate.yaml "
    "--out results/comparisons/runpod_support_width_validation_probe "
    "&& python3 -m relaleap.experiments.check_artifacts "
    "--comparison-dir results/comparisons/runpod_support_width_validation_probe "
    "--out results/comparisons/runpod_support_width_validation_probe/artifact_check.json"
)


@dataclass(frozen=True)
class PodEndpoint:
    pod_id: str
    public_ip: str
    ssh_port: int
    status: str | None
    cost_per_hr: Any
    image_name: str | None
    ports: tuple[str, ...]


def _read_api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if key:
        return key
    try:
        user = os.environ.get("USER") or subprocess.check_output(
            ["whoami"], text=True
        ).strip()
        key = subprocess.check_output(
            [
                "security",
                "find-generic-password",
                "-a",
                user,
                "-s",
                KEYCHAIN_SERVICE,
                "-w",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        key = ""
    if not key:
        raise SystemExit(
            "RunPod API key not found. Set RUNPOD_API_KEY or save it in "
            f"macOS Keychain as {KEYCHAIN_SERVICE!r}."
        )
    return key


def _api_json(path: str, *, query: dict[str, str] | None = None) -> Any:
    api_key = _read_api_key()
    url = RUNPOD_API_URL + path
    if query:
        url += "?" + urllib.parse.urlencode(query)
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"RunPod API request failed: HTTP {exc.code}: {body}") from exc


def _list_pods() -> list[dict[str, Any]]:
    pods = _api_json(
        "/pods",
        query={
            "includeMachine": "true",
            "includeNetworkVolume": "true",
            "includeTemplate": "true",
        },
    )
    if not isinstance(pods, list):
        raise SystemExit("RunPod API returned an unexpected non-list pod response.")
    return pods


def _select_pod(pods: list[dict[str, Any]], pod_id: str | None) -> dict[str, Any]:
    if pod_id:
        for pod in pods:
            if pod.get("id") == pod_id:
                return pod
        raise SystemExit(f"No RunPod pod found with id {pod_id!r}.")
    running = [pod for pod in pods if pod.get("desiredStatus") == "RUNNING"]
    if len(running) == 1:
        return running[0]
    if not running:
        raise SystemExit("No running RunPod pods found.")
    ids = ", ".join(str(pod.get("id")) for pod in running)
    raise SystemExit(f"Multiple running RunPod pods found; pass --pod-id. IDs: {ids}")


def _endpoint_from_pod(pod: dict[str, Any]) -> PodEndpoint:
    mappings = pod.get("portMappings") or {}
    public_ip = pod.get("publicIp")
    ssh_port = mappings.get("22") or mappings.get(22)
    if not public_ip or not ssh_port:
        raise SystemExit(
            "Selected pod does not expose full SSH via public IP and TCP port 22."
        )
    return PodEndpoint(
        pod_id=str(pod.get("id")),
        public_ip=str(public_ip),
        ssh_port=int(ssh_port),
        status=pod.get("desiredStatus"),
        cost_per_hr=pod.get("costPerHr"),
        image_name=pod.get("imageName") or pod.get("image"),
        ports=tuple(str(port) for port in (pod.get("ports") or ())),
    )


def _selected_endpoint(pod_id: str | None) -> PodEndpoint:
    return _endpoint_from_pod(_select_pod(_list_pods(), pod_id))


def _ssh_args(endpoint: PodEndpoint, identity: Path) -> list[str]:
    return [
        "ssh",
        "-i",
        str(identity),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-p",
        str(endpoint.ssh_port),
        f"root@{endpoint.public_ip}",
    ]


def _ssh_shell_quote(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def _safe_pod_summary(pod: dict[str, Any]) -> dict[str, Any]:
    env = pod.get("env") or {}
    mappings = pod.get("portMappings") or {}
    return {
        "id": pod.get("id"),
        "status": pod.get("desiredStatus"),
        "costPerHr": pod.get("costPerHr"),
        "imageName": pod.get("imageName") or pod.get("image"),
        "gpuCount": pod.get("gpuCount"),
        "ports": pod.get("ports") or [],
        "sshPort": mappings.get("22") or mappings.get(22),
        "publicIpPresent": bool(pod.get("publicIp")),
        "hasJupyterPassword": bool(env.get("JUPYTER_PASSWORD")),
        "hasPublicKeyEnv": bool(env.get("PUBLIC_KEY") or env.get("SSH_PUBLIC_KEY")),
        "volumeInGb": pod.get("volumeInGb"),
        "containerDiskInGb": pod.get("containerDiskInGb"),
    }


def command_status(args: argparse.Namespace) -> None:
    pods = _list_pods()
    print(json.dumps([_safe_pod_summary(pod) for pod in pods], indent=2))


def command_ssh_command(args: argparse.Namespace) -> None:
    endpoint = _selected_endpoint(args.pod_id)
    print(_ssh_shell_quote(_ssh_args(endpoint, args.identity)))


def command_probe_ssh(args: argparse.Namespace) -> None:
    endpoint = _selected_endpoint(args.pod_id)
    cmd = _ssh_args(endpoint, args.identity) + ["echo relaleap-runpod-ssh-ok"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "SSH probe failed. Enable/start SSH on the pod first; run "
            "`python tools/runpod_ssh_runner.py setup-snippet` and paste the "
            "printed block into the RunPod web terminal."
        ) from exc


def command_setup_snippet(args: argparse.Namespace) -> None:
    public_key = args.public_key.read_text(encoding="utf-8").strip()
    if not public_key.startswith(("ssh-rsa ", "ssh-ed25519 ", "ecdsa-")):
        raise SystemExit(f"{args.public_key} does not look like an SSH public key.")
    print(
        "\n".join(
            [
                "apt-get update",
                "DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server rsync git",
                "mkdir -p ~/.ssh",
                "chmod 700 ~/.ssh",
                "cat >> ~/.ssh/authorized_keys <<'EOF'",
                public_key,
                "EOF",
                "chmod 600 ~/.ssh/authorized_keys",
                "service ssh start || /usr/sbin/sshd",
            ]
        )
    )


def command_bootstrap(args: argparse.Namespace) -> None:
    endpoint = _selected_endpoint(args.pod_id)
    remote_dir = shlex.quote(args.remote_dir)
    repo_url = shlex.quote(args.repo_url)
    remote = f"""
set -euo pipefail
if [ ! -d {remote_dir}/.git ]; then
  rm -rf {remote_dir}
  git clone {repo_url} {remote_dir}
else
  cd {remote_dir}
  git fetch origin main
  git reset --hard origin/main
fi
cd {remote_dir}
python3 --version
python3 - <<'PY'
try:
    import torch
    print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())
except Exception as exc:
    print('torch check failed', repr(exc))
PY
python3 -m pip install --upgrade pip
python3 -m pip install -e . --no-build-isolation
python3 -m unittest discover -s tests
"""
    subprocess.run(_ssh_args(endpoint, args.identity) + [remote], check=True)


def command_run(args: argparse.Namespace) -> None:
    endpoint = _selected_endpoint(args.pod_id)
    remote = f"set -euo pipefail; cd {shlex.quote(args.remote_dir)}; {args.command}"
    subprocess.run(_ssh_args(endpoint, args.identity) + [remote], check=True)


def command_fetch(args: argparse.Namespace) -> None:
    endpoint = _selected_endpoint(args.pod_id)
    args.local_dest.mkdir(parents=True, exist_ok=True)
    ssh_transport = _ssh_shell_quote(_ssh_args(endpoint, args.identity))
    remote_path = (
        f"root@{endpoint.public_ip}:{shlex.quote(args.remote_path.rstrip('/'))}/"
    )
    subprocess.run(
        [
            "rsync",
            "-az",
            "--progress",
            "-e",
            ssh_transport,
            remote_path,
            str(args.local_dest),
        ],
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", help="RunPod pod id. Optional if exactly one pod is running.")
    parser.add_argument(
        "--identity",
        type=Path,
        default=DEFAULT_IDENTITY,
        help=f"SSH private key path. Default: {DEFAULT_IDENTITY}",
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    status = subparsers.add_parser("status", help="List pods without printing secrets.")
    status.set_defaults(func=command_status)

    ssh_command = subparsers.add_parser("ssh-command", help="Print the full SSH command.")
    ssh_command.set_defaults(func=command_ssh_command)

    probe = subparsers.add_parser("probe-ssh", help="Test non-interactive SSH access.")
    probe.set_defaults(func=command_probe_ssh)

    snippet = subparsers.add_parser(
        "setup-snippet",
        help="Print commands to paste into the RunPod web terminal to enable SSH.",
    )
    snippet.add_argument(
        "--public-key",
        type=Path,
        default=DEFAULT_PUBLIC_KEY,
        help=f"SSH public key path. Default: {DEFAULT_PUBLIC_KEY}",
    )
    snippet.set_defaults(func=command_setup_snippet)

    bootstrap = subparsers.add_parser(
        "bootstrap",
        help="Clone/update RelaLeap on the pod and install/test it.",
    )
    bootstrap.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    bootstrap.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    bootstrap.set_defaults(func=command_bootstrap)

    run = subparsers.add_parser("run", help="Run a command inside the remote RelaLeap repo.")
    run.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    run.add_argument("--command", default=DEFAULT_RUN_COMMAND)
    run.set_defaults(func=command_run)

    fetch = subparsers.add_parser("fetch", help="Fetch remote artifacts with rsync.")
    fetch.add_argument("--remote-path", default=f"{DEFAULT_REMOTE_DIR}/results/")
    fetch.add_argument("--local-dest", type=Path, default=Path("results/runpod_fetch"))
    fetch.set_defaults(func=command_fetch)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
