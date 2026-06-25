from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


def _load_runner_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "runpod_ssh_runner.py"
    spec = importlib.util.spec_from_file_location("runpod_ssh_runner", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RunPodSshRunnerTest(unittest.TestCase):
    def test_safe_summary_redacts_secret_values(self) -> None:
        runner = _load_runner_module()
        pod = {
            "id": "pod123",
            "desiredStatus": "RUNNING",
            "costPerHr": 0.44,
            "imageName": "runpod/pytorch:test",
            "gpuCount": 1,
            "ports": ["8888/http", "22/tcp"],
            "portMappings": {"22": 22188},
            "publicIp": "203.0.113.1",
            "env": {"JUPYTER_PASSWORD": "secret", "PUBLIC_KEY": ""},
            "volumeInGb": 0,
            "containerDiskInGb": 50,
        }

        summary = runner._safe_pod_summary(pod)

        self.assertEqual(summary["id"], "pod123")
        self.assertEqual(summary["sshPort"], 22188)
        self.assertTrue(summary["hasJupyterPassword"])
        self.assertNotIn("secret", repr(summary))

    def test_endpoint_requires_public_ip_and_ssh_mapping(self) -> None:
        runner = _load_runner_module()
        endpoint = runner._endpoint_from_pod(
            {
                "id": "pod123",
                "publicIp": "203.0.113.1",
                "portMappings": {"22": "22188"},
                "desiredStatus": "RUNNING",
                "costPerHr": 0.44,
                "imageName": "runpod/pytorch:test",
                "ports": ["22/tcp"],
            }
        )

        self.assertEqual(endpoint.pod_id, "pod123")
        self.assertEqual(endpoint.public_ip, "203.0.113.1")
        self.assertEqual(endpoint.ssh_port, 22188)

    def test_fetch_uses_host_only_in_remote_path(self) -> None:
        runner = _load_runner_module()
        endpoint = runner.PodEndpoint(
            pod_id="pod123",
            public_ip="203.0.113.1",
            ssh_port=22188,
            status="RUNNING",
            cost_per_hr=0.44,
            image_name="runpod/pytorch:test",
            ports=("22/tcp",),
        )
        calls = []
        original_selected_endpoint = runner._selected_endpoint
        original_run = runner.subprocess.run
        try:
            runner._selected_endpoint = lambda pod_id: endpoint
            runner.subprocess.run = lambda cmd, check: calls.append((cmd, check))
            with tempfile.TemporaryDirectory() as tmpdir:
                args = type(
                    "Args",
                    (),
                    {
                        "pod_id": None,
                        "identity": Path("/tmp/id_rsa"),
                        "local_dest": Path(tmpdir) / "fetch",
                        "remote_path": "/workspace/relaleap/results/",
                    },
                )()

                runner.command_fetch(args)
        finally:
            runner._selected_endpoint = original_selected_endpoint
            runner.subprocess.run = original_run

        self.assertEqual(len(calls), 1)
        command, check = calls[0]
        self.assertTrue(check)
        transport = command[command.index("-e") + 1]
        self.assertNotIn("root@", transport)
        self.assertIn("root@203.0.113.1:/workspace/relaleap/results/", command)

    def test_setup_snippet_starts_sshd_directly(self) -> None:
        runner = _load_runner_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            public_key_path = Path(tmpdir) / "id_rsa.pub"
            public_key_path.write_text(
                "ssh-rsa AAAATESTKEY codex@example\n",
                encoding="utf-8",
            )
            args = type("Args", (), {"public_key": public_key_path})()
            output = io.StringIO()

            with redirect_stdout(output):
                runner.command_setup_snippet(args)

            snippet = output.getvalue()
            self.assertIn("mkdir -p /root/.ssh /run/sshd", snippet)
            self.assertIn("PermitRootLogin prohibit-password", snippet)
            self.assertIn("/usr/sbin/sshd -t", snippet)
            self.assertIn("/usr/sbin/sshd -E /tmp/sshd-codex.log", snippet)
            self.assertIn("relaleap-runpod-ssh-ready", snippet)


if __name__ == "__main__":
    unittest.main()
