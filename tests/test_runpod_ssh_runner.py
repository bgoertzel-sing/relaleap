from __future__ import annotations

import importlib.util
import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
