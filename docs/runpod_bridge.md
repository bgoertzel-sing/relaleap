# RunPod Bridge

RunPod is the temporary GPU backend for RelaLeap when Colab is unreliable.
The source of truth remains the GitHub repo plus command-generated artifacts.
RunPod should only run ordinary repository commands, then send artifact trees
back for the same local artifact checks used elsewhere.

## Current Access Model

The RunPod API key is stored in macOS Keychain as:

```text
codex-runpod-api-key
```

The local helper reads that key without printing it:

```bash
python tools/runpod_ssh_runner.py status
```

Codex can start or stop an existing pod when Ben explicitly asks:

```bash
python tools/runpod_ssh_runner.py --pod-id POD_ID start
python tools/runpod_ssh_runner.py --pod-id POD_ID stop
```

If exactly one pod exists, `--pod-id` may be omitted. For convenience, the
helper also accepts `start --pod-id POD_ID` and `stop --pod-id POD_ID`. It
prints only a redacted pod summary and does not create or delete pods.

The existing test pod is visible through the API and exposes Jupyter plus TCP
port 22, but SSH is not accepting connections yet. Before automation can sync
files or run experiments, enable SSH once through the RunPod web terminal.

## Enable SSH Once

The helper can print a pasteable setup block using the local SSH public key:

```bash
python tools/runpod_ssh_runner.py setup-snippet
```

Paste the printed block into the RunPod web terminal for the pod. It installs
`openssh-server`, `rsync`, and `git`, appends the local public key to
`authorized_keys`, writes a small root-key SSH config, creates the container
runtime directory SSH needs, and starts `sshd` directly. A successful setup ends
with:

```text
relaleap-runpod-ssh-ready
```

Then verify from the local machine:

```bash
python tools/runpod_ssh_runner.py probe-ssh
```

If multiple pods are running, add `--pod-id POD_ID`.

If the setup block fails inside the web terminal, run this in the same terminal
and paste the output back into Codex:

```bash
tail -n 100 /tmp/sshd-codex.log 2>/dev/null || true
/usr/sbin/sshd -t
```

## Bootstrap RelaLeap

After SSH works, clone or refresh RelaLeap on the pod and run the local test
suite:

```bash
python tools/runpod_ssh_runner.py bootstrap
```

The bootstrap command uses `/workspace/relaleap` and the official GitHub repo.
It creates `/workspace/relaleap/.venv-runpod` with access to the template's
preinstalled PyTorch/CUDA packages, avoiding Ubuntu's system-Python package
guard while preserving the CUDA stack.

## Run A GPU Probe

Run the default support-width validation probe:

```bash
python tools/runpod_ssh_runner.py run
```

Or pass a specific command:

```bash
python tools/runpod_ssh_runner.py run --command 'python -m relaleap.experiments.compare --config configs/char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2.yaml --config configs/char_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate_seed2.yaml --out results/comparisons/runpod_char_larger_contextual_router_seed2'
```

## Fetch Artifacts

Fetch the remote `results/` tree into a local ignored directory:

```bash
python tools/runpod_ssh_runner.py fetch
```

Then run the same artifact checker locally against the fetched comparison
directory before treating RunPod output as evidence.

## Cost Hygiene

The helper does not create or delete pods automatically. The RelaLeap automation
should prefer RunPod over Colab whenever a RunPod pod is already running and SSH
is reachable, but it should not start or stop pods on its own. Pod lifecycle
changes are interaction-level actions: Ben asks Codex to start or stop the pod,
or Ben uses the RunPod console directly.
