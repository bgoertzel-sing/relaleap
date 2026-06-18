#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$ROOT_DIR/.venv-conda"
PKGS_DIR="$ROOT_DIR/.conda-pkgs"

cd "$ROOT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH."
  echo "Install Miniconda/Anaconda or activate a shell where conda is available."
  exit 1
fi

if [ ! -x "$ENV_DIR/bin/python" ]; then
  echo "Creating local Python 3.11 environment at:"
  echo "  $ENV_DIR"
  CONDA_PKGS_DIRS="$PKGS_DIR" conda create -p "$ENV_DIR" python=3.11 -y
else
  echo "Using existing environment:"
  echo "  $ENV_DIR"
fi

PYTHON="$ENV_DIR/bin/python"

echo "Python:"
"$PYTHON" --version

echo "Upgrading pip..."
"$PYTHON" -m pip install --upgrade pip

echo "Installing RelaLeap Colab bridge dependencies..."
"$PYTHON" -m pip install -e '.[colab-bridge]' --no-build-isolation

echo "Installing Chromium for Playwright..."
"$PYTHON" -m playwright install chromium

cat <<EOF

Colab bridge setup complete.

Run this to open the notebook and log in:

  "$PYTHON" tools/colab_playwright_runner.py --manual-login

Then try:

  "$PYTHON" tools/colab_playwright_runner.py --run-all

EOF
