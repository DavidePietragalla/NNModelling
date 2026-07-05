#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$SCRIPT_DIR/docs2"

echo "=== NNModelling Documentation Builder ==="

# Create uv venv for Sphinx if it doesn't exist
cd "$DOCS_DIR"
if [ ! -d ".venv" ]; then
    echo "Creating uv virtual environment for Sphinx..."
    uv venv
fi

# Install dependencies via uv
echo "Installing Sphinx and theme..."
uv pip install -q -r requirements.txt

# Make sphinx-build available
source .venv/bin/activate

# Build Sphinx HTML docs (clean build)
echo "Building Sphinx HTML documentation..."
sphinx-build -b html source build/html -W

echo ""
echo "=== Documentation complete! ==="
echo "HTML output: $DOCS_DIR/build/html/index.html"
