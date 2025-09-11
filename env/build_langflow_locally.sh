#!/usr/bin/env bash
set -euo pipefail

# --- Put Apptainer temp & cache on your large HDD ---
export APPTAINER_CACHEDIR="/media/cameron/HDD/.apptainer/cache"
export APPTAINER_TMPDIR="/media/cameron/HDD/apptainer_tmp"

mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

echo "Apptainer cache: $APPTAINER_CACHEDIR"
echo "Apptainer tmp:   $APPTAINER_TMPDIR"
df -h "$APPTAINER_CACHEDIR" || true
df -h "$APPTAINER_TMPDIR" || true

# --- Sanity checks ---
test -f langflow-agent.def || { echo "langflow-agent.def not found"; exit 1; }
test -s chemflow.yml || { echo "chemflow.yml not found or empty"; exit 1; }

# --- Clean any old cache to free space (safe) ---
apptainer cache clean -a || true

# --- Build the image ---
# Prefer rootless with fakeroot if supported:
if apptainer build --help 2>/dev/null | grep -q -- "--fakeroot"; then
  echo "Building with --fakeroot..."
  apptainer build --fakeroot langflow-agent.sif langflow-agent.def
else
  echo "Building without --fakeroot (may require sudo on some systems)..."
  # Uncomment if your setup requires sudo:
  # sudo APPTAINER_TMPDIR="$APPTAINER_TMPDIR" APPTAINER_CACHEDIR="$APPTAINER_CACHEDIR" \
  #   apptainer build langflow-agent.sif langflow-agent.def
  apptainer build langflow-agent.sif langflow-agent.def
fi

echo "Build complete: ./langflow-agent.sif"

# --- Quick smoke test ---
echo "Testing Python in the image..."
apptainer exec ./langflow-agent.sif python -V
apptainer exec ./langflow-agent.sif which python

