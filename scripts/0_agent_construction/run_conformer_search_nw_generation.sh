#!/usr/bin/env bash
# run_pipeline.sh
# Runs the conformer search to produce products_memory.csv, then generates NWChem inputs.

set -euo pipefail

# --- Find the directory this script lives in (handles spaces) ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# --- Config (override via env vars if needed) ---
PYTHON_BIN="${PYTHON_BIN:-python}"
IN_CSV="${IN_CSV:-$SCRIPT_DIR/reaction_products.csv}"
OUT_CSV="${OUT_CSV:-$SCRIPT_DIR/products_memory.csv}"
CONFORMER_SCRIPT="${CONFORMER_SCRIPT:-$SCRIPT_DIR/conformer.py}"
NW_SCRIPT="${NW_SCRIPT:-$SCRIPT_DIR/generate_nw.py}"
# -----------------------------------------------

echo "[info] Script dir: $SCRIPT_DIR"
echo "[info] Using python: $PYTHON_BIN"
echo "[info] Conformer script: $CONFORMER_SCRIPT"
echo "[info] NWChem generator: $NW_SCRIPT"
echo "[info] Input products table: $IN_CSV"
echo "[info] Output products memory: $OUT_CSV"

# Basic checks
[[ -f "$CONFORMER_SCRIPT" ]] || { echo "[error] Missing $CONFORMER_SCRIPT"; exit 1; }
[[ -f "$NW_SCRIPT" ]] || { echo "[error] Missing $NW_SCRIPT"; exit 1; }
[[ -f "$IN_CSV" ]] || { echo "[error] Missing input CSV: $IN_CSV"; exit 1; }

echo "[step] Running conformer search → $OUT_CSV"
"$PYTHON_BIN" "$CONFORMER_SCRIPT" --in "$IN_CSV" --out "$OUT_CSV"

# Sanity check output exists and is non-empty
if [[ ! -s "$OUT_CSV" ]]; then
  echo "[error] $OUT_CSV was not created or is empty."
  exit 1
fi
echo "[ok] Created $OUT_CSV ($(wc -l < "$OUT_CSV") lines)"

echo "[step] Generating NWChem scripts"
"$PYTHON_BIN" "$NW_SCRIPT"

echo "[done] Pipeline completed successfully."
