#!/usr/bin/env bash
# Master runner: Step 1 (extract frames) → Step 2 (T-DEED inference)
#
# Usage (from pipeline root):
#   bash src/tdeed_integration/run_all.sh
#
# Requires: ffmpeg, conda env with T-DEED deps active

set -euo pipefail

# Derive paths from this script's own location — works wherever home lives
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TDEED_DIR="$(cd "$PIPELINE_DIR/../models/T-DEED" && pwd)"

echo "══════════════════════════════════════════"
echo " T-DEED integration test (20-second clip)"
echo "══════════════════════════════════════════"

# ── Step 1: extract frames ─────────────────────────────────────────────────
echo ""
echo "─── Step 1: extract frames ───"
bash "$SCRIPT_DIR/extract_frames.sh"

# ── Step 2: T-DEED inference ───────────────────────────────────────────────
echo ""
echo "─── Step 2: T-DEED inference ───"
cd "$TDEED_DIR"
python "$SCRIPT_DIR/run_tdeed_test.py"

echo ""
echo "══ Done. Results in /tmp/tdeed_test_out/detections.json ══"
