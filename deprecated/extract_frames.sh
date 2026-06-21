#!/usr/bin/env bash
# Extract ALL frames from the full match for T-DEED inference.
# Uses 224p for disk efficiency (~3 GB vs ~15 GB for 720p).
# Skips extraction if enough frames already exist (re-run safe).
#
# Output: <work_dir>/tdeed_full_frames/full_match/frame1.jpg …

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK_DIR="$(cd "$PIPELINE_DIR/.." && pwd)"        # parent of project root
DATA_DIR="$PIPELINE_DIR/data"
OUT_DIR="$WORK_DIR/tdeed_full_frames/full_match"

# ── find video (prefer 224p for disk efficiency) ───────────────────────────
VIDEO=$(find "$DATA_DIR" -name "224p.mp4" | sort | head -1)
if [[ -z "$VIDEO" ]]; then
    VIDEO=$(find "$DATA_DIR" -name "720p.mp4" | sort | head -1)
fi
if [[ -z "$VIDEO" ]]; then
    echo "ERROR: no 224p.mp4 or 720p.mp4 found under $DATA_DIR" >&2
    exit 1
fi
echo "Using video: $VIDEO"
echo "Output dir : $OUT_DIR"

# ── skip if already extracted ───────────────────────────────────────────────
if [[ -d "$OUT_DIR" ]]; then
    COUNT=$(find "$OUT_DIR" -maxdepth 1 -name "frame*.jpg" 2>/dev/null | wc -l | xargs)
    if [[ "$COUNT" -gt 100000 ]]; then
        echo "Found $COUNT frames in $OUT_DIR — skipping extraction"
        exit 0
    fi
fi

# ── extract all frames ─────────────────────────────────────────────────────
mkdir -p "$OUT_DIR"
echo "Extracting all frames (this takes a few minutes)..."
ffmpeg -y -i "$VIDEO" -vf fps=25 -q:v 2 "$OUT_DIR/frame%d.jpg"

COUNT=$(find "$OUT_DIR" -maxdepth 1 -name "frame*.jpg" | wc -l | xargs)
echo "Extracted $COUNT frames to $OUT_DIR"
