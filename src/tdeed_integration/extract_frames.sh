#!/usr/bin/env bash
# Step 1 — extract 500 frames (20 s @ 25 fps) starting at 5:00 from the
# first match video found under data/.
#
# Usage:
#   bash src/tdeed_integration/extract_frames.sh
#
# Output: /tmp/tdeed_test_frames/test_20s/frame1.jpg … frame500.jpg

set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="$PIPELINE_DIR/data"
OUT_DIR="/tmp/tdeed_test_frames/test_20s"

# ── find the first available video ─────────────────────────────────────────
VIDEO=$(find "$DATA_DIR" -name "720p.mp4" | sort | head -1)
if [[ -z "$VIDEO" ]]; then
    VIDEO=$(find "$DATA_DIR" -name "224p.mp4" | sort | head -1)
fi
if [[ -z "$VIDEO" ]]; then
    echo "ERROR: no 720p.mp4 or 224p.mp4 found under $DATA_DIR" >&2
    exit 1
fi
echo "Using video: $VIDEO"

# ── extract 500 frames ─────────────────────────────────────────────────────
rm -rf "/tmp/tdeed_test_frames"
mkdir -p "$OUT_DIR"

ffmpeg -y -ss 01:03:50 -i "$VIDEO" -t 20 -vf fps=25 \
    -q:v 2 "$OUT_DIR/frame%d.jpg"

COUNT=$(ls "$OUT_DIR"/frame*.jpg 2>/dev/null | wc -l)
echo "Extracted $COUNT frames to $OUT_DIR"
if [[ "$COUNT" -lt 400 ]]; then
    echo "WARNING: expected ~500 frames, got $COUNT — video may be shorter than 5:20" >&2
fi
