"""
sliding_window.py
-----------------
Cuts a football match video into overlapping 60-second clips.
Slide every 30 seconds (50% overlap) so no action gets split across clips.

What's new in this version:
    - Half boundary is read from Labels-ball.json (SoccerNet annotation)
      not hardcoded to 2700s. Real halftime is at 46:04 in this match,
      not exactly 45:00. This fixes second-half time alignment.

Usage:
    python sliding_window.py
    python sliding_window.py --video 224p.mp4
    python sliding_window.py --duration 60
    python sliding_window.py --step 30
    python sliding_window.py --test

Each clip is saved temporarily to data/temp/clip_current.mp4
After action_recognizer.py processes it, the clip is deleted.
"""

import os
import json
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime


# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent.parent
DATA_DIR    = BASE_DIR / "data"
MATCH_DIR   = DATA_DIR / "2019-10-01 - Blackburn Rovers - Nottingham Forest"
TEMP_DIR    = DATA_DIR / "temp"
LOG_DIR     = BASE_DIR / "logs"

DEFAULT_VIDEO = MATCH_DIR / "720p.mp4"
TEST_VIDEO    = MATCH_DIR / "224p.mp4"
LABELS_PATH   = MATCH_DIR / "Labels-ball.json"

# fallback if Labels-ball.json not available
DEFAULT_HALFTIME_SEC = 2700.0   # 45:00


# ── helpers ────────────────────────────────────────────────────────────────

def get_halftime_sec(labels_path: Path = LABELS_PATH) -> float:
    """
    Read actual halftime from Labels-ball.json.
    Format example: "halftime": "1 - 46:04"
    Returns seconds (e.g. 2764.0 for 46:04).
    Falls back to 2700.0 if file unavailable.
    """
    if not labels_path.exists():
        print(f"  [halftime] Labels-ball.json not found, using default 45:00")
        return DEFAULT_HALFTIME_SEC

    try:
        with open(labels_path, encoding="utf-8") as f:
            data = json.load(f)
        ht_str = data.get("halftime", "")
        # parse "1 - 46:04" → minutes + seconds
        if " - " in ht_str:
            _, time_part = ht_str.split(" - ")
            mm, ss = time_part.strip().split(":")
            sec = int(mm) * 60 + int(ss)
            print(f"  [halftime] read from Labels-ball.json: {ht_str} → {sec}s")
            return float(sec)
    except Exception as e:
        print(f"  [halftime] error reading Labels-ball.json: {e}")

    print(f"  [halftime] using default 45:00")
    return DEFAULT_HALFTIME_SEC


def get_video_duration(video_path: Path) -> float:
    """Use ffprobe to get the total duration of the video in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info   = json.loads(result.stdout)
    return float(info["format"]["duration"])


def extract_clip(video_path: Path, start_sec: float, duration: int, out_path: Path) -> bool:
    """ffmpeg stream copy — fast, no re-encoding."""
    cmd = [
        "ffmpeg",
        "-ss", str(start_sec),
        "-i", str(video_path),
        "-t", str(duration),
        "-c", "copy",
        "-avoid_negative_ts", "1",
        "-y",
        str(out_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def seconds_to_gametime(seconds: float, halftime_sec: float = None) -> str:
    """
    Convert raw video seconds to match game time string.
    e.g. 554.3 → "1st 09:14"
    
    Uses actual halftime_sec if provided (real halftime from match data),
    falls back to 2700s (exactly 45:00).
    """
    if halftime_sec is None:
        halftime_sec = DEFAULT_HALFTIME_SEC

    if seconds < halftime_sec:
        half    = "1st"
        minutes = int(seconds // 60)
        secs    = int(seconds % 60)
    else:
        half    = "2nd"
        adj     = seconds - halftime_sec
        minutes = int(adj // 60)
        secs    = int(adj % 60)
    return f"{half} {minutes:02d}:{secs:02d}"


def delete_clip(clip_path: Path):
    if clip_path.exists():
        clip_path.unlink()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN WINDOW GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def run_sliding_window(
    video_path    : Path,
    clip_duration : int = 60,
    step          : int = 30,
    test_mode     : bool = False,
    halftime_sec  : float = None,
    on_clip_ready = None,
):
    """
    Slide a window across the video. Each clip is extracted, passed to
    on_clip_ready callback, then deleted.

    Args:
        video_path    : path to .mp4
        clip_duration : seconds per clip (default 60)
        step          : slide step seconds (default 30 → 50% overlap)
        test_mode     : only first 3 clips
        halftime_sec  : actual halftime (loaded from Labels-ball.json if None)
        on_clip_ready : callback (clip_path, start_sec, end_sec, gametime)
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if halftime_sec is None:
        halftime_sec = get_halftime_sec()

    log_path = LOG_DIR / f"sliding_window_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    print(f"\n{'='*60}")
    print(f"  Soccer EKG — Sliding Window")
    print(f"{'='*60}")
    print(f"  Video       : {video_path.name}")
    print(f"  Clip size   : {clip_duration}s | Step: {step}s | Overlap: {clip_duration-step}s")
    print(f"  Halftime    : {halftime_sec:.0f}s ({int(halftime_sec//60)}:{int(halftime_sec%60):02d})")
    print(f"  Temp dir    : {TEMP_DIR}")
    if test_mode:
        print(f"  TEST MODE   : only first 3 clips")
    print(f"{'='*60}\n")

    total_duration = get_video_duration(video_path)
    total_clips    = int((total_duration - clip_duration) / step) + 1
    print(f"  Match duration : {total_duration/60:.1f} min  →  ~{total_clips} clips total\n")

    clip_count = 0
    start_sec  = 0.0

    while start_sec + clip_duration <= total_duration:

        end_sec   = start_sec + clip_duration
        gametime  = seconds_to_gametime(start_sec, halftime_sec)
        clip_path = TEMP_DIR / "clip_current.mp4"

        print(f"[Clip {clip_count+1:03d}]  {gametime}  ({start_sec:.0f}s → {end_sec:.0f}s)")

        t0      = time.time()
        success = extract_clip(video_path, start_sec, clip_duration, clip_path)
        elapsed = time.time() - t0

        if not success:
            print(f"  ⚠ ffmpeg failed — skipping")
            start_sec += step
            continue

        print(f"  ✓ clip ready ({elapsed:.2f}s) → {clip_path.name}")

        with open(log_path, "a") as f:
            f.write(f"{clip_count+1},{start_sec},{end_sec},{gametime},{clip_path}\n")

        if on_clip_ready:
            on_clip_ready(clip_path, start_sec, end_sec, gametime)

        delete_clip(clip_path)
        print(f"  ✓ clip deleted\n")

        clip_count += 1
        start_sec  += step

        if test_mode and clip_count >= 3:
            print("  [TEST MODE] stopping after 3 clips.")
            break

    print(f"\n{'='*60}")
    print(f"  Done. Processed {clip_count} clips.")
    print(f"  Log: {log_path}")
    print(f"{'='*60}\n")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video",    type=str, default=None)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--step",     type=int, default=30)
    parser.add_argument("--test",     action="store_true")
    args = parser.parse_args()

    if args.video:
        video_path = Path(args.video)
    elif args.test:
        video_path = TEST_VIDEO
        print("  [TEST MODE] using 224p.mp4")
    else:
        video_path = DEFAULT_VIDEO

    if not video_path.exists():
        print(f"  ERROR: video not found: {video_path}")
        exit(1)

    def dummy_callback(clip_path, start_sec, end_sec, gametime):
        print(f"  → [callback] gametime {gametime}")

    run_sliding_window(
        video_path    = video_path,
        clip_duration = args.duration,
        step          = args.step,
        test_mode     = args.test,
        on_clip_ready = dummy_callback,
    )