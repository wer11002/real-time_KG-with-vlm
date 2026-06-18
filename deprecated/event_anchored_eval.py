"""
event_anchored_eval.py — GT-anchored evaluation pipeline (fix 048)

Uses JAIST GT event times as video clip anchors instead of running the full
sliding-window pipeline.  For each GT event a 60-second clip is extracted
centered on that event's timestamp, the VLM detects actions in the clip, and
the commentator generates text.  AI text is saved alongside GT text for direct
metric comparison with evaluate_commentary.py.

Usage:
    # Full run on subset
    python src/commentator/event_anchored_eval.py

    # Quick test: 3 matches, 10 events each
    python src/commentator/event_anchored_eval.py \\
        --matches Burnley Dortmund Newcastle --max-events 10

    # Then evaluate
    python src/commentator/evaluate_commentary.py \\
        --data-dir data/sn_long_subset/ \\
        --ai-file ai_commentary_anchored.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import requests

# ── path setup ────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "1_video_processor"))
sys.path.insert(0, str(BASE_DIR / "src" / "commentator"))

from action_recognizer import detect_actions         # noqa: E402
from commentator import agent_commentate, LLM_URL    # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────

HALF1_FALLBACK_SEC = 45 * 60   # 2700 s — used when 1_720p.mp4 is absent
CLIP_LEN_DEFAULT   = 60
MIN_WORDS          = 15        # minimum words before triggering a regen


# ── helpers ───────────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _check_llm_server() -> bool:
    """Return True if the LLM server at localhost:8001 responds."""
    try:
        r = requests.get("http://localhost:8001/v1/models", timeout=5)
        return r.status_code < 500
    except Exception:
        return False


def _probe_video_duration(path: Path) -> Optional[float]:
    """Return video duration in seconds via ffprobe, or None on failure."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stderr=subprocess.DEVNULL,
        )
        return float(out.strip())
    except Exception:
        return None


def _half1_duration(match_folder: Path) -> float:
    """Duration of the first half in seconds (probed if possible)."""
    h1 = match_folder / "1_720p.mp4"
    if h1.exists():
        dur = _probe_video_duration(h1)
        if dur:
            return dur
    return float(HALF1_FALLBACK_SEC)


def _abs_seconds(entry: dict, half1_dur: float) -> float:
    """Convert GT entry fields to absolute seconds in the concatenated video."""
    half   = entry.get("half",   1)
    minute = entry.get("minute", 0)
    second = entry.get("second", 0)
    # SoccerNet concatenates 1_720p + 2_720p; half 2 starts at half1_dur.
    return (half - 1) * half1_dur + minute * 60 + second


def _gametime_str(entry: dict) -> str:
    half   = entry.get("half",   1)
    minute = entry.get("minute", 0)
    return f"{'1st' if half == 1 else '2nd'} {minute:02d}:00"


def _extract_clip(
    video_path: Path,
    start: float,
    duration: float,
    out_path: Path,
) -> bool:
    """Extract a clip with ffmpeg. Returns True on success."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss",  str(start),
        "-i",   str(video_path),
        "-t",   str(duration),
        "-c",   "copy",
        "-an",
        str(out_path),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return out_path.exists() and out_path.stat().st_size > 0
    except subprocess.CalledProcessError:
        return False


def _best_detection(detections: list) -> Optional[dict]:
    """Return the first (highest-priority) detection, or None."""
    return detections[0] if detections else None


def _commentate(
    detection: Optional[dict],
    gt_entry: dict,
    match_name: str,
) -> str:
    """Build a SimpleNamespace event and call agent_commentate."""
    if detection:
        action      = detection.get("action") or gt_entry.get("event_type", "Unknown")
        team        = detection.get("team")
        body_part   = detection.get("body_part")
        pitch_zone  = detection.get("pitch_zone")
        outcome     = detection.get("outcome")
        description = detection.get("description", "")
    else:
        # VLM found nothing — fall back to GT metadata for commentary
        action      = gt_entry.get("event_type", "Unknown")
        team        = gt_entry.get("team")
        body_part   = None
        pitch_zone  = None
        outcome     = None
        description = ""

    event_obj = SimpleNamespace(
        action      = action,
        gametime    = _gametime_str(gt_entry),
        player      = gt_entry.get("player"),
        team        = team,
        match_name  = match_name,
        description = description,
        body_part   = body_part,
        pitch_zone  = pitch_zone,
        outcome     = outcome,
    )

    # ttl_path unused in event-anchored mode (fix 047: no tools passed to LLM)
    text = agent_commentate(event_obj, ttl_path="")
    if _word_count(text) < MIN_WORDS:
        text = agent_commentate(
            event_obj,
            ttl_path="",
            extra_hint=(
                "The previous response was too short. "
                "Write 2-3 sentences of broadcast-style commentary "
                "following the example style in the system prompt."
            ),
        )
    return text


def _make_result(entry: dict, detection: Optional[dict], ai_text: str) -> dict:
    return {
        "minute"             : entry.get("minute"),
        "half"               : entry.get("half"),
        "second"             : entry.get("second", 0),
        "gt_text"            : entry.get("human_text", ""),
        "ai_text"            : ai_text,
        "event_type_from_VLM": detection["action"] if detection else "Unknown",
    }


# ── per-match runner ──────────────────────────────────────────────────────

def process_match(
    match_folder: Path,
    clip_len: int,
    max_events: Optional[int],
) -> list:
    """Run anchored evaluation for one match. Returns list of result dicts."""
    gt_path = match_folder / "human_commentary.json"
    video   = match_folder / "720p.mp4"

    if not gt_path.exists():
        print(f"  [SKIP] no human_commentary.json in {match_folder.name}")
        return []
    if not video.exists():
        print(f"  [SKIP] no 720p.mp4 in {match_folder.name}")
        return []

    with open(gt_path, encoding="utf-8") as f:
        gt_entries = json.load(f)

    if max_events:
        gt_entries = gt_entries[:max_events]

    half1_dur  = _half1_duration(match_folder)
    clips_dir  = match_folder / "anchored_clips"
    match_name = match_folder.name

    probed = (match_folder / "1_720p.mp4").exists()
    print(f"  GT events : {len(gt_entries)}")
    print(f"  Half 1 dur: {int(half1_dur)}s ({'probed' if probed else 'fallback 2700s'})")

    n_detected  = 0
    n_commented = 0
    results     = []
    t0          = time.time()

    for i, entry in enumerate(gt_entries):
        abs_sec    = _abs_seconds(entry, half1_dur)
        clip_start = max(0.0, abs_sec - clip_len / 2)
        clip_path  = clips_dir / f"event_{i:04d}.mp4"

        print(
            f"  [{i+1:03d}/{len(gt_entries):03d}] "
            f"H{entry.get('half',1)} {entry.get('minute',0):02d}' "
            f"{entry.get('event_type','?'):<12} "
            f"@ {clip_start:.0f}s",
            end="", flush=True,
        )

        # 1. extract clip
        if not _extract_clip(video, clip_start, clip_len, clip_path):
            print(" [clip FAILED]")
            results.append(_make_result(entry, None, ""))
            continue

        # 2. VLM detection
        try:
            detections = detect_actions(str(clip_path), clip_start_sec=clip_start)
        except Exception as exc:
            print(f" [VLM error: {exc}]")
            detections = []

        best = _best_detection(detections)
        if best:
            n_detected += 1

        # 3. commentator
        try:
            text = _commentate(best, entry, match_name)
            if text:
                n_commented += 1
        except Exception as exc:
            print(f" [commentator error: {exc}]")
            text = ""

        vlm_tag = f"VLM={best['action']}" if best else "VLM=none"
        print(f"  {vlm_tag} → {_word_count(text)}w")

        results.append(_make_result(entry, best, text))

    elapsed = time.time() - t0
    print(
        f"\n  {len(gt_entries)} GT events | "
        f"{n_detected} VLM hits | "
        f"{n_commented} comments | "
        f"{elapsed / 60:.1f} min"
    )
    return results


# ── entry point ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="GT-anchored evaluation: GT times → 60s clips → VLM → commentary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--data-dir",
        default=str(BASE_DIR / "data" / "sn_short"),
        help="Root folder containing per-match sub-folders "
             "(default: data/sn_short/)",
    )
    ap.add_argument(
        "--matches",
        nargs="*",
        metavar="SUBSTR",
        help="Filter by folder-name substring(s), e.g. --matches Burnley Newcastle",
    )
    ap.add_argument(
        "--max-events",
        type=int,
        default=None,
        metavar="N",
        help="Cap GT events per match (useful for quick testing)",
    )
    ap.add_argument(
        "--clip-len",
        type=int,
        default=CLIP_LEN_DEFAULT,
        metavar="SEC",
        help=f"Clip duration in seconds (default: {CLIP_LEN_DEFAULT})",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data dir not found: {data_dir}")
        sys.exit(1)

    # verify LLM server before spending time on VLM inference
    print("Checking LLM server at localhost:8001 ...", end=" ", flush=True)
    if _check_llm_server():
        print("OK\n")
    else:
        print(
            "\n\nERROR: LLM server is not reachable at http://localhost:8001\n"
            "Start the vLLM server first, e.g.:\n"
            "  vllm serve <model-name> --port 8001 --served-model-name commentator\n"
            "Then re-run this script."
        )
        sys.exit(1)

    match_folders = sorted(f for f in data_dir.iterdir() if f.is_dir())
    if args.matches:
        match_folders = [
            f for f in match_folders
            if any(m.lower() in f.name.lower() for m in args.matches)
        ]

    if not match_folders:
        print(
            f"No match folders found in {data_dir}"
            + (f" matching {args.matches}" if args.matches else "")
            + "."
        )
        sys.exit(1)

    print(f"Found {len(match_folders)} match folder(s).\n")

    for folder in match_folders:
        print(f"{'═' * 62}")
        print(f"Match: {folder.name}")

        results = process_match(folder, args.clip_len, args.max_events)
        if not results:
            continue

        out_path = folder / "ai_commentary_anchored.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Saved → {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
