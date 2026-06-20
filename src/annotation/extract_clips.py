"""
extract_clips.py — Cut 10-second video clips around each event timestamp.
Each clip = 5 seconds before event + 5 seconds after.
"""

import json
import subprocess
from pathlib import Path

CLIP_BEFORE = 5    # seconds before event
CLIP_AFTER  = 5    # seconds after event
CLIP_LEN    = CLIP_BEFORE + CLIP_AFTER

def event_to_seconds(event: dict) -> float:
    """Convert event's minute/half to absolute seconds in the match video."""
    half_offset = 0 if event["half"] == 1 else 45 * 60   # second half starts at 45 min
    minute = float(event["minute"])
    return half_offset + minute * 60

def extract_clip(video_path: Path, event: dict, out_dir: Path) -> Path:
    """Extract a 10-second clip centred on the event time."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{event['rating_id']}.mp4"
    
    if out.exists():
        return out  # idempotent
    
    event_sec = event_to_seconds(event)
    start_sec = max(0, event_sec - CLIP_BEFORE)
    
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-ss", str(start_sec),
        "-i", str(video_path),
        "-t", str(CLIP_LEN),
        "-c:v", "libx264", "-preset", "ultrafast",
        "-an",                # no audio (faster, smaller)
        "-vf", "scale=640:-2", # downscale to 640px wide for web
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [err] ffmpeg failed for {event['rating_id']}: {result.stderr[:200]}")
        return None
    return out

def main():
    events = json.load(open("data/annotation/events_to_rate.json"))
    clips_dir = Path("data/annotation/clips")
    clips_dir.mkdir(parents=True, exist_ok=True)
    
    succeeded, failed = 0, 0
    for ev in events:
        match_dir = Path(ev["_match_dir"])
        video = match_dir / "720p.mp4"
        if not video.exists():
            video = match_dir / "224p.mp4"   # fallback
        if not video.exists():
            print(f"  [skip] no video in {match_dir.name}")
            failed += 1
            continue
        
        out = extract_clip(video, ev, clips_dir)
        if out:
            ev["_clip"] = str(out.resolve())
            succeeded += 1
            if succeeded % 20 == 0:
                print(f"  [progress] {succeeded}/{len(events)} clips extracted")
        else:
            failed += 1
    
    json.dump(events, open("data/annotation/events_to_rate.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"\n  Extracted {succeeded} clips, {failed} failed.")
    print(f"  Clips saved in {clips_dir}")


if __name__ == "__main__":
    main()
