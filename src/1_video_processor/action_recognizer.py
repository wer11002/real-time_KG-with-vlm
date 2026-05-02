"""
action_recognizer.py
--------------------
Detects soccer actions from a 60-second video clip using Qwen2-VL.

What's new in this version (C2):
    - Kit colors loaded dynamically from ESPN API hex colors
      instead of hardcoded TEAM_COLOR_MAP
    - build_color_map(team1, color1_hex, team2, color2_hex) converts
      ESPN hex → color name → team mapping
    - Works for any match without changing source code

Model: Qwen/Qwen2-VL-7B-Instruct

Setup:
    pip install qwen-vl-utils

Quick test:
    python action_recognizer.py --test
    python action_recognizer.py --clip path/to/clip.mp4
"""

import cv2
import re
import json
import torch
import subprocess
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


# ── model config ───────────────────────────────────────────────────────────
MODEL_NAME  = "Qwen/Qwen2-VL-7B-Instruct"
NUM_FRAMES  = 8

VALID_ACTIONS = {"Shot", "Goal", "Foul", "Corner", "Free_Kick", "Substitution", "Offside"}

# default color map — used as fallback if ESPN colors not loaded
# overridden at runtime by build_color_map()
_TEAM_COLOR_MAP: Dict[str, Optional[str]] = {}


# ── model singleton ────────────────────────────────────────────────────────
_model     = None
_processor = None
_device    = None


def load_model():
    """Load Qwen2-VL model and processor. Called once at startup."""
    global _model, _processor, _device

    if _model is not None:
        return _model, _processor, _device

    print(f"  [model] loading {MODEL_NAME}...")

    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    _device    = "cuda" if torch.cuda.is_available() else "cpu"
    _processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _model     = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype       = torch.float16 if _device == "cuda" else torch.float32,
        device_map        = "auto",
        trust_remote_code = True,
    )
    _model.eval()
    print(f"  [model] loaded on {_device} ✓")

    return _model, _processor, _device


# ═══════════════════════════════════════════════════════════════════════════
# DYNAMIC KIT COLOR MAP (C2 fix)
# ═══════════════════════════════════════════════════════════════════════════

def hex_to_color_name(hex_color: str) -> str:
    """
    Convert ESPN hex color code to a human-readable color name.
    ESPN gives primary kit color as 6-digit hex (e.g. "0000fa" = blue).
    Returns a color name the VLM would use to describe the kit.
    """
    hex_color = hex_color.strip("#").lower()
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except (ValueError, IndexError):
        return "unknown"

    # dominant channel determines color family
    max_channel = max(r, g, b)
    brightness  = (r + g + b) / 3

    if brightness < 40:
        return "black"
    if brightness > 200 and max_channel == brightness:
        return "white"

    if r > g and r > b:
        if r > 150 and g < 80:
            return "red"
        return "orange/red"

    if g > r and g > b:
        if g > 150:
            return "green"
        return "dark green"

    if b > r and b > g:
        if b > 150:
            return "blue"
        return "dark blue"

    if r > 150 and g > 150 and b < 80:
        return "yellow"

    if r > 150 and b > 150 and g < 80:
        return "purple"

    return "mixed"


def build_color_map(
    team1      : str,
    color1_hex : str,
    team2      : str,
    color2_hex : str,
) -> Dict[str, Optional[str]]:
    """
    Build a color → team mapping from ESPN hex colors.
    Called once at pipeline startup with the two teams' primary colors.

    Args:
        team1      : e.g. "Blackburn Rovers"
        color1_hex : e.g. "0000fa" (from ESPN API)
        team2      : e.g. "Nottingham Forest"
        color2_hex : e.g. "c8102e"

    Returns:
        {"blue": "Blackburn Rovers", "red": "Nottingham Forest", "white": None}
    """
    color1 = hex_to_color_name(color1_hex)
    color2 = hex_to_color_name(color2_hex)

    color_map = {
        color1 : team1,
        color2 : team2,
        "white": None,    # away kits often white — ambiguous
    }

    print(f"  [colors] {team1} → #{color1_hex} → {color1}")
    print(f"  [colors] {team2} → #{color2_hex} → {color2}")

    return color_map


def set_color_map(color_map: Dict[str, Optional[str]]):
    """Set the global color map. Called from main.py at startup."""
    global _TEAM_COLOR_MAP
    _TEAM_COLOR_MAP = color_map


def load_colors_from_espn(date: str, team1: str, team2: str) -> Dict[str, Optional[str]]:
    """
    Fetch team colors from ESPN API and build the color map.
    Tries exact date first, then ±1 day.
    Falls back to empty map if API fails.
    """
    import requests
    from thefuzz import fuzz
    from datetime import datetime, timedelta

    LEAGUES  = ["eng.2", "eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "uefa.champions"]
    date_str = date.replace("-", "")

    try:
        base_date  = datetime.strptime(date_str, "%Y%m%d")
        candidates = [
            date_str,
            (base_date + timedelta(days=1)).strftime("%Y%m%d"),
            (base_date - timedelta(days=1)).strftime("%Y%m%d"),
        ]
    except ValueError:
        candidates = [date_str]

    game_id = None
    league  = None

    for search_date in candidates:
        for lg in LEAGUES:
            url = (f"http://site.api.espn.com/apis/site/v2/sports/soccer"
                   f"/{lg}/scoreboard?dates={search_date}")
            try:
                r = requests.get(url, timeout=10)
                if r.status_code != 200:
                    continue
                for event in r.json().get("events", []):
                    name = event.get("name", "").lower()
                    s1   = fuzz.token_set_ratio(team1.lower(), name)
                    s2   = fuzz.token_set_ratio(team2.lower(), name)
                    if s1 > 80 and s2 > 80:
                        game_id = event.get("id")
                        league  = lg
                        break
            except Exception:
                continue
            if game_id:
                break
        if game_id:
            break

    if not game_id:
        print(f"  [colors] WARNING: match not found — color map empty")
        return {}

    url = (f"http://site.api.espn.com/apis/site/v2/sports/soccer"
           f"/{league}/summary?event={game_id}")
    try:
        r    = requests.get(url, timeout=10)
        data = r.json()
        teams_data = {}
        for team_entry in data.get("rosters", []):
            t    = team_entry.get("team", {})
            name = t.get("displayName", "")
            col  = t.get("color", "")
            if name and col:
                teams_data[name] = col

        if len(teams_data) == 2:
            names  = list(teams_data.keys())
            colors = list(teams_data.values())
            color_map = build_color_map(names[0], colors[0], names[1], colors[1])
            set_color_map(color_map)
            return color_map

    except Exception as e:
        print(f"  [colors] WARNING: failed to load colors: {e}")

    return {}


# ═══════════════════════════════════════════════════════════════════════════
# FRAME EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_frames(clip_path: str, num_frames: int = NUM_FRAMES):
    """Sample num_frames evenly. Returns (frames, duration_sec, frame_times)."""
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"  [frames] ERROR: cannot open {clip_path}")
        return None, 0, []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    duration_sec = total_frames / fps

    indices = (np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
               if total_frames >= num_frames else list(range(total_frames)))

    frames, frame_times = [], []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_times.append(idx / fps if fps > 0 else 0.0)

    cap.release()

    while len(frames) < num_frames and frames:
        frames.append(frames[-1])
        frame_times.append(frame_times[-1])

    return frames, duration_sec, frame_times


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a soccer match analyst.
Analyze video frames from a soccer match and identify ALL key events visible.
Always respond in valid JSON format only — no other text outside the JSON."""

ACTION_PROMPT = """These 8 frames are sampled evenly from a 60-second soccer clip.
Frame 1 = 0s, frame 2 = ~8s, frame 3 = ~15s, frame 4 = ~23s,
frame 5 = ~30s, frame 6 = ~38s, frame 7 = ~45s, frame 8 = ~52s.

Identify EVERY key soccer action visible across these frames:
- Shot (attempt on goal, header, blocked shot)
- Goal (ball in net, celebration after a goal)
- Foul (tackle, trip, push, handball, contact)
- Corner (corner kick being taken)
- Free_Kick (free kick being taken)
- Substitution (player being replaced)
- Offside (offside call, linesman flag)

For each action, provide the jersey number if readable and the kit color.
Multiple actions can occur in the same clip — include all of them.

Respond ONLY with this JSON (no markdown, no extra text):
{
  "actions": [
    {
      "action": "Shot",
      "frame_index": 5,
      "jersey": "7" or null,
      "team_color": "blue" or "red" or "white" or null,
      "description": "one sentence describing the action",
      "confidence": 0.0 to 1.0
    }
  ]
}

If no clear soccer actions are visible, return: {"actions": []}"""


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def run_inference(frames: List, model, processor, device: str) -> List[Dict]:
    """Run Qwen2-VL on frames. Returns list of raw action dicts."""
    from qwen_vl_utils import process_vision_info
    from PIL import Image as PILImage

    pil_images = [PILImage.fromarray(f) for f in frames]
    content    = [{"type": "image", "image": img} for img in pil_images]
    content.append({"type": "text", "text": ACTION_PROMPT})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text          = [text],
        images        = image_inputs,
        videos        = video_inputs,
        padding       = True,
        return_tensors= "pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens = 500,
            do_sample      = False,
            temperature    = None,
            top_p          = None,
        )

    generated = output_ids[:, inputs.input_ids.shape[1]:]
    response  = processor.batch_decode(
        generated, skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    try:
        response = re.sub(r"```json|```", "", response).strip()
        parsed   = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except Exception:
                print(f"  [vlm] JSON parse failed: {response[:120]}")
                return []
        else:
            print(f"  [vlm] no JSON found: {response[:120]}")
            return []

    actions = parsed.get("actions", [])
    return [a for a in actions if isinstance(a, dict)
            and a.get("action") in VALID_ACTIONS]


# ═══════════════════════════════════════════════════════════════════════════
# RESOLVE TEAM FROM COLOR
# ═══════════════════════════════════════════════════════════════════════════

def resolve_team(team_color: Optional[str]) -> Optional[str]:
    """
    Map kit color string to team name using the dynamic color map.
    Falls back to None if color not in map or map is empty.
    """
    if not team_color or not _TEAM_COLOR_MAP:
        return None
    color_lower = team_color.lower().strip()
    # exact match first
    if color_lower in _TEAM_COLOR_MAP:
        return _TEAM_COLOR_MAP[color_lower]
    # partial match (e.g. "blue/white" matches "blue")
    for key, team in _TEAM_COLOR_MAP.items():
        if key and key in color_lower:
            return team
    return None


# ═══════════════════════════════════════════════════════════════════════════
# TIME ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════

def estimate_time(frame_index, frame_times: List[float], duration_sec: float) -> float:
    if frame_index is None:
        return duration_sec / 2.0
    try:
        idx = int(frame_index) - 1
    except (ValueError, TypeError):
        return duration_sec / 2.0
    if 0 <= idx < len(frame_times):
        return frame_times[idx]
    return duration_sec / 2.0


# ═══════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_actions(clip_path: str, clip_start_sec: float = 0.0) -> List[Dict]:
    """
    Main entry point for the pipeline.
    Returns ALL detected actions in the clip.
    """
    model, processor, device = load_model()

    extracted = extract_frames(clip_path, NUM_FRAMES)
    if not extracted or not extracted[0]:
        return []
    frames, duration_sec, frame_times = extracted

    raw_actions = run_inference(frames, model, processor, device)
    if not raw_actions:
        return []

    detections = []
    for raw in raw_actions:
        action = raw.get("action")
        if action not in VALID_ACTIONS:
            continue

        time_in_clip = estimate_time(raw.get("frame_index"), frame_times, duration_sec)
        video_time   = clip_start_sec + time_in_clip
        team_color   = raw.get("team_color")
        team         = resolve_team(team_color)

        detections.append({
            "action"      : action,
            "jersey"      : raw.get("jersey"),
            "team"        : team,
            "team_color"  : team_color,
            "description" : raw.get("description", ""),
            "video_time"  : video_time,
            "time_in_clip": round(time_in_clip, 1),
            "confidence"  : float(raw.get("confidence", 0.5)),
        })

    return detections


# ═══════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=str, default=None)
    parser.add_argument("--test", action="store_true",
                        help="Extract test clip at 9:00")
    args = parser.parse_args()

    # test color loading
    print("── Color map test ──")
    color_map = load_colors_from_espn(
        date  = "2019-10-01",
        team1 = "Blackburn Rovers",
        team2 = "Nottingham Forest",
    )
    print(f"  Color map: {color_map}\n")

    # test resolve
    print("── Color resolution test ──")
    tests = [
        ("blue",       "Blackburn Rovers"),
        ("blue/white", "Blackburn Rovers"),
        ("red",        "Nottingham Forest"),
        ("white",      None),
        ("yellow",     None),
    ]
    for color, expected in tests:
        result = resolve_team(color)
        ok     = result == expected
        print(f"  {'✓' if ok else '✗'} '{color}' → {result}  (expected: {expected})")

    if not args.clip and not args.test:
        print("\n  (skipping VLM inference — no --clip or --test flag)")
        print("  Usage: python action_recognizer.py --test")
        exit(0)

    if args.test:
        import os
        BASE      = Path(__file__).parent.parent.parent
        video     = str(BASE / "data/2019-10-01 - Blackburn Rovers - Nottingham Forest/720p.mp4")
        clip_path = str(BASE / "data/temp/test_clip_vlm.mp4")
        os.makedirs(str(BASE / "data/temp"), exist_ok=True)
        print("\n  [test] extracting clip at 9:00 → 10:00")
        subprocess.run([
            "ffmpeg", "-ss", "540", "-i", video,
            "-t", "60", "-c", "copy", "-y", clip_path
        ], capture_output=True)
        clip_start = 540.0
    else:
        clip_path  = args.clip
        clip_start = 0.0

    print(f"\n  Clip   : {clip_path}")
    print(f"  Running Qwen2-VL multi-action detection...\n")

    actions = detect_actions(clip_path, clip_start_sec=clip_start)

    if actions:
        print(f"  Detected {len(actions)} action(s):\n")
        for i, a in enumerate(actions, 1):
            print(f"  Action {i}:")
            print(f"    Type        : {a['action']}")
            print(f"    Jersey      : #{a['jersey']}" if a['jersey'] else "    Jersey      : not detected")
            print(f"    Team        : {a['team'] or a['team_color'] or 'unknown'}")
            print(f"    Description : {a['description']}")
            print(f"    Video time  : {a['video_time']:.1f}s")
            print(f"    Confidence  : {a['confidence']:.2f}")
            print()
    else:
        print("  No soccer actions detected.")

    print("  Done!")