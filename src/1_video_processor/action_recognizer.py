"""
action_recognizer.py
--------------------
Detects soccer actions from a 60-second video clip using Qwen3-VL.

What's new in this version (C2):
    - Kit colors loaded dynamically from ESPN API hex colors
      instead of hardcoded TEAM_COLOR_MAP
    - build_color_map(team1, color1_hex, team2, color2_hex) converts
      ESPN hex → color name → team mapping
    - Works for any match without changing source code

Model: Qwen/Qwen3-VL-30B-A3B-Instruct

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
MODEL_NAME  = "Qwen/Qwen3-VL-30B-A3B-Instruct"
NUM_FRAMES  = 30

VALID_ACTIONS = {"Shot", "Goal", "Foul", "Corner", "Free_Kick", "Substitution", "Offside"}


GOAL_KEYWORDS = {"back of the net", "into the net", "hits the net",
                 "goal", "scores", "scored", "equalis", "1-0", "1-1",
                 "2-0", "2-1", "2-2", "opens the scoring"}

GOAL_VERIFY_PROMPT = """These frames are from a 60-second football clip.
Look at EVERY frame carefully.
Did the ball visibly cross the goal line and enter the net at any point in these frames?
Also check: are players celebrating with arms raised? Is the goalkeeper retrieving the ball from inside the net?

Respond ONLY with this exact JSON:
{"goal_scored": true or false, "evidence": "one sentence describing what you see"}"""


# default color map — used as fallback if ESPN colors not loaded
# overridden at runtime by build_color_map()
_TEAM_COLOR_MAP: Dict[str, Optional[str]] = {}


# ── model singleton ────────────────────────────────────────────────────────
_model     = None
_processor = None
_device    = None


def load_model():
    """Load VLM model and processor. Called once at startup."""
    global _model, _processor, _device

    if _model is not None:
        return _model, _processor, _device

    print(f"  [model] loading {MODEL_NAME}...")

    from pathlib import Path
    from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor

    _device    = "cuda" if torch.cuda.is_available() else "cpu"
    _processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Qwen3-VL-30B MoE needs an explicit offload_folder when device_map="auto"
    # decides to spill some experts to disk (happens on shared-memory chips
    # like Grace-Blackwell where another process holds part of unified RAM).
    offload_dir = Path("data/model_offload")
    offload_dir.mkdir(parents=True, exist_ok=True)

    _model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype       = torch.float16 if _device == "cuda" else torch.float32,
        device_map        = "auto",
        offload_folder    = str(offload_dir),
        offload_state_dict= True,
        trust_remote_code = True,
    )
    _model.eval()
    print(f"  [model] loaded on {_device} ✓  (offload dir: {offload_dir})")

    return _model, _processor, _device


# ═══════════════════════════════════════════════════════════════════════════
# DYNAMIC KIT COLOR MAP (C2 fix)
# ═══════════════════════════════════════════════════════════════════════════

# Named color palette (RGB) for nearest-neighbor matching.
# Covers common football kit colors; nearest Euclidean distance wins.
_COLOR_PALETTE: Dict[str, tuple] = {
    "black"     : (  0,   0,   0),
    "white"     : (255, 255, 255),
    "red"       : (220,  20,  20),
    "maroon"    : (128,   0,   0),
    "orange"    : (230, 100,   0),
    "yellow"    : (240, 210,   0),
    "gold"      : (200, 160,   0),
    "green"     : (  0, 180,   0),
    "dark green": (  0, 100,   0),
    "sky blue"  : (100, 180, 240),
    "blue"      : ( 20,  60, 200),
    "dark blue" : (  0,   0, 110),
    "navy"      : (  0,   0,  60),
    "purple"    : (130,   0, 150),
    "pink"      : (240,  80, 150),
    "gray"      : (130, 130, 130),
}


def hex_to_color_name(hex_color: str) -> str:
    """Convert ESPN hex color to nearest named color via Euclidean RGB distance."""
    hex_color = hex_color.strip("#").lower()
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except (ValueError, IndexError):
        return "unknown"

    best_name = "unknown"
    best_dist = float("inf")
    for name, (pr, pg, pb) in _COLOR_PALETTE.items():
        dist = ((r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def build_color_map(
    team1      : str,
    color1_hex : str,
    team2      : str,
    color2_hex : str,
) -> Dict[str, Optional[str]]:
    """
    Build a color → team mapping from ESPN hex colors.
    Returns empty dict if both teams resolve to the same color name
    (collision) — caller falls back to ambiguous multi-team lookup.
    """
    color1 = hex_to_color_name(color1_hex)
    color2 = hex_to_color_name(color2_hex)

    print(f"  [colors] {team1} → #{color1_hex} → {color1}")
    print(f"  [colors] {team2} → #{color2_hex} → {color2}")

    if color1 == color2:
        print(f"  [colors] WARNING: kit collision — both teams map to '{color1}', color disambiguation disabled")
        return {}

    color_map: Dict[str, Optional[str]] = {
        color1: team1,
        color2: team2,
        "white": None,    # away kits often white — ambiguous
    }
    return color_map


def set_color_map(color_map: Dict[str, Optional[str]]):
    """Set the global color map. Called from main.py at startup."""
    global _TEAM_COLOR_MAP
    _TEAM_COLOR_MAP = color_map


def load_colors_from_game_id(
    game_id: str,
    league : str,
    team1  : str,
    team2  : str,
) -> Dict[str, Optional[str]]:
    """
    Fetch team colors using a pre-resolved game_id + league.
    Skips the full scoreboard scan — call this when ESPNScraper already
    resolved the game_id so we don't hit the ESPN API a third time.
    """
    import requests

    url = (f"http://site.api.espn.com/apis/site/v2/sports/soccer"
           f"/{league}/summary?event={game_id}")
    try:
        r          = requests.get(url, timeout=10)
        data       = r.json()
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

    if total_frames >= num_frames:
        # Density-biased sampling: dense at center, sparse at edges.
        # The event we're describing happens in the middle of the clip
        # (event-anchored mode), so concentrate VLM attention there.
        # Maps t in [0,1] through power curve around 0.5:
        #   u = 2*t - 1          (maps to [-1, 1])
        #   u = sign(u)*|u|^2.5  (pushes toward edges → frames cluster center)
        #   x = 0.5 + 0.5*u     (back to [0, 1])
        # Effect with p=2.5: ~60% of frames in the middle 33% of the clip.
        t = np.linspace(0, 1, num_frames)
        u = 2 * t - 1
        u = np.sign(u) * np.abs(u) ** 2.5
        positions = 0.5 + 0.5 * u
        indices = (positions * (total_frames - 1)).astype(int).tolist()
    else:
        indices = list(range(total_frames))

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


def verify_goal(frames: list, model, processor, device) -> dict:
    """
    Focused second-pass check: did a goal occur in these frames?
    Returns {"goal_scored": bool, "evidence": str}
    """
    try:
        content = []
        for frame in frames:
            import PIL.Image
            if isinstance(frame, np.ndarray):
                pil_img = PIL.Image.fromarray(frame)
            else:
                pil_img = frame
            content.append({"type": "image", "image": pil_img})
        content.append({"type": "text", "text": GOAL_VERIFY_PROMPT})

        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt").to(device)

        import torch
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256)
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        response = processor.decode(generated, skip_special_tokens=True)

        import re, json
        match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"  [goal_verify] error: {e}")
    return {"goal_scored": False, "evidence": ""}


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are analyzing frames from a football broadcast.
Your default answer is {"actions": []}. Only override that default when
you have unmistakable visual proof of a specific action.
A missed detection is always better than a false one.
Always respond in valid JSON format only — no other text outside the JSON."""

ACTION_PROMPT = """These 32 frames are sampled from a 60-second football clip
(more frames from the second half of the clip where key moments resolve)
(one frame roughly every 2 seconds).

IMPORTANT: Most 60-second clips contain NO scorable action — just passing,
running, and positioning. Return {"actions": []} for those clips.
Only report an action when you have clear visual proof from the frames.

DO NOT report:
- A player running, dribbling, or positioned near goal
- A clearance or goalkeeper distribution
- Broadcast replays (score graphic change, slow-motion replay)
- Any action you are less than 65% confident about

Strict per-action criteria — ALL conditions must be visible:

SHOT: foot or head makes CONTACT with the ball AND the ball is moving
  toward goal in the same or adjacent frame. Stance or wind-up is NOT a shot.

GOAL: any of these are visible — ball crossing the goal line OR ball seen
  inside the net after entering, OR attacking players celebrating with arms
  raised and running toward each other while defenders look dejected or stand
  still, OR the goalkeeper retrieves the ball from inside the net, OR the
  referee points to the centre circle. A goal is distinct from a Shot —
  report Goal only when you are confident the ball has already entered the
  net, not just when a shot is being attempted.

FOUL: physical contact is visible AND the fouled player falls to the ground
  OR the referee raises their arm or shows a card.

CORNER: ball is stationary at the corner of the pitch near the
  corner flag AND a player is about to kick it, OR players from
  both teams are clustered inside the penalty box while one player
  stands in the corner area. Broadcast cameras showing the ball
  going to the corner flag area also counts.
  A cross from open play or a long ball to the wing is NOT a corner.

FREE_KICK: referee blows whistle and a player takes a stationary kick,
  OR a wall of defenders is visible near a stationary ball,
  OR players from both teams are standing around a stationary ball waiting
  for a kick to be taken.

SUBSTITUTION: player walking off the pitch while another walks on,
  OR a substitution board with numbers is visible.

OFFSIDE: linesman flag is clearly raised, OR referee signals with arm.

For each confirmed action report jersey number, kit colors, pitch zone, and body part.

Respond ONLY with this JSON (no markdown, no extra text):
{
  "actions": [
    {
      "action": "Shot",
      "frame_index": 5,
      "jersey": "7" or null,
      "team_color": "blue" or "red" or "white" or null,
      "shorts_color": "white" or null,
      "socks_color": "blue" or null,
      "kit_pattern": "solid" or "striped" or "hooped" or null,
      "pitch_zone": "penalty_box" or "edge_of_area" or "midfield" or "own_half" or "wing" or null,
      "body_part": "right_foot" or "left_foot" or "header" or null,
      "outcome": "goal" or "saved_high" or "saved_low" or "saved_side" or "wide_left" or "wide_right" or "over_bar" or "blocked" or "on_target" or null,
      "foul_type": "tackle" or "handball" or "push" or "shirt_pull" or "trip" or "elbow" or null,
      "team_side": "home" or "away" or null,
      "ball_visible": true or false,
      "score": "0-0" or "1-0" or "2-1" etc or null,
      "description": "WHO (jersey# and kit color if visible), WHAT action with WHAT technique (left/right foot, header, slide tackle), WHERE on pitch (be specific: top of box, six-yard box, left flank), WHAT HAPPENED immediately after (saved by keeper, hit post, went wide, goal scored). Example: 'Player #7 in blue cuts inside from the right and drives a low right-footed shot from the edge of the penalty area — saved low by the goalkeeper to his left'"
    }
  ]
}

pitch_zone applies to ALL actions. body_part applies to Shot and Goal only — set null for other actions.
outcome applies to Shot and Goal only — set null for all other actions.
foul_type applies to Foul only — set null for all other actions.
team_side applies to ALL actions — infer from which team appears to be performing the action.
ball_visible applies to ALL actions — true if ball is visible in at least one frame.
score: read the scoreboard overlay if visible — format is "home_goals-away_goals" e.g. "1-0", "2-1". Set null if scoreboard not visible or unreadable.
score applies to ALL actions — always check the scoreboard overlay in every frame.
If no action meets the strict criteria above, return: {"actions": []}"""


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def run_inference(frames: List, model, processor, device: str,
                  recent_context: str = "") -> List[Dict]:
    """Run Qwen3-VL on frames. Returns list of raw action dicts."""
    from qwen_vl_utils import process_vision_info
    from PIL import Image as PILImage

    pil_images = [PILImage.fromarray(f) for f in frames]
    content    = [{"type": "image", "image": img} for img in pil_images]

    prompt = ACTION_PROMPT
    if recent_context:
        prompt += f"\n\n{recent_context}"
    content.append({"type": "text", "text": prompt})

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
            max_new_tokens = 1024,
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
        # strip Qwen3 thinking block if present (outputs <think>...</think> by default)
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
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


def _seconds_to_gametime(seconds: float, halftime_sec: float) -> str:
    """Convert absolute video seconds to match gametime string e.g. '1st 09:37'."""
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


# ═══════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_actions(clip_path: str, clip_start_sec: float = 0.0,
                   halftime_sec: float = 2700.0,
                   recent_events: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Main entry point for the pipeline.
    Returns ALL detected actions in the clip.
    """
    model, processor, device = load_model()

    extracted = extract_frames(clip_path, NUM_FRAMES)
    if not extracted or not extracted[0]:
        return []
    frames, duration_sec, frame_times = extracted

    recent_context = ""
    if recent_events:
        parts = [f"{e.get('action','?')} at {e.get('gametime','?')}"
                 for e in recent_events]
        recent_context = (
            f"Recent detections in the last 2 minutes: {', '.join(parts)}.\n"
            f"Do NOT re-detect the same action type within 30 seconds of a prior detection."
        )

    raw_actions = run_inference(frames, model, processor, device,
                                recent_context=recent_context)
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
            "shorts_color": raw.get("shorts_color"),
            "socks_color" : raw.get("socks_color"),
            "kit_pattern" : raw.get("kit_pattern"),
            "pitch_zone"  : raw.get("pitch_zone"),
            "body_part"   : raw.get("body_part"),
            "outcome"     : raw.get("outcome"),
            "foul_type"   : raw.get("foul_type"),
            "team_side"   : raw.get("team_side"),
            "ball_visible": raw.get("ball_visible"),
            "score"       : raw.get("score"),
            "description" : raw.get("description", ""),
            "video_time"  : video_time,
            "time_in_clip": round(time_in_clip, 1),
            "confidence"  : 1.0,
            "gametime"    : _seconds_to_gametime(video_time, halftime_sec),
        })

    # Way 4: two-pass goal verification
    # If a high-confidence Shot has goal keywords in description,
    # re-run a focused goal check on the same frames.
    has_goal = any(d.get("action") == "Goal" for d in detections)
    if not has_goal:
        for d in detections:
            if d.get("action") != "Shot":
                continue
            desc = str(d.get("description", "")).lower()
            outcome = str(d.get("outcome", "")).lower()
            has_keyword = any(kw in desc for kw in GOAL_KEYWORDS)
            has_outcome = outcome in {"scored", "goal"}
            SAVED_OUTCOMES = {"saved", "blocked", "wide", "over", "saved_high",
                              "saved_low", "off_target", "missed"}
            if outcome in SAVED_OUTCOMES:
                continue
            if has_keyword or has_outcome:
                print(f"  [goal_verify] Shot with goal evidence — running goal check")
                result = verify_goal(frames, model, processor, device)
                if result.get("goal_scored"):
                    print(f"  [goal_verify] GOAL confirmed: {result.get('evidence')}")
                    goal_det = dict(d)
                    goal_det["action"]     = "Goal"
                    goal_det["confidence"] = 1.0
                    goal_det["outcome"]    = "scored"
                    detections.append(goal_det)
                    break  # only one goal per clip
                else:
                    print(f"  [goal_verify] not a goal: {result.get('evidence')}")

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
    print(f"  Running Qwen3-VL multi-action detection...\n")

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