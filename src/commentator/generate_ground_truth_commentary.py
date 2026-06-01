"""
generate_ground_truth_commentary.py
─────────────────────────────────────
Generates contextual ground-truth commentary from Labels-ball.json using
Qwen3-VL-30B in text-only mode (no images).

Usage:
    # Peek at Labels-ball.json schema before running anything
    python src/commentator/generate_ground_truth_commentary.py --peek

    # Single match (partial name ok)
    python src/commentator/generate_ground_truth_commentary.py --match "Blackburn"

    # All matches under data/
    python src/commentator/generate_ground_truth_commentary.py --all
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = BASE_DIR / "data"

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"

KEY_LABELS = {"SHOT", "GOAL", "CORNER", "FREEKICK", "PENALTY"}

# Map SoccerNet label → output event_type
LABEL_TO_TYPE = {
    "SHOT"    : "Shot",
    "GOAL"    : "Goal",
    "CORNER"  : "Corner",
    "FREEKICK": "Free_Kick",
    "PENALTY" : "Penalty",
}


# ── Data discovery ──────────────────────────────────────────────────────────

def find_match_folders() -> list[Path]:
    """
    Return all match folders that contain a Labels-ball.json.
    Handles two layouts:
      - data/<match>/Labels-ball.json  (flat)
      - data/<league>/<season>/<match>/Labels-ball.json  (nested)
    """
    found = []
    for p in DATA_DIR.rglob("Labels-ball.json"):
        found.append(p.parent)
    return sorted(set(found))


def find_match(name_fragment: str) -> Path:
    folders = find_match_folders()
    hits = [f for f in folders if name_fragment.lower() in f.name.lower()]
    if not hits:
        print(f"No match folder found containing '{name_fragment}'")
        print("Available folders:")
        for f in folders:
            print(f"  {f}")
        sys.exit(1)
    if len(hits) > 1:
        print(f"Ambiguous — {len(hits)} folders match '{name_fragment}':")
        for h in hits:
            print(f"  {h}")
        sys.exit(1)
    return hits[0]


# ── Labels-ball.json parsing ────────────────────────────────────────────────

def parse_game_time(game_time: str) -> tuple[int, int, int]:
    """
    "1 - 03:27"  →  (half=1, minute=3, second=27)
    "2 - 07:12"  →  (half=2, minute=7, second=12)
    """
    try:
        half_str, time_str = game_time.split(" - ", 1)
        half = int(half_str.strip())
        mins, secs = time_str.strip().split(":")
        return half, int(mins), int(secs)
    except Exception:
        return 1, 0, 0


def to_abs_second(half: int, minute: int, second: int) -> int:
    return (half - 1) * 45 * 60 + minute * 60 + second


def load_annotations(labels_path: Path) -> list[dict]:
    """
    Parse Labels-ball.json and return sorted key events.
    Infers Goals: a SHOT with no SAVE annotation within 5 s is treated as Goal
    only when no explicit GOAL label exists.
    """
    with open(labels_path, encoding="utf-8") as f:
        data = json.load(f)

    # Accept both top-level key variants
    anns = data.get("annotations") or data.get("events") or []

    # Normalise label to uppercase for comparison
    raw = []
    for a in anns:
        label = str(a.get("label", "")).strip().upper().replace(" ", "")
        game_time = a.get("gameTime", a.get("game_time", ""))
        half, minute, second = parse_game_time(game_time)
        raw.append({
            "label"     : label,
            "half"      : half,
            "minute"    : minute,
            "second"    : second,
            "abs_second": to_abs_second(half, minute, second),
            "team"      : a.get("team", "home"),
            "visibility": a.get("visibility", "visible"),
        })

    # Check for explicit GOAL annotations
    has_explicit_goal = any(r["label"] == "GOAL" for r in raw)

    # Collect SAVE timestamps to filter out SHOT→Goal inference
    save_times = {r["abs_second"] for r in raw if r["label"] == "SAVE"}

    events = []
    for r in raw:
        label = r["label"]
        if label not in KEY_LABELS:
            continue

        # Infer Goal from Shot when no explicit Goals exist
        if label == "SHOT" and not has_explicit_goal:
            nearby_save = any(
                abs(r["abs_second"] - s) <= 5 for s in save_times
            )
            if not nearby_save:
                label = "GOAL"

        events.append({**r, "label": label})

    events.sort(key=lambda e: e["abs_second"])
    return events


# ── Commentary history ──────────────────────────────────────────────────────

def format_history(events: list[dict], current_idx: int) -> str:
    start = max(0, current_idx - 5)
    past  = events[start:current_idx]
    lines = []
    for e in past:
        half_str = "1H" if e["half"] == 1 else "2H"
        lines.append(
            f"[{e['minute']:02d}:{e['second']:02d}] {half_str} "
            f"{e['label']} ({e['team']})"
        )
    return "\n".join(lines) if lines else "No prior events."


# ── Qwen model (lazy-loaded once) ──────────────────────────────────────────

_model     = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    import torch
    from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor

    print(f"Loading {MODEL_ID} (text-only)...")
    _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model     = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    _model.eval()
    print("Model ready.\n")
    return _model, _processor


def generate_commentary(match_name: str, event: dict, history: str) -> str:
    import torch

    model, processor = _load_model()

    half_str  = "1st" if event["half"] == 1 else "2nd"
    team_side = "the home side" if event["team"] == "home" else "the away side"

    system = (
        "You are a football match commentator. Write exactly ONE sentence "
        "of natural commentary. Reference past events when relevant "
        "(e.g. 'his second attempt', 'just minutes after the corner'). "
        "Never invent player names — say 'the home side' or 'the away side'. "
        "Always respond in English only."
    )
    user = (
        f"Match: {match_name}\n"
        f"History (last 5 events):\n{history}\n"
        f"Current event: [{event['minute']:02d}:{event['second']:02d}] "
        f"{half_str} half — {event['label']} by {team_side}\n"
        f"Write one sentence of commentary."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    text   = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80)

    response = processor.decode(
        out[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip()
    return response


# ── Per-match processing ────────────────────────────────────────────────────

def process_match(folder: Path):
    labels_path = folder / "Labels-ball.json"
    if not labels_path.exists():
        print(f"  [skip] No Labels-ball.json in {folder.name}")
        return

    match_name = folder.name
    print(f"\n{'─'*60}")
    print(f"Match : {match_name}")

    events = load_annotations(labels_path)
    key    = [e for e in events if e["label"] in KEY_LABELS | {"GOAL"}]

    if not key:
        print("  [skip] No key events found in labels.")
        return

    print(f"Events: {len(key)} key events")

    output = []
    for i, event in enumerate(key):
        history   = format_history(key, i)
        text      = generate_commentary(match_name, event, history)
        event_type = LABEL_TO_TYPE.get(event["label"], event["label"].capitalize())

        entry = {
            "minute"    : event["minute"],
            "half"      : event["half"],
            "event_type": event_type,
            "team"      : event["team"],
            "human_text": text,
        }
        output.append(entry)

        print(
            f"  [{i+1:02d}/{len(key):02d}] "
            f"{event['minute']:02d}:{event['second']:02d} "
            f"{'1H' if event['half']==1 else '2H'} "
            f"{event['label']:<10} → \"{text}\""
        )

    out_path = folder / "ground_truth_commentary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved {len(output)} entries → {out_path}")


# ── Peek ────────────────────────────────────────────────────────────────────

def peek():
    folders = find_match_folders()
    if not folders:
        print("No Labels-ball.json found under data/")
        return

    path = folders[0] / "Labels-ball.json"
    print(f"\nPeeking at: {path}\n")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print("Top-level keys:", list(data.keys()))
    anns = data.get("annotations") or data.get("events") or []
    print(f"Total annotations: {len(anns)}\n")
    print("First 3 annotations:")
    for a in anns[:3]:
        print(f"  {json.dumps(a, indent=4)}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", help="Partial match folder name")
    ap.add_argument("--all",   action="store_true", help="Process all matches")
    ap.add_argument("--peek",  action="store_true",
                    help="Print Labels-ball.json schema and exit")
    args = ap.parse_args()

    if args.peek:
        peek()
        return

    if args.match:
        folder = find_match(args.match)
        process_match(folder)
    elif args.all:
        folders = find_match_folders()
        print(f"Found {len(folders)} match folder(s).")
        for folder in folders:
            process_match(folder)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
