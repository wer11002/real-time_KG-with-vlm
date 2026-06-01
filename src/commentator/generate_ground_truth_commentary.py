"""
generate_ground_truth_commentary.py
─────────────────────────────────────
Generates ground-truth commentary from the ESPN CSV using Qwen3-VL-30B
in text-only mode.

Each event gets a broadcast sentence that is:
  - Factually anchored to the ESPN Full_Text (player, action, outcome)
  - Contextually aware of the last 5 events ("just after the corner",
    "his second attempt in five minutes")

Output saved as data/<match>/ground_truth_commentary.json — same format
as human_commentary.json so evaluate_commentary.py can consume it directly.

Usage:
    python src/commentator/generate_ground_truth_commentary.py \\
        --csv data/blackburn_forest_2019-10-01.csv \\
        --match "Blackburn Rovers vs Nottingham Forest"

    # Generate for all CSVs found under data/
    python src/commentator/generate_ground_truth_commentary.py --all
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = BASE_DIR / "data"

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"

# Event types we generate commentary for (skip substitutions/offsides for brevity)
COMMENTARY_TYPES = {"Shot", "Goal", "Corner", "Free_Kick", "Foul"}

SYSTEM_PROMPT = (
    "You are a live football match commentator on British TV. "
    "Write exactly ONE sentence of natural broadcast commentary for the current event. "
    "Rules:\n"
    "1. Use the ESPN description as your factual source — do not invent new facts.\n"
    "2. Reference recent history when natural: 'another attempt', "
    "'just minutes after the corner', 'following the free kick'.\n"
    "3. Vary your language — avoid repeating the same opener.\n"
    "4. Be concise and vivid. One sentence only.\n"
    "5. Always respond in English only."
)


# ── ESPN CSV parser ─────────────────────────────────────────────────────────

def parse_espn_csv(path: str) -> list[dict]:
    """
    Columns: Time, Player, Team, Action_Type, Yellow_Card, Red_Card, Full_Text
    Time has trailing apostrophe e.g. "63'" → strip it → float.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_time = row.get("Time", "0").strip().rstrip("'")
            try:
                t = float(raw_time)
            except ValueError:
                continue
            action = row.get("Action_Type", "").strip()
            if not action or action == "None":
                continue
            rows.append({
                "minute"     : t,
                "half"       : 1 if t <= 45 else 2,
                "half_minute": t if t <= 45 else t - 45,
                "event_type" : action,
                "player"     : row.get("Player", "").strip(),
                "team"       : row.get("Team",   "").strip(),
                "full_text"  : row.get("Full_Text", "").strip(),
            })
    return sorted(rows, key=lambda r: r["minute"])


# ── Model (lazy-loaded once) ────────────────────────────────────────────────

_model     = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    import torch
    from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor

    print(f"Loading {MODEL_ID} (text-only)…")
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


def _generate(messages: list[dict]) -> str:
    import torch
    model, processor = _load_model()

    text   = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80)

    return processor.decode(
        out[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip()


# ── History builder ─────────────────────────────────────────────────────────

def _format_history(past_events: list[dict]) -> str:
    """
    Format the last N events as a bullet list for the Qwen prompt.
    Each line: "63' 2H  GOAL  Adam Armstrong (Blackburn Rovers) — 'Armstrong fires...' "
    """
    lines = []
    for e in past_events[-5:]:
        half_str = "1H" if e["half"] == 1 else "2H"
        snippet  = e["full_text"][:120].rstrip(".") if e["full_text"] else "—"
        lines.append(
            f"  {int(e['minute'])}' {half_str}  {e['event_type']:<10} "
            f"{e['player'] or '?'} ({e['team'] or '?'}) — \"{snippet}\""
        )
    return "\n".join(lines) if lines else "  (no prior events)"


# ── Commentary generation ───────────────────────────────────────────────────

def generate_commentary(event: dict, history: list[dict], match_name: str) -> str:
    half_str = "1st" if event["half"] == 1 else "2nd"
    hist_str = _format_history(history)

    user = (
        f"Match: {match_name}\n\n"
        f"Recent events (oldest → newest):\n{hist_str}\n\n"
        f"Current event — {int(event['minute'])}' ({half_str} half):\n"
        f"  Type   : {event['event_type']}\n"
        f"  Player : {event['player'] or 'Unknown'}\n"
        f"  Team   : {event['team'] or 'Unknown'}\n"
        f"  ESPN   : \"{event['full_text']}\"\n\n"
        f"Write one sentence of broadcast commentary for this event, "
        f"referencing recent history where natural."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user},
    ]
    return _generate(messages)


# ── Per-CSV processing ──────────────────────────────────────────────────────

def process_csv(csv_path: Path, match_name: str, out_dir: Path):
    print(f"\n{'─'*62}")
    print(f"CSV   : {csv_path.name}")
    print(f"Match : {match_name}")

    all_events = parse_espn_csv(str(csv_path))
    key_events = [e for e in all_events if e["event_type"] in COMMENTARY_TYPES]

    if not key_events:
        print("  [skip] No commentary-worthy events found in CSV.")
        return

    print(f"Events: {len(key_events)} (from {len(all_events)} total ESPN rows)\n")

    output  = []
    history = []   # grows as we process — Qwen sees what was "said" before

    for i, event in enumerate(key_events):
        text = generate_commentary(event, history, match_name)

        entry = {
            "minute"    : int(event["minute"]),
            "half"      : event["half"],
            "event_type": event["event_type"],
            "player"    : event["player"],
            "team"      : event["team"],
            "human_text": text,
        }
        output.append(entry)

        # add this event to history so the next call sees it
        history.append({**event, "full_text": text})  # use generated sentence as "memory"

        half_str = "1H" if event["half"] == 1 else "2H"
        print(
            f"  [{i+1:02d}/{len(key_events):02d}] "
            f"{int(event['minute'])}' {half_str} "
            f"{event['event_type']:<10} "
            f"→ \"{text}\""
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ground_truth_commentary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved {len(output)} entries → {out_path}")


# ── helpers (mirrors generate_espn_csv.py) ──────────────────────────────────

def _parse_match_folder(folder: Path):
    """'YYYY-MM-DD - Team One - Team Two' → (date, team1, team2) or None."""
    parts = folder.name.split(" - ", 2)
    if len(parts) != 3:
        return None
    date, team1, team2 = parts
    if not (len(date) == 10 and date[4] == "-" and date[7] == "-"):
        return None
    return date.strip(), team1.strip(), team2.strip()


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _csv_path_for(date: str, team1: str, team2: str) -> Path:
    return DATA_DIR / f"{date}_{_slug(team1)}_{_slug(team2)}.csv"


# ── CSV discovery ───────────────────────────────────────────────────────────

def find_all_csvs() -> list[tuple[Path, str, Path]]:
    """
    Returns list of (csv_path, match_name, out_dir) tuples.
    Iterates match folders (YYYY-MM-DD - Team1 - Team2) and derives the
    expected CSV path from the same slug logic used by generate_espn_csv.py.
    Skips folders whose CSV does not exist.
    """
    results = []
    for folder in sorted(DATA_DIR.iterdir()):
        if not folder.is_dir():
            continue
        parsed = _parse_match_folder(folder)
        if not parsed:
            continue
        date, team1, team2 = parsed
        csv_path = _csv_path_for(date, team1, team2)
        if not csv_path.exists():
            print(f"[SKIP]  {folder.name}")
            print(f"        → CSV not found: {csv_path.name}")
            continue
        results.append((csv_path, folder.name, folder))
    return results


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",   help="Path to ESPN CSV file")
    ap.add_argument("--match", help="Match name (used in Qwen prompt + output path)")
    ap.add_argument("--out",   help="Output directory (default: data/<match>/)")
    ap.add_argument("--all",   action="store_true",
                    help="Process all CSVs found under data/")
    args = ap.parse_args()

    if args.all:
        entries = find_all_csvs()
        if not entries:
            print("No CSV files found under data/")
            sys.exit(1)
        print(f"Found {len(entries)} CSV file(s).")
        for csv_path, match_name, out_dir in entries:
            process_csv(csv_path, match_name, out_dir)
        return

    if not args.csv:
        # Auto-detect: prefer unique CSV; otherwise require explicit --csv
        csvs = list(DATA_DIR.glob("*.csv"))
        if len(csvs) == 1:
            args.csv = str(csvs[0])
            print(f"Auto-detected CSV: {csvs[0].name}")
        else:
            ap.print_help()
            sys.exit(1)

    csv_path = Path(args.csv)

    if args.out:
        out_dir    = Path(args.out)
        match_name = args.match or csv_path.stem.replace("_", " ").title()
    else:
        # Find the match folder whose slug-derived CSV path matches this file.
        out_dir    = None
        match_name = args.match or csv_path.stem.replace("_", " ").title()
        for folder in DATA_DIR.iterdir():
            if not folder.is_dir():
                continue
            parsed = _parse_match_folder(folder)
            if not parsed:
                continue
            date, team1, team2 = parsed
            if _csv_path_for(date, team1, team2) == csv_path.resolve():
                out_dir    = folder
                match_name = args.match or folder.name
                break
        if out_dir is None:
            out_dir = DATA_DIR / csv_path.stem

    process_csv(csv_path, match_name, out_dir)


if __name__ == "__main__":
    main()
