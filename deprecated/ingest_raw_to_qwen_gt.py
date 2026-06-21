"""
ingest_raw_to_qwen_gt.py
─────────────────────────────────────────────────────────────────────
Per-event contextual ground-truth synthesis.

For every match in data/kg_output/ekg.ttl this script:

    1. Pulls the event timeline from the KG (ordered by period, minute).
    2. Loads the three raw human sources scraped earlier:
         data/<match>/groundtruth_sources/sky_report.txt
         data/<match>/groundtruth_sources/bbc_report.txt   (optional)
         data/<match>/groundtruth_sources/espn_full_commentary.json
    3. For each KG event, builds a context bundle of
         — the last 5 KG events (rolling history)
         — the full Sky and BBC narratives
         — ESPN entries within ±5 minutes of the current event
    4. Calls Qwen3-VL-30B in text-only mode (reusing the loader from
       generate_ground_truth_commentary.py — model loaded once for the
       whole run) to write ONE sentence of contextual commentary
       grounded in the human sources.

Output (per match):
    data/<match>/qwen_ground_truth_commentary_v2.json

    [
      {"minute": 63, "half": 2, "event_type": "Goal",
       "player": "Adam Armstrong", "team": "Blackburn Rovers",
       "human_text": "Armstrong gets his reward — third effort of the half ..."},
      ...
    ]

The "_v2" suffix preserves the older ground_truth_commentary.json
generated from ESPN events alone, so we can compare downstream.

Usage:
    python src/commentator/ingest_raw_to_qwen_gt.py --all
    python src/commentator/ingest_raw_to_qwen_gt.py --match "Blackburn"
"""

import argparse
import json
import re
import sys
from pathlib import Path

from rdflib import Graph

# Reuse the Qwen loader from the sibling GT generator. Importing the
# private functions keeps the cached _model / _processor singletons in
# that module so the 30B weights are only paged in once for the run.
from generate_ground_truth_commentary import _load_model, _generate

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
TTL_PATH = DATA_DIR / "kg_output" / "ekg.ttl"

ESPN_WINDOW_MIN = 5
PAST_EVENTS     = 5

SYSTEM_PROMPT = (
    "You are generating ground truth football commentary for research. "
    "You will be given multiple real human sources (Sky Sports report, "
    "BBC report, ESPN play-by-play) plus the last 5 events that "
    "occurred. "
    "Write exactly ONE sentence of contextual commentary for the "
    "current event, grounded in what the real human sources actually "
    "said. "
    "Reference past events when relevant (e.g. 'his second attempt', "
    "'minutes after the corner', 'again'). "
    "Never invent player names not present in the sources. "
    "Always respond in English only."
)


# ════════════════════════════════════════════════════════════════════
# KG parsing helpers (mirror extract_ai_commentary_per_match.py)
# ════════════════════════════════════════════════════════════════════

GAMETIME_RE = re.compile(r"(\d+)(?:st|nd|rd|th)\s+(\d+):(\d+)")


def _parse_gametime(gametime: str) -> tuple[int, int, int]:
    """'1st 18:05' → (1, 18, 5). Returns (0, 0, 0) on parse failure."""
    m = GAMETIME_RE.search(str(gametime))
    if not m:
        return 0, 0, 0
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _coerce_int(value, fallback: int) -> int:
    """int(literal) but tolerant of bogus boolean / string values."""
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return fallback


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def find_folder_for_match(match_uri: str, label: str) -> Path | None:
    candidates = [f for f in DATA_DIR.iterdir() if f.is_dir()]
    if label:
        for f in candidates:
            if f.name == label:
                return f
        for f in candidates:
            if label.lower() in f.name.lower():
                return f
    slug = match_uri.rstrip("/").split("/")[-1].split("#")[-1]
    if slug.startswith("match_"):
        slug = slug[len("match_"):]
    slug_norm = _norm(slug)
    for f in candidates:
        if _norm(f.name) == slug_norm:
            return f
    for f in candidates:
        fnorm = _norm(f.name)
        if slug_norm and (slug_norm in fnorm or fnorm in slug_norm):
            return f
    return None


# ════════════════════════════════════════════════════════════════════
# KG queries
# ════════════════════════════════════════════════════════════════════

def load_matches(g: Graph) -> list[tuple[str, str]]:
    q = """
    PREFIX ekg:  <http://soccerekg.org/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?match ?label WHERE {
        ?match a ekg:Match .
        OPTIONAL { ?match rdfs:label ?label }
    }
    """
    return [(str(r.match), str(r.label) if r.label else "")
            for r in g.query(q)]


def load_events_for_match(g: Graph, match_uri: str) -> list[dict]:
    q = """
    PREFIX ekg:  <http://soccerekg.org/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?e ?type ?minute ?period ?gametime ?playerLabel ?teamLabel WHERE {
        ?e ekg:inMatch      <%s> ;
           ekg:hasEventType  ?type ;
           ekg:hasMinute     ?minute ;
           ekg:hasPeriodNumber     ?period ;
           ekg:hasTime       ?gametime .
        OPTIONAL {
            { ?e ekg:isPerformedBy ?p } UNION { ?p ekg:performed ?e }
            ?p rdfs:label ?playerLabel .
        }
        OPTIONAL {
            ?e ekg:involvedTeam ?t .
            ?t rdfs:label      ?teamLabel .
        }
    }
    ORDER BY ?period ?minute ?gametime ?e
    """ % match_uri

    events = []
    for r in g.query(q):
        gametime  = str(r.gametime) if r.gametime else ""
        gh, gm, gs = _parse_gametime(gametime)
        half      = _coerce_int(r.period, gh)
        minute    = _coerce_int(r.minute, gm)
        if half not in (1, 2):
            half = gh if gh in (1, 2) else 1
        events.append({
            "uri"       : str(r.e),
            "event_type": str(r.type),
            "minute"    : minute,
            "second"    : gs,
            "half"      : half,
            "abs_minute": minute + (45 if half == 2 else 0),
            "gametime"  : gametime,
            "player"    : str(r.playerLabel) if r.playerLabel else "",
            "team"      : str(r.teamLabel)   if r.teamLabel   else "",
        })
    return events


# ════════════════════════════════════════════════════════════════════
# Source loaders
# ════════════════════════════════════════════════════════════════════

def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _read_espn(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _flatten(text: str) -> str:
    """Collapse whitespace so a multi-paragraph report fits cleanly in
    a single prompt block without confusing newlines."""
    return re.sub(r"\s+", " ", text).strip()


def _espn_near(espn: list[dict], abs_minute: int,
                window: int = ESPN_WINDOW_MIN) -> list[dict]:
    """Return ESPN entries within ±window minutes, sorted by minute."""
    out = []
    for e in espn:
        m = e.get("minute_absolute")
        if isinstance(m, (int, float)) and abs(m - abs_minute) <= window:
            out.append(e)
    return sorted(out, key=lambda e: int(e.get("minute_absolute", 0)))


# ════════════════════════════════════════════════════════════════════
# Prompt building
# ════════════════════════════════════════════════════════════════════

def _format_history(past: list[dict]) -> str:
    if not past:
        return "  (none — this is the first event of the match)"
    lines = []
    for e in past:
        half_str = "1H" if e["half"] == 1 else "2H"
        lines.append(
            f"  {e['minute']:>2d}' {half_str}  "
            f"{e['event_type']:<12} "
            f"{e['player'] or '?'} ({e['team'] or '?'})"
        )
    return "\n".join(lines)


def _format_espn_window(entries: list[dict]) -> str:
    if not entries:
        return "  (no ESPN entries within the ±5 minute window)"
    lines = []
    for e in entries:
        minute = e.get("minute") or f"{e.get('minute_absolute', '?')}'"
        guess  = e.get("event_type_guess", "")
        text   = e.get("text", "")
        lines.append(f"  [{minute}] {guess} — {text}")
    return "\n".join(lines)


def build_prompt(event: dict, history: list[dict], match_name: str,
                 sky_text: str, bbc_text: str,
                 espn_entries: list[dict]) -> list[dict]:
    half_str = "1st" if event["half"] == 1 else "2nd"
    nearby   = _espn_near(espn_entries, event["abs_minute"])

    user = (
        f"Match: {match_name}\n\n"
        f"Current event: [{half_str} "
        f"{event['minute']:02d}:{event['second']:02d}] "
        f"{event['event_type']} "
        f"by {event['player'] or 'unidentified'} "
        f"({event['team'] or 'unknown team'})\n\n"
        f"Past 5 events:\n{_format_history(history)}\n\n"
        f"Real human sources:\n\n"
        f"--- SKY SPORTS REPORT ---\n"
        f"{_flatten(sky_text) if sky_text else '[not available for this match]'}\n\n"
        f"--- BBC REPORT ---\n"
        f"{_flatten(bbc_text) if bbc_text else '[not available for this match]'}\n\n"
        f"--- ESPN PLAY-BY-PLAY (entries near this minute) ---\n"
        f"{_format_espn_window(nearby)}\n\n"
        "Write one sentence of contextual commentary for the current event."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user},
    ]


# ════════════════════════════════════════════════════════════════════
# Per-match processing
# ════════════════════════════════════════════════════════════════════

def process_match(g: Graph, match_uri: str, match_label: str) -> dict:
    folder = find_folder_for_match(match_uri, match_label)
    name   = match_label or match_uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    if folder is None:
        print(f"  [skip] No folder for {name}")
        return {"status": "no_folder"}

    events = load_events_for_match(g, match_uri)
    if not events:
        print(f"  [skip] {folder.name} — no KG events")
        return {"status": "no_events"}

    src_dir   = folder / "groundtruth_sources"
    sky_text  = _read_text(src_dir / "sky_report.txt")
    bbc_text  = _read_text(src_dir / "bbc_report.txt")
    espn_data = _read_espn(src_dir / "espn_full_commentary.json")

    print(f"\n{'─'*72}")
    print(f"  MATCH : {folder.name}")
    print(f"  Sources: sky={len(sky_text)}c bbc={len(bbc_text)}c "
          f"espn={len(espn_data)} entries")
    print(f"  Events: {len(events)}")
    print(f"{'─'*72}")

    output  : list[dict] = []
    history : list[dict] = []
    for i, ev in enumerate(events, 1):
        messages = build_prompt(ev, history, folder.name,
                                sky_text, bbc_text, espn_data)
        text       = _generate(messages)
        half_label = "1st" if ev["half"] == 1 else "2nd"
        print(f"  [{i:>3d}/{len(events)}] {half_label} "
              f"{ev['minute']:02d}:{ev['second']:02d} "
              f"{ev['event_type']:<12} → \"{text[:120]}\"")

        output.append({
            "minute"    : ev["minute"],
            "half"      : ev["half"],
            "event_type": ev["event_type"],
            "player"    : ev["player"],
            "team"      : ev["team"],
            "human_text": text,
        })
        history.append({**ev, "generated_text": text})
        history = history[-PAST_EVENTS:]

    out_path = folder / "qwen_ground_truth_commentary_v2.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n  Saved {len(output)} entries → {out_path}")
    return {"status": "ok", "events": len(output), "path": str(out_path)}


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Synthesise contextual ground-truth commentary by "
                    "calling Qwen on each KG event with raw Sky / BBC / "
                    "ESPN sources as grounding.",
    )
    ap.add_argument("--all",   action="store_true",
                    help="Process every match in the KG")
    ap.add_argument("--match", help="Partial folder-name filter (e.g. 'Blackburn')")
    args = ap.parse_args()

    if not args.all and not args.match:
        ap.print_help()
        sys.exit(1)

    if not TTL_PATH.exists():
        print(f"TTL not found: {TTL_PATH}")
        sys.exit(1)

    print(f"Loading KG from {TTL_PATH} …")
    g = Graph()
    g.parse(str(TTL_PATH), format="turtle")
    print(f"  {len(g):,} triples loaded")

    matches = load_matches(g)
    if args.match:
        nf      = args.match.lower()
        matches = [(u, l) for u, l in matches if nf in (l or u).lower()]
    if not matches:
        print("No matches selected.")
        sys.exit(1)
    print(f"  {len(matches)} match(es) selected\n")

    # Trigger Qwen load ONCE up-front so we don't pay it mid-loop.
    print("Loading Qwen model (one-time) …")
    _load_model()

    summary = []
    for match_uri, match_label in matches:
        result = process_match(g, match_uri, match_label)
        summary.append((match_label or match_uri, result))

    print(f"\n{'═'*72}")
    print("  DONE")
    print(f"{'═'*72}")
    for name, s in summary:
        if s.get("status") == "ok":
            print(f"  ✓ {name[:55]:<55} {s.get('events', 0):>4} events")
        else:
            print(f"  ✗ {name[:55]:<55} {s.get('status', '?')}")


if __name__ == "__main__":
    main()
