"""
kg_builder.py — Soccer EKG builder (RDF/OWL + TKG)
─────────────────────────────────────────────────────
Writes a clean A-Box aligned to ekg_tbox.ttl.

Every event / player / team / match instance gets exactly ONE rdf:type
(its leaf class). No foreign-vocab (prov / schema / foaf) multi-typing.

Two entry points:

1. CSV MODE (for testing)
       python kg_builder.py --fast

2. REAL-TIME MODE (for the pipeline)
       from kg_builder import ingest_matched_event
       ingest_matched_event(matched_event, match_name, match_date, ekg, last_event)
"""

import csv
import json
import logging
import re
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from rdflib import Literal, RDF, RDFS, XSD

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

_SCHEMA_FILE = _THIS_DIR / "ekg_schema.py"
if not _SCHEMA_FILE.exists():
    raise FileNotFoundError(
        f"ekg_schema.py missing from {_THIS_DIR}\n"
        f"Fix: git checkout HEAD -- src/4_kg_builder/ekg_schema.py"
    )

from ekg_schema import EKG_Graph, EKG, INST, ACTION_TO_CLASS, typed_literal, add_typed_triple


# ── Paths ──────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "blackburn_forest_2019-10-01.csv"
OUT_DIR  = DATA_DIR / "kg_output"
TTL_PATH    = OUT_DIR  / "ekg.ttl"
STREAM_PATH = OUT_DIR  / "events_stream.jsonl"

_LOG_DIR = DATA_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.WARNING,
    handlers=[
        logging.FileHandler(str(_LOG_DIR / "typing_warnings.log")),
        logging.StreamHandler(),
    ],
    format="%(asctime)s %(levelname)s %(message)s",
)

DEFAULT_MATCH_DATE = "2019-10-01"


def _append_stream(event_data: dict):
    STREAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STREAM_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event_data, default=str) + "\n")


def clear_stream():
    STREAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    STREAM_PATH.write_text("")


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def normalize_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower().strip()).strip("_")


def parse_time_min(time_str: str) -> float:
    t = re.sub(r"'", "", str(time_str).strip())
    if "+" in t:
        base, extra = t.split("+", 1)
        try: return float(base.strip()) + float(extra.strip())
        except: return 0.0
    try: return float(t.strip())
    except: return 0.0


def extract_assist(full_text: str) -> Optional[str]:
    m = re.search(r"Assisted by ([A-Z][a-zA-Z\s\-'\.]+?)(?:\.|with\b)", full_text or "")
    return m.group(1).strip() if m else None


def _extract_date(match_name: str) -> str:
    m = re.search(r"\d{4}-\d{2}-\d{2}", match_name)
    return m.group(0) if m else match_name.split(" - ")[0].strip()


def _extract_teams(match_name: str) -> Tuple[Optional[str], Optional[str]]:
    parts = match_name.split(" - ", 2)
    if len(parts) == 3:
        return parts[1].strip(), parts[2].strip()
    return None, None


# ═══════════════════════════════════════════════════════════════════════════
# A-BOX INSTANCE CREATORS — each writes ONLY ekg:* leaf classes
# ═══════════════════════════════════════════════════════════════════════════

def get_or_create_match(match_name: str, ekg: EKG_Graph) -> Tuple[str, str]:
    """
    Match instance, single-typed as ekg:LeagueMatch.
    Adds ekg:hasHomeTeam / ekg:hasAwayTeam when team names are parseable.
    """
    mid  = normalize_id(match_name)
    date = _extract_date(match_name)

    if mid in ekg._seen_matches:
        return mid, date

    match_uri = ekg.match_uri(mid)
    ekg.g.add((match_uri, RDF.type,    EKG.LeagueMatch))
    ekg.g.add((match_uri, RDFS.label,  Literal(match_name)))
    add_typed_triple(ekg.g, match_uri, EKG.hasDate, date, context=f"match={mid}")

    home, away = _extract_teams(match_name)
    if home:
        ekg.g.add((match_uri, EKG.hasHomeTeam, ekg.team_uri(normalize_id(home))))
    if away:
        ekg.g.add((match_uri, EKG.hasAwayTeam, ekg.team_uri(normalize_id(away))))

    ekg._seen_matches.add(mid)
    return mid, date


def get_or_create_team(team_name: str, ekg: EKG_Graph) -> str:
    """Team instance, single-typed as ekg:Team."""
    tid = normalize_id(team_name)
    if tid in ekg._seen_teams:
        return tid

    team_uri = ekg.team_uri(tid)
    ekg.g.add((team_uri, RDF.type,   EKG.Team))
    ekg.g.add((team_uri, RDFS.label, Literal(team_name)))

    ekg._seen_teams.add(tid)
    return tid


def get_or_create_player(
    player_name : str,
    team_id     : Optional[str],
    ekg         : EKG_Graph,
    match_date  : str = DEFAULT_MATCH_DATE,
) -> Tuple[str, bool]:
    """Player instance, single-typed as ekg:Player. playsFor is a direct triple."""
    pid    = normalize_id(player_name)
    is_new = pid not in ekg._seen_players

    if is_new:
        player_uri = ekg.player_uri(pid)
        ekg.g.add((player_uri, RDF.type,   EKG.Player))
        ekg.g.add((player_uri, RDFS.label, Literal(player_name)))

        if team_id:
            team_uri = ekg.team_uri(team_id)
            ekg.g.add((player_uri, EKG.playsFor, team_uri))

        ekg._seen_players.add(pid)

    return pid, is_new


def add_participates_in(player_id: str, match_id: str, ekg: EKG_Graph):
    player_uri = ekg.player_uri(player_id)
    match_uri  = ekg.match_uri(match_id)
    if (player_uri, EKG.participatedIn, match_uri) not in ekg.g:
        ekg.g.add((player_uri, EKG.participatedIn, match_uri))


def prepopulate_roster(roster_lookup, match_name: str,
                       match_date: str, ekg: EKG_Graph) -> int:
    """Pre-populate all Player + Team nodes from a roster."""
    match_id, _ = get_or_create_match(match_name, ekg)
    total = 0

    for team_name in roster_lookup.get_teams():
        team_id = get_or_create_team(team_name, ekg)
        players = roster_lookup.get_all_players(team_name)

        for p in players:
            player_name = p["name"]
            jersey      = p["jersey"]
            pid         = normalize_id(player_name)

            if pid not in ekg._seen_players:
                player_uri = ekg.player_uri(pid)
                ekg.g.add((player_uri, RDF.type,            EKG.Player))
                ekg.g.add((player_uri, RDFS.label,          Literal(player_name)))
                add_typed_triple(ekg.g, player_uri, EKG.hasJerseyNumber, jersey, context=f"player={pid}")

                team_uri = ekg.team_uri(team_id)
                ekg.g.add((player_uri, EKG.playsFor, team_uri))
                # team→player back-link via the new T-Box's ekg:hasPlayer
                ekg.g.add((team_uri, EKG.hasPlayer, player_uri))

                ekg._seen_players.add(pid)
                total += 1

        print(f"  [kg] pre-populated {len(players)} players for {team_name}")

    print(f"  [kg] roster pre-population done: {total} players, {ekg.triple_count()} triples")
    return total


# ═══════════════════════════════════════════════════════════════════════════
# CORE: CREATE ONE EVENT NODE — single leaf rdf:type, new property names
# ═══════════════════════════════════════════════════════════════════════════

def _create_event_node(
    ekg         : EKG_Graph,
    match_id    : str,
    time_raw    : str,
    event_type  : str,
    player_id   : Optional[str],
    team_id     : Optional[str],
    full_text   : str = "",
    confidence  : Optional[float] = None,
    matched     : bool = True,
    last_event  : dict = None,
    description  : Optional[str] = None,
    jersey       : Optional[str] = None,
    team_color   : Optional[str] = None,
    shorts_color : Optional[str] = None,
    socks_color  : Optional[str] = None,
    kit_pattern  : Optional[str] = None,
    pitch_zone   : Optional[str] = None,
    body_part    : Optional[str] = None,
    outcome      : Optional[str] = None,
    foul_type    : Optional[str] = None,
    team_side    : Optional[str] = None,
    ball_visible : Optional[bool] = None,
) -> Tuple[str, list]:
    """
    Create one Event node with EXACTLY ONE rdf:type (the leaf class from
    ACTION_TO_CLASS, e.g. ekg:Shot). The T-Box's subClassOf chain handles
    parent classes via reasoning — kg_builder never asserts them.
    """
    ekg._event_count += 1
    event_id  = f"{ekg._event_count:04d}"
    event_uri = ekg.event_uri(event_id, match_id)

    # Exactly one rdf:type, the leaf class.
    leaf_class = ACTION_TO_CLASS.get(event_type, EKG.Event)
    ekg.g.add((event_uri, RDF.type, leaf_class))

    # Data properties carrying VLM / ESPN info.
    ekg.g.add((event_uri, EKG.hasEventType, Literal(event_type)))
    ekg.g.add((event_uri, EKG.hasTime,      Literal(time_raw)))
    add_typed_triple(ekg.g, event_uri, EKG.isMatched, matched, context=f"event={event_id}")

    # hasMinute (decimal within half) + hasPeriodNumber (1 / 2).
    # The new T-Box reserves ekg:hasPeriod as an ObjectProperty
    # (Match → Period); the integer half number is stored as
    # ekg:hasPeriodNumber on the event.
    try:
        half, t  = time_raw.strip().split(" ", 1)
        mm, ss   = t.strip().split(":")
        period   = 1 if half == "1st" else 2
        minute   = int(mm) + int(ss) / 60.0
        add_typed_triple(ekg.g, event_uri, EKG.hasMinute,       round(minute, 3), context=f"event={event_id}")
        add_typed_triple(ekg.g, event_uri, EKG.hasPeriodNumber, period,           context=f"event={event_id}")
    except Exception:
        print(f"  [kg] WARNING: cannot parse hasMinute/hasPeriodNumber from '{time_raw}'")

    if full_text:
        ekg.g.add((event_uri, EKG.hasFullText,    Literal(full_text)))
    if description:
        ekg.g.add((event_uri, EKG.hasDescription, Literal(description)))
    if jersey:
        add_typed_triple(ekg.g, event_uri, EKG.detectedJersey, jersey, context=f"event={event_id}")
    if team_color:
        ekg.g.add((event_uri, EKG.hasDetectedColor,       Literal(str(team_color))))
    if shorts_color:
        ekg.g.add((event_uri, EKG.hasDetectedShortsColor, Literal(str(shorts_color))))
    if socks_color:
        ekg.g.add((event_uri, EKG.hasDetectedSocksColor,  Literal(str(socks_color))))
    if kit_pattern:
        ekg.g.add((event_uri, EKG.hasDetectedKitPattern, Literal(str(kit_pattern))))
    if pitch_zone:
        ekg.g.add((event_uri, EKG.hasPitchZone,   Literal(str(pitch_zone).lower().strip())))
    if body_part:
        ekg.g.add((event_uri, EKG.hasBodyPart,    Literal(str(body_part))))
    if outcome:
        ekg.g.add((event_uri, EKG.hasOutcome,     Literal(str(outcome))))
    if foul_type:
        ekg.g.add((event_uri, EKG.hasFoulType,    Literal(str(foul_type))))
    if team_side:
        ekg.g.add((event_uri, EKG.hasTeamSide,    Literal(str(team_side))))
    if ball_visible is not None:
        add_typed_triple(ekg.g, event_uri, EKG.hasBallVisible, ball_visible, context=f"event={event_id}")

    new_edges = []

    # inMatch (event → match)
    match_uri = ekg.match_uri(match_id)
    ekg.g.add((event_uri, EKG.inMatch, match_uri))
    new_edges.append(f"event_{event_id} --[inMatch]--> {match_id}")

    # precededBy + inverse precedes
    if last_event is not None and match_id in last_event:
        prev_uri = ekg.event_uri(last_event[match_id], match_id)
        ekg.g.add((event_uri, EKG.precededBy, prev_uri))
        ekg.g.add((prev_uri,  EKG.precedes,   event_uri))
        new_edges.append(f"event_{event_id} --[precededBy]--> event_{last_event[match_id]}")
    if last_event is not None:
        last_event[match_id] = event_id

    # involvedTeam (event → team)
    if team_id:
        team_uri = ekg.team_uri(team_id)
        ekg.g.add((event_uri, EKG.involvedTeam, team_uri))
        new_edges.append(f"event_{event_id} --[involvedTeam]--> {team_id}")

    # performed + inverse isPerformedBy
    if player_id:
        player_uri = ekg.player_uri(player_id)
        ekg.g.add((player_uri, EKG.performed,      event_uri))
        ekg.g.add((event_uri,  EKG.isPerformedBy, player_uri))
        new_edges.append(f"{player_id} --[performed]--> event_{event_id}")

    _append_stream({
        "event_id"   : event_id,
        "event_type" : event_type,
        "time_raw"   : time_raw,
        "player_id"  : player_id,
        "team_id"    : team_id,
        "match_id"   : match_id,
        "description": description,
        "pitch_zone" : pitch_zone,
        "body_part"  : body_part,
        "outcome"    : outcome,
        "foul_type"  : foul_type,
        "team_side"  : team_side,
        "confidence" : confidence,
        "timestamp"  : datetime.utcnow().isoformat(),
    })

    return event_id, new_edges


# ═══════════════════════════════════════════════════════════════════════════
# INGEST — CSV MODE
# ═══════════════════════════════════════════════════════════════════════════

def ingest_event(row: dict, ekg: EKG_Graph, last_event: dict) -> dict:
    match_name  = row["Match"]
    team_name   = row["Team"]   if row["Team"]   != "None" else None
    time_raw    = row["Time"]
    player_name = row["Player"] if row["Player"] != "None" else None
    action_type = row["Action_Type"]
    full_text   = row["Full_Text"]
    yellow      = row["Yellow_Card"] == "1"
    red         = row["Red_Card"]    == "1"

    new_edges     = []
    is_new_player = False

    match_id, match_date = get_or_create_match(match_name, ekg)
    team_id  = get_or_create_team(team_name, ekg) if team_name else None

    player_id = None
    if player_name:
        player_id, is_new_player = get_or_create_player(
            player_name, team_id, ekg, match_date)
        if is_new_player and team_id:
            new_edges.append(f"{player_name} --[playsFor (TKG:{match_date})]--> {team_name}")
        add_participates_in(player_id, match_id, ekg)

    main_event_id, edges = _create_event_node(
        ekg, match_id, time_raw,
        event_type = action_type,
        player_id  = player_id,
        team_id    = team_id,
        full_text  = full_text,
        matched    = True,
        last_event = last_event,
    )
    new_edges.extend(edges)

    assist_name = extract_assist(full_text)
    if assist_name and assist_name != player_name:
        assist_pid, assist_new = get_or_create_player(
            assist_name, team_id, ekg, match_date)
        ekg.g.add((ekg.event_uri(main_event_id), EKG.assistedBy, ekg.player_uri(assist_pid)))
        new_edges.append(f"event_{main_event_id} --[assistedBy]--> {assist_name}")
        if assist_new:
            is_new_player = True

    card_type = "RedCard" if red else ("YellowCard" if yellow else None)
    if card_type:
        card_event_id, card_edges = _create_event_node(
            ekg, match_id, time_raw,
            event_type = card_type,
            player_id  = player_id,
            team_id    = team_id,
            full_text  = f"{card_type} following {action_type}",
            matched    = True,
            last_event = last_event,
        )
        new_edges.extend(card_edges)
        ekg.g.add((ekg.event_uri(main_event_id), EKG.triggered, ekg.event_uri(card_event_id)))
        new_edges.append(f"event_{main_event_id} --[triggered]--> event_{card_event_id} ({card_type})")

    return {
        "time": time_raw, "player": player_name,
        "action_type": action_type, "team": team_name,
        "is_new_player": is_new_player,
        "yellow": yellow, "red": red, "card": card_type,
        "new_edges": new_edges,
    }


# ═══════════════════════════════════════════════════════════════════════════
# INGEST — REAL-TIME MODE
# ═══════════════════════════════════════════════════════════════════════════

def ingest_matched_event(
    matched,
    match_name   : str,
    match_date   : str,
    ekg          : EKG_Graph,
    last_event   : dict,
    description  : Optional[str] = None,
    jersey       : Optional[str] = None,
    team_color   : Optional[str] = None,
    shorts_color : Optional[str] = None,
    socks_color  : Optional[str] = None,
    kit_pattern  : Optional[str] = None,
    pitch_zone   : Optional[str] = None,
    body_part    : Optional[str] = None,
    outcome      : Optional[str] = None,
    foul_type    : Optional[str] = None,
    team_side    : Optional[str] = None,
    ball_visible : Optional[bool] = None,
) -> dict:
    action_type = matched.action
    time_raw    = matched.gametime or f"{matched.video_time/60:.1f}'"
    full_text   = matched.espn_text or ""
    confidence  = matched.confidence

    new_edges     = []
    is_new_player = False

    match_id, _ = get_or_create_match(match_name, ekg)

    if matched.matched and matched.player:
        team_id = get_or_create_team(matched.team, ekg) if matched.team else None
        player_id, is_new_player = get_or_create_player(
            matched.player, team_id, ekg, match_date)
        if is_new_player and team_id:
            new_edges.append(f"{matched.player} --[playsFor (TKG:{match_date})]--> {matched.team}")
        add_participates_in(player_id, match_id, ekg)
    else:
        player_id = None
        team_id   = get_or_create_team(matched.team, ekg) if matched.team else None

    if team_id is None:
        print(f"  [kg] WARNING: team_id=None for {matched.action} at {matched.gametime} "
              f"(matched.team={matched.team!r}) — involvedTeam will be skipped")

    event_id, edges = _create_event_node(
        ekg, match_id, time_raw,
        event_type   = action_type,
        player_id    = player_id,
        team_id      = team_id,
        full_text    = full_text,
        confidence   = confidence,
        matched      = matched.matched,
        last_event   = last_event,
        description  = description,
        jersey       = jersey,
        team_color   = team_color,
        shorts_color = shorts_color,
        socks_color  = socks_color,
        kit_pattern  = kit_pattern,
        pitch_zone   = pitch_zone,
        body_part    = body_part,
        outcome      = outcome,
        foul_type    = foul_type,
        team_side    = team_side,
        ball_visible = ball_visible,
    )
    new_edges.extend(edges)

    card_type = None
    if matched.matched and full_text:
        if re.search(r"\bred card\b", full_text, re.IGNORECASE):
            card_type = "RedCard"
        elif re.search(r"\byellow card\b|booked", full_text, re.IGNORECASE):
            card_type = "YellowCard"

    if card_type:
        card_event_id, card_edges = _create_event_node(
            ekg, match_id, time_raw,
            event_type = card_type,
            player_id  = player_id,
            team_id    = team_id,
            matched    = True,
            last_event = last_event,
        )
        new_edges.extend(card_edges)
        ekg.g.add((ekg.event_uri(event_id), EKG.triggered, ekg.event_uri(card_event_id)))
        new_edges.append(f"event_{event_id} --[triggered]--> event_{card_event_id} ({card_type})")

    return {
        "time": time_raw,
        "player": matched.player if matched.matched else "UNKNOWN",
        "action_type": action_type, "team": matched.team,
        "is_new_player": is_new_player,
        "matched": matched.matched, "card": card_type,
        "new_edges": new_edges,
    }


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

ACTION_ICONS = {
    "Goal": "GOAL", "Shot": "shot", "Foul": "foul",
    "Corner": "corner", "Offside": "offside",
    "FreeKick": "free kick", "Substitution": "sub",
    "YellowCard": "YC", "RedCard": "RC", "Penalty": "penalty",
}

def print_event(res: dict, ekg: EKG_Graph):
    icon   = ACTION_ICONS.get(res["action_type"], res["action_type"])
    player = res.get("player") or "(no player)"
    team   = res.get("team")   or "—"
    tag    = " << NEW PLAYER >>" if res.get("is_new_player") else ""
    card   = f" [{res.get('card')}]" if res.get("card") else ""
    print(f"\n[{res['time']:>6}]  {icon:<12}  {player}{card}{tag}")
    print(f"          Team: {team}")
    for edge in res["new_edges"]:
        print(f"          + {edge}")
    print(f"          KG -> {ekg.stats()}")


# ═══════════════════════════════════════════════════════════════════════════
# CSV MODE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def stream_simulation(csv_path: Path, scale: float = 1.0, fast: bool = False):
    with open(csv_path, encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f),
                      key=lambda r: parse_time_min(r.get("Time", "0")))

    ekg        = EKG_Graph()
    last_event = {}
    prev_min   = None

    match_name = rows[0]["Match"] if rows else "Unknown"
    mode_label = "FAST" if fast else f"scale={scale}s/min"

    print(f"\n{'='*62}")
    print(f"  MATCH : {match_name}")
    print(f"  EVENTS: {len(rows)}  |  MODE: {mode_label}")
    print(f"  T-Box : {len(ekg.g)} triples at start")
    print(f"{'='*62}")

    for i, row in enumerate(rows):
        curr_min = parse_time_min(row.get("Time", "0"))
        if not fast and prev_min is not None:
            wait = max(0.1, min(5.0, (curr_min - prev_min) * scale))
            time.sleep(wait)
        prev_min = curr_min
        result   = ingest_event(row, ekg, last_event)
        print_event(result, ekg)
        if (i + 1) % 20 == 0:
            ekg.save(TTL_PATH)

    ekg.save(TTL_PATH)
    yellow = ekg.events_by_type("YellowCard")
    red    = ekg.events_by_type("RedCard")

    print(f"\n{'='*62}")
    print(f"  MATCH ENDED")
    print(f"  Final EKG    : {ekg.stats()}")
    print(f"  Yellow cards : {len(yellow)}")
    print(f"  Red cards    : {len(red)}")
    print(f"  Saved to     : {TTL_PATH}")
    print(f"{'='*62}\n")
    return ekg


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--fast",  action="store_true")
    args = parser.parse_args()
    stream_simulation(CSV_PATH, scale=args.scale, fast=args.fast)
