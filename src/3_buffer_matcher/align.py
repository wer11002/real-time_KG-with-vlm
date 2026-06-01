"""
align.py
--------
Matches video-detected events (from buffer.py) to ESPN scraped events.

Matching strategy (two-tier):

  TIER 1 — Jersey-first (primary, used when Qwen2-VL detects a jersey):
      jersey number → roster_lookup → player name + team
      → directly identifies the player without time search
      → still confirms with ESPN event nearby for action type

  TIER 2 — Fuzzy time match (fallback, used when no jersey detected):
      search ESPN events within ±2 min with compatible action type
      → picks closest time match

Output per buffered event:
    matched      : True/False
    player       : name (if matched)
    team         : team (if matched)
    espn_time    : ESPN-reported minute (if matched)
    match_method : "jersey" or "time" or "unmatched"

Quick test:
    python align.py
"""

import sys
import dataclasses
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict


# ═══════════════════════════════════════════════════════════════════════════
# FUZZY ACTION MAPPING
# ═══════════════════════════════════════════════════════════════════════════

ACTION_MAP: Dict[str, List[str]] = {
    "Shot"        : ["Shot", "Goal"],
    "Goal"        : ["Goal", "Shot"],
    "Penalty"     : ["Goal", "Shot"],
    "Foul"        : ["Foul", "Free_Kick"],
    "Free_Kick"   : ["Free_Kick", "Foul"],
    "Corner"      : ["Corner"],
    "Offside"     : ["Offside"],
    "Substitution": ["Substitution"],
}


def action_matches(video_action: str, espn_action: str) -> bool:
    va = video_action.strip()
    ea = espn_action.strip()
    return ea in ACTION_MAP.get(va, [va])


# ═══════════════════════════════════════════════════════════════════════════
# MATCHED EVENT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MatchedEvent:
    """Result of aligning one video event against ESPN data."""

    # video-side fields
    video_time    : float
    action        : str
    confidence    : float
    gametime      : str

    # alignment result
    matched       : bool
    player        : Optional[str]   = None
    team          : Optional[str]   = None
    espn_time     : Optional[float] = None
    espn_text     : Optional[str]   = None
    time_diff     : Optional[float] = None
    match_method  : str             = "unmatched"  # "jersey", "time", "unmatched"

    # VLM fields (passed through from action_recognizer)
    jersey        : Optional[str]   = None
    description   : Optional[str]   = None
    team_color    : Optional[str]   = None   # jersey color e.g. "blue/white"
    shorts_color  : Optional[str]   = None
    socks_color   : Optional[str]   = None
    kit_pattern   : Optional[str]   = None
    pitch_zone    : Optional[str]   = None
    body_part     : Optional[str]   = None
    outcome       : Optional[str]   = None   # Shot/Goal result e.g. "saved_low"
    foul_type     : Optional[str]   = None   # Foul sub-type e.g. "tackle"
    team_side     : Optional[str]   = None   # "home" or "away"
    ball_visible         : Optional[bool]  = None   # quality flag
    vlm_confidence_score : Optional[float] = None

    def to_dict(self):
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# TIER 1 — JERSEY-FIRST MATCHING
# ═══════════════════════════════════════════════════════════════════════════

def match_by_jersey(
    video_event        : dict,
    espn_events        : List[dict],
    roster_lookup,
    time_tolerance_min : float = 3.0,
) -> Optional[MatchedEvent]:
    """
    Try to match by jersey number first.
    
    Steps:
      1. Get jersey + team_color from video_event (set by Qwen2-VL)
      2. Look up player name via roster_lookup
      3. Find ESPN event near the same time with compatible action
      4. Return MatchedEvent with match_method="jersey"

    Returns None if jersey not available or lookup fails.
    """
    jersey      = video_event.get("jersey")
    team_color  = video_event.get("team_color")
    description = video_event.get("description")

    if not jersey:
        return None   # no jersey detected — fall through to time matching

    video_minute = video_event["video_time"] / 60.0
    video_action = video_event["action"]

    # try direct team lookup first
    team_hint = video_event.get("team")
    player    = None
    team      = None

    if team_hint and roster_lookup:
        player = roster_lookup.find(jersey=jersey, team=team_hint)
        if player:
            team = team_hint

    # fallback: try color-based lookup
    if not player and team_color and roster_lookup:
        candidates = roster_lookup.find_by_color(jersey=jersey, color=team_color)
        if len(candidates) == 1:
            # only one team has this jersey number — unambiguous
            player = candidates[0]["player"]
            team   = candidates[0]["team"]
        elif len(candidates) > 1:
            # ambiguous — use ESPN event nearby to resolve team
            for e in espn_events:
                if abs(e["time"] - video_minute) <= time_tolerance_min:
                    for c in candidates:
                        if c["team"] == e.get("team"):
                            player = c["player"]
                            team   = c["team"]
                            break
                if player:
                    break

    if not player:
        return None   # jersey lookup failed — fall through

    # find ESPN event nearby to confirm action + get espn_text
    espn_match = None
    best_diff  = float("inf")

    for e in espn_events:
        diff = abs(e["time"] - video_minute)
        if diff <= time_tolerance_min and action_matches(video_action, e["action"]):
            if diff < best_diff:
                best_diff  = diff
                espn_match = e

    # Upgrade Shot→Goal if ESPN confirms Goal; never downgrade
    final_action = video_action
    if espn_match and video_action == "Shot" and espn_match.get("action") == "Goal":
        final_action = "Goal"
        print(f"  [align] Shot→Goal upgrade at {video_event['gametime']} "
              f"(ESPN confirms Goal)")

    return MatchedEvent(
        video_time   = video_event["video_time"],
        action       = final_action,
        confidence   = video_event["confidence"],
        gametime     = video_event["gametime"],
        matched      = True,
        player       = player,
        team         = team,
        espn_time    = espn_match["time"]      if espn_match else None,
        espn_text    = espn_match["full_text"] if espn_match else None,
        time_diff    = round(best_diff, 2)     if espn_match else None,
        match_method = "jersey",
        jersey       = jersey,
        description  = description,
        team_color   = team_color,
        shorts_color = video_event.get("shorts_color"),
        socks_color  = video_event.get("socks_color"),
        kit_pattern  = video_event.get("kit_pattern"),
        pitch_zone   = video_event.get("pitch_zone"),
        body_part    = video_event.get("body_part"),
        outcome      = video_event.get("outcome"),
        foul_type    = video_event.get("foul_type"),
        team_side    = video_event.get("team_side"),
        ball_visible = video_event.get("ball_visible"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 2 — FUZZY TIME MATCHING (fallback)
# ═══════════════════════════════════════════════════════════════════════════

def match_by_time(
    video_event        : dict,
    espn_events        : List[dict],
    time_tolerance_min : float = 2.0,
) -> MatchedEvent:
    """
    Match by time + action type (enrichment only — no ESPN gate).
    ESPN play-by-play used only in evaluate.py for post-hoc scoring.
    All events from the buffer pass through; no confidence gate.
    """
    video_minute = video_event["video_time"] / 60.0
    video_action = video_event["action"]
    description  = video_event.get("description")
    jersey       = video_event.get("jersey")

    candidates = []
    for e in espn_events:
        diff = abs(e["time"] - video_minute)
        if diff <= time_tolerance_min and action_matches(video_action, e["action"]):
            candidates.append((diff, e))

    kit = {
        "shorts_color": video_event.get("shorts_color"),
        "socks_color" : video_event.get("socks_color"),
        "kit_pattern" : video_event.get("kit_pattern"),
        "pitch_zone"  : video_event.get("pitch_zone"),
        "body_part"   : video_event.get("body_part"),
        "outcome"     : video_event.get("outcome"),
        "foul_type"   : video_event.get("foul_type"),
        "team_side"   : video_event.get("team_side"),
        "ball_visible": video_event.get("ball_visible"),
    }

    if not candidates:
        return MatchedEvent(
            video_time   = video_event["video_time"],
            action       = video_action,
            confidence   = video_event["confidence"],
            gametime     = video_event["gametime"],
            matched      = False,
            team         = video_event.get("team"),
            match_method = "unmatched",
            jersey       = jersey,
            description  = description,
            team_color   = video_event.get("team_color"),
            **kit,
        )

    candidates.sort(key=lambda x: x[0])
    best_diff, best = candidates[0]

    # Upgrade Shot→Goal if ESPN confirms Goal; never downgrade
    final_action = video_action
    if video_action == "Shot" and best.get("action") == "Goal":
        final_action = "Goal"
        print(f"  [align] Shot→Goal upgrade at {video_event['gametime']} "
              f"(ESPN confirms Goal)")

    return MatchedEvent(
        video_time   = video_event["video_time"],
        action       = final_action,
        confidence   = video_event["confidence"],
        gametime     = video_event["gametime"],
        matched      = True,
        player       = best.get("player"),
        team         = best.get("team"),
        espn_time    = best.get("time"),
        espn_text    = best.get("full_text"),
        time_diff    = round(best_diff, 2),
        match_method = "time",
        jersey       = jersey,
        description  = description,
        team_color   = video_event.get("team_color"),
        **kit,
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN MATCH FUNCTION — two-tier
# ═══════════════════════════════════════════════════════════════════════════

def match_event(
    video_event        : dict,
    espn_events        : List[dict],
    time_tolerance_min : float = 2.0,
    roster_lookup      = None,
) -> MatchedEvent:
    """
    Two-tier matching:
      1. Try jersey-first (if jersey in video_event and roster_lookup provided)
      2. Fall back to fuzzy time matching
    """
    # tier 1: jersey
    if roster_lookup and video_event.get("jersey"):
        result = match_by_jersey(
            video_event, espn_events, roster_lookup,
            time_tolerance_min = time_tolerance_min + 1.0  # slightly wider for jersey
        )
        if result:
            return result

    # tier 2: time fallback
    return match_by_time(video_event, espn_events, time_tolerance_min)


def align_buffer(
    buffer_events      : List,
    espn_events        : List[dict],
    time_tolerance_min : float = 2.0,
    roster_lookup      = None,
    espn_scraper       = None,   # if provided, consume matched events to prevent duplicates
) -> List[MatchedEvent]:
    """
    Align all buffered video events against ESPN data.
    Supports both VideoEvent dataclass objects and plain dicts.

    If espn_scraper is provided, successfully matched ESPN events are
    marked as consumed — preventing duplicate KG nodes when two
    overlapping clips both detect the same real event.
    """
    results = []
    for v in buffer_events:
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            v_dict = {
                "video_time"  : v.video_time,
                "action"      : v.action,
                "confidence"  : v.confidence,
                "gametime"    : v.gametime,
                "jersey"      : getattr(v, "jersey",       None),
                "team"        : getattr(v, "team",         None),
                "team_color"  : getattr(v, "team_color",   None),
                "shorts_color": getattr(v, "shorts_color", None),
                "socks_color" : getattr(v, "socks_color",  None),
                "kit_pattern" : getattr(v, "kit_pattern",  None),
                "pitch_zone"  : getattr(v, "pitch_zone",   None),
                "body_part"   : getattr(v, "body_part",    None),
                "outcome"     : getattr(v, "outcome",      None),
                "foul_type"   : getattr(v, "foul_type",    None),
                "team_side"   : getattr(v, "team_side",    None),
                "ball_visible": getattr(v, "ball_visible", None),
                "description" : getattr(v, "description",  None),
            }
        else:
            v_dict = v

        matched = match_event(v_dict, espn_events, time_tolerance_min, roster_lookup)

        # consume the matched ESPN event to prevent duplicate KG nodes
        if matched.matched and matched.espn_time is not None and espn_scraper:
            espn_scraper.consume_event(
                time   = matched.espn_time,
                action = matched.action,
            )

        results.append(matched)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def summarize(matched_events: List[MatchedEvent]) -> str:
    if not matched_events:
        return "no events aligned"

    matched_count = sum(1 for e in matched_events if e.matched)
    jersey_count  = sum(1 for e in matched_events if e.match_method == "jersey")
    time_count    = sum(1 for e in matched_events if e.match_method == "time")

    lines = [f"\n── Alignment results: {matched_count}/{len(matched_events)} matched "
             f"(jersey={jersey_count} time={time_count}) ──"]

    for e in matched_events:
        if e.matched:
            method = f"[{e.match_method}]"
            jersey = f" #{e.jersey}" if e.jersey else ""
            lines.append(
                f"  ✓ {e.gametime:<12} {e.action:<10} → "
                f"{e.player or '—':<22} ({e.team})  "
                f"Δt={e.time_diff}min {method}{jersey}"
            )
        else:
            lines.append(
                f"  ✗ {e.gametime:<12} {e.action:<10} → UNKNOWN"
            )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    sys.path.insert(0, str(Path(__file__).parent.parent / "2_web_scraper"))
    sys.path.insert(0, str(Path(__file__).parent.parent / "1_video_processor"))

    from espn_scraper  import ESPNScraper
    from roster_lookup import RosterLookup

    print("─── align.py self-test ───\n")

    # load ESPN data
    scraper = ESPNScraper()
    scraper.find_and_load("2019-10-01", "Blackburn Rovers", "Nottingham Forest")
    espn_events = scraper.get_all_events()
    print(f"ESPN: {len(espn_events)} events\n")

    # load roster
    lookup = RosterLookup()
    lookup.load_from_espn("2019-10-01", "Blackburn Rovers", "Nottingham Forest")

    # ── test 1: jersey-first matching ──────────────────────────────────────
    print("── Test 1: jersey-first matching ──")
    video_with_jersey = [
        {   # Qwen2-VL detected jersey #7 + Shot → should match Adam Armstrong
            "video_time" : 554.2,
            "action"     : "Shot",
            "confidence" : 0.85,
            "gametime"   : "1st 09:14",
            "jersey"     : "7",
            "team"       : "Blackburn Rovers",
            "team_color" : "blue/white",
            "description": "Player #7 takes a right-footed shot from outside the box",
        },
        {   # Qwen2-VL detected jersey #23 + Foul → should match Joe Lolley
            "video_time" : 1560.0,
            "action"     : "Foul",
            "confidence" : 0.72,
            "gametime"   : "1st 26:00",
            "jersey"     : "23",
            "team"       : "Nottingham Forest",
            "team_color" : "red",
            "description": "Player #23 commits a foul in midfield",
        },
    ]

    results_jersey = align_buffer(
        video_with_jersey, espn_events,
        time_tolerance_min=2.0,
        roster_lookup=lookup
    )
    print(summarize(results_jersey))

    # ── test 2: fallback time matching (no jersey) ─────────────────────────
    print("\n── Test 2: fallback time matching (no jersey) ──")
    video_no_jersey = [
        {
            "video_time" : 554.2,
            "action"     : "Shot",
            "confidence" : 0.37,
            "gametime"   : "1st 09:14",
        },
        {
            "video_time" : 1800.0,
            "action"     : "Shot",
            "confidence" : 0.65,
            "gametime"   : "1st 30:00",
        },
    ]

    results_time = align_buffer(
        video_no_jersey, espn_events,
        time_tolerance_min=2.0,
        roster_lookup=lookup
    )
    print(summarize(results_time))

    # ── action map checks ──────────────────────────────────────────────────
    print("\n── Action map checks ──")
    print(f"  'Goal' ↔ 'Shot'      : {action_matches('Goal',   'Shot')}   (True)")
    print(f"  'Shot' ↔ 'Goal'      : {action_matches('Shot',   'Goal')}   (True)")
    print(f"  'Corner' ↔ 'Foul'   : {action_matches('Corner', 'Foul')}  (False)")
    print(f"  'Foul' ↔ 'Free_Kick' : {action_matches('Foul', 'Free_Kick')}  (True)")

    print("\n✓ all good!")