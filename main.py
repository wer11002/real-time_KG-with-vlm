"""
main.py — Soccer EKG Real-Time Pipeline (V2)
─────────────────────────────────────────────

What's new in this version:
  1. Log noise fixed — HTTP debug lines silenced
  2. KG pre-populated with ALL players at startup
     (player nodes exist before first clip runs)
  3. Multi-folder support — processes all match folders
     in data/ directory in chronological order,
     growing ONE KG across all matches

Run:
    python main.py                    # all matches in data/ folder
    python main.py --test             # 5 clips, 224p, first match only
    python main.py --clips 20         # first 20 clips per match
    python main.py --match "2019-10-01 - Blackburn Rovers - Nottingham Forest"
                                      # specific match only
    python main.py --espn-every 3     # ESPN tick every 3 clips
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# ── make all modules importable ────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src" / "1_video_processor"))
sys.path.insert(0, str(BASE_DIR / "src" / "2_web_scraper"))
sys.path.insert(0, str(BASE_DIR / "src" / "3_buffer_matcher"))
sys.path.insert(0, str(BASE_DIR / "src" / "4_kg_builder"))
sys.path.insert(0, str(BASE_DIR / "src" / "commentator"))

from sliding_window    import (
    extract_clip, get_video_duration, seconds_to_gametime,
    delete_clip, TEMP_DIR, get_halftime_sec,
)
from action_recognizer import detect_actions, load_model, load_colors_from_espn, load_colors_from_game_id
from buffer            import EventBuffer
from espn_scraper      import ESPNScraper
from roster_lookup     import RosterLookup
from align             import align_buffer, summarize
from kg_builder        import (
    EKG_Graph, ingest_matched_event, prepopulate_roster, TTL_PATH, clear_stream
)
from commentator       import event_queue, start_commentator, log_match_boundary

DATA_DIR      = BASE_DIR / "data"
CLIP_DURATION = 60
CLIP_STEP     = 30


# ═══════════════════════════════════════════════════════════════════════════
# SCORE-STATE VALIDATION (deferred)
# ═══════════════════════════════════════════════════════════════════════════

class ScoreValidator:
    """
    Deferred Goal validator backed by the ESPN scraper score.
    Goals are queued; each clip we query ESPN to see whether the score
    has incremented.  If yes → confirm.  If no change within
    REJECT_AFTER_MIN minutes → reject (downgrade to Shot).
    """
    REJECT_AFTER_MIN = 2.0   # minutes after queuing before rejecting unconfirmed goals

    def __init__(self, scraper, team1: str, team2: str):
        self._scraper    = scraper
        self._team1      = team1   # home team name
        self._team2      = team2   # away team name
        self._last_score = {"home": 0, "away": 0}
        self._pending    = []      # [{det, start_sec, gametime, queued_minute, waited}]

    # ── helpers ──────────────────────────────────────────────────────────────

    def _get_espn_score(self, minute: float) -> dict:
        """
        Count ESPN goals up to `minute` and return {"home": int, "away": int}.
        Uses skip_consumed=False so already-consumed events are still counted.
        """
        if self._scraper is None:
            return dict(self._last_score)

        home, away = 0, 0
        for ev in self._scraper.get_events_up_to(minute, skip_consumed=False):
            if ev.get("action") not in ("Goal",):
                continue
            team = (ev.get("team") or "").strip()
            if not team:
                continue
            t1, t2 = self._team1.lower(), self._team2.lower()
            tl     = team.lower()
            if t1 in tl or tl in t1:
                home += 1
            elif t2 in tl or tl in t2:
                away += 1
        return {"home": home, "away": away}

    # ── public API ────────────────────────────────────────────────────────────

    def queue(self, det: dict, start_sec: float, gametime: str,
              current_minute: float) -> None:
        """Add a Goal detection to the pending queue."""
        self._pending.append({
            "det"           : det,
            "start_sec"     : start_sec,
            "gametime"      : gametime,
            "queued_minute" : current_minute,
            "waited"        : 0,
        })

    def tick(self, current_minute: float) -> list[tuple[dict, float, str, bool]]:
        """
        Call once per clip with the current match minute (start_sec / 60.0).
        Returns list of (det, orig_start_sec, orig_gametime, is_goal).
        """
        results = []

        # no pending goals → nothing to do
        if not self._pending and self._scraper is None:
            return results

        # 1. Age all pending entries
        for entry in self._pending:
            entry["waited"] += 1

        # 2. Query ESPN score at current minute
        espn_score = self._get_espn_score(current_minute)
        home_diff  = espn_score["home"] - self._last_score["home"]
        away_diff  = espn_score["away"] - self._last_score["away"]
        total_diff = home_diff + away_diff

        # 3. Score sanity: ignore regressions or jumps > 1
        if home_diff >= 0 and away_diff >= 0 and total_diff == 1:
            # ESPN confirms a goal — confirm oldest pending regardless of team_side
            # (VLM team_side is unreliable for Goals; trust ESPN score increment)
            match_idx = None
            if self._pending:
                match_idx = 0

            if match_idx is not None:
                entry = self._pending.pop(match_idx)
                self._last_score = dict(espn_score)
                print(f"  [score-defer] CONFIRMED Goal at {entry['gametime']} "
                      f"(ESPN score now {espn_score['home']}-{espn_score['away']})")
                results.append(
                    (entry["det"], entry["start_sec"], entry["gametime"], True)
                )

        # 4. Reject entries that have waited too long without a score change
        still_pending = []
        for entry in self._pending:
            age = current_minute - entry["queued_minute"]
            if age >= self.REJECT_AFTER_MIN:
                print(f"  [score-defer] REJECTED Goal at {entry['gametime']} "
                      f"(no ESPN score change after {age:.1f} min)")
                results.append(
                    (entry["det"], entry["start_sec"], entry["gametime"], False)
                )
            else:
                still_pending.append(entry)
        self._pending = still_pending

        return results

    def flush_all(self) -> list[tuple[dict, float, str, bool]]:
        """End of match: reject all remaining pending goals as Shot."""
        results = []
        for p in self._pending:
            print(f"  [score-defer] END-OF-MATCH → Shot at {p['gametime']}")
            results.append((p["det"], p["start_sec"], p["gametime"], False))
        self._pending = []
        return results

REGISTRY_PATH = BASE_DIR / "data" / "kg_output" / "processed_matches.json"


# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = log_dir / f"pipeline_{timestamp}.log"

    # silence noisy HTTP/network/model debug loggers
    for noisy in [
        "httpx", "httpcore", "urllib3", "requests",
        "urllib3.connectionpool", "huggingface_hub",
        "transformers", "torch", "filelock",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    import builtins
    def _logged_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        logger.info(msg)
    builtins.print = _logged_print

    return log_path


# ═══════════════════════════════════════════════════════════════════════════
# CHECKPOINT REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

def load_registry() -> set:
    if REGISTRY_PATH.exists():
        import json
        return set(json.loads(REGISTRY_PATH.read_text()))
    return set()


def save_registry(registry: set):
    import json
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(sorted(registry), indent=2))


# ═══════════════════════════════════════════════════════════════════════════
# MATCH DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

def parse_match_folder(folder: Path):
    """
    Parse a match folder name into (date, team1, team2).
    Expected format: "YYYY-MM-DD - Team One - Team Two"
    Returns None if folder doesn't match pattern.
    """
    name = folder.name
    parts = name.split(" - ", 2)
    if len(parts) != 3:
        return None
    date, team1, team2 = parts
    # validate date format
    if not (len(date) == 10 and date[4] == "-" and date[7] == "-"):
        return None
    return date.strip(), team1.strip(), team2.strip()


def find_match_folders(data_dir: Path) -> List[Path]:
    """
    Find all valid match folders under data_dir, up to 3 levels deep.
    Handles both flat layout (data/<match>/) and nested SoccerNet layout
    (data/<league>/<season>/<match>/).

    A valid match folder must:
      - have a name that passes parse_match_folder() (starts with YYYY-MM-DD)
      - contain 720p.mp4 or 224p.mp4
    """
    matches = []

    def _scan(directory: Path, depth: int):
        if depth > 3:
            return
        try:
            entries = sorted(directory.iterdir())
        except PermissionError:
            return
        for folder in entries:
            if not folder.is_dir():
                continue
            if parse_match_folder(folder):
                has_video = (folder / "720p.mp4").exists() or (folder / "224p.mp4").exists()
                if has_video:
                    matches.append(folder)
                # don't recurse into a valid match folder
            else:
                _scan(folder, depth + 1)

    _scan(data_dir, depth=1)
    result = sorted(matches, key=lambda f: f.name)
    print(f"  [find_match_folders] found {len(result)} valid match folder(s) under {data_dir}")
    return result


def get_video_path(match_folder: Path, prefer_720p: bool = True) -> Optional[Path]:
    """Return best available video in folder."""
    if prefer_720p and (match_folder / "720p.mp4").exists():
        return match_folder / "720p.mp4"
    if (match_folder / "224p.mp4").exists():
        return match_folder / "224p.mp4"
    return None


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE MATCH PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_match(
    match_folder  : Path,
    ekg           : EKG_Graph,
    last_event    : dict,
    roster_global : RosterLookup,
    scraper_global: ESPNScraper,
    max_clips     : int  = None,
    espn_every    : int  = 5,
    clip_duration : int  = CLIP_DURATION,
    clip_step     : int  = CLIP_STEP,
    save_every    : int  = 3,
    use_224p      : bool = False,
) -> dict:
    """
    Process one match folder. Grows the shared EKG in-place.
    Returns summary stats for this match.
    """
    info = parse_match_folder(match_folder)
    if not info:
        print(f"  [skip] cannot parse folder name: {match_folder.name}")
        return {}

    match_date, team1, team2 = info
    match_name = f"{match_date} - {team1} - {team2}"
    video_path = get_video_path(match_folder, prefer_720p=not use_224p)

    if not video_path:
        print(f"  [skip] no video found in {match_folder.name}")
        return {}

    labels_path = match_folder / "Labels-ball.json"
    halftime_sec = get_halftime_sec(labels_path)

    print(f"\n{'═'*70}")
    print(f"  Match  : {match_name}")
    print(f"  Video  : {video_path.name}")
    print(f"  Half   : {int(halftime_sec//60)}:{int(halftime_sec%60):02d}")
    print(f"{'═'*70}")

    # per-match ESPN + roster
    scraper = ESPNScraper()
    n_espn  = scraper.find_and_load(match_date, team1, team2)
    if n_espn == 0:
        print(f"  [skip] no ESPN data for {match_name}")
        return {}

    roster = RosterLookup()
    if scraper.game_id and scraper.league:
        roster.load_from_game_id(scraper.game_id, scraper.league)
        color_map = load_colors_from_game_id(scraper.game_id, scraper.league, team1, team2)
    else:
        roster.load_from_espn(match_date, team1, team2)
        color_map = load_colors_from_espn(match_date, team1, team2)
    roster.set_color_map(color_map)

    # PRE-POPULATE KG with all players before clips run
    print(f"\n  Pre-populating KG with roster...")
    n_prepop = prepopulate_roster(roster, match_name, match_date, ekg)
    print(f"  KG after pre-population: {ekg.stats()}")

    buffer = EventBuffer(dedup_window_sec=60.0)  # was 30s — raised to match cross-batch window
    # Cross-batch dedup: tracks (action, video_time) of events already ingested
    # into the KG. Catches near-duplicates that span different ESPN tick batches
    # (e.g. 5:12 shot in batch N and 5:14 shot in batch N+1 — same real event).
    ingested_cache: list = []   # [(action, video_time_sec), ...]
    CROSS_DEDUP_SEC  = 60.0     # same action within 60s = duplicate

    score_validator = ScoreValidator(scraper, team1, team2)

    video_duration = get_video_duration(video_path)
    total_clips    = int((video_duration - clip_duration) / clip_step) + 1
    if max_clips:
        total_clips = min(total_clips, max_clips)

    print(f"\n  Duration : {video_duration/60:.1f} min  →  {total_clips} clips")
    print(f"{'─'*70}\n")

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    clip_path   = TEMP_DIR / "clip_current.mp4"
    clip_count  = 0
    tick_count  = 0
    start_sec   = 0.0
    t0          = time.time()

    while clip_count < total_clips:

        end_sec  = start_sec + clip_duration
        gametime = seconds_to_gametime(start_sec, halftime_sec)

        print(f"[Clip {clip_count+1:03d}/{total_clips}]  {gametime}  "
              f"({start_sec:.0f}s → {end_sec:.0f}s)")

        t_e0 = time.time()
        ok   = extract_clip(video_path, start_sec, clip_duration, clip_path)
        if not ok:
            print(f"  ⚠ ffmpeg failed — skipping")
            start_sec  += clip_step
            clip_count += 1
            continue
        t_extract = time.time() - t_e0

        t_d0       = time.time()
        # buffer events (pre-flush)
        recent   = buffer.get_recent(start_sec, minutes=2.0)
        recent_d = [{"action": e.action, "gametime": e.gametime}
                    for e in recent]

        # also include recently ingested events (post-flush, last 2 min)
        cutoff_sec = start_sec - 120.0
        for act, t in ingested_cache:
            if t >= cutoff_sec:
                gt    = seconds_to_gametime(t, halftime_sec)
                entry = {"action": act, "gametime": gt}
                if entry not in recent_d:
                    recent_d.append(entry)

        detections = detect_actions(str(clip_path), clip_start_sec=start_sec,
                                    halftime_sec=halftime_sec,
                                    recent_events=recent_d if recent_d else None)
        t_detect   = time.time() - t_d0

        # resolve previously queued Goals against current ESPN score
        current_minute = start_sec / 60.0
        n_added = 0
        for rdet, rorig_start, rorig_gametime, is_goal in score_validator.tick(current_minute):
            if not is_goal:
                rdet = dict(rdet)
                rdet["action"]  = "Shot"
                rdet["outcome"] = rdet.get("outcome") or "on_target"
                print(f"  [score-defer] REJECTED → Shot at {rorig_gametime}")
            else:
                print(f"  [score-defer] CONFIRMED Goal at {rorig_gametime}")
            n_added += buffer.add_from_detections([rdet], rorig_start, rorig_gametime)

        # route this clip: Goals → queue, everything else → buffer now
        for det in detections:
            if det.get("action") == "Goal":
                score_validator.queue(det, start_sec, gametime, start_sec / 60.0)
                print(f"  [score-defer] Goal QUEUED at {det.get('gametime')} — awaiting score")
            else:
                n_added += buffer.add_from_detections([det], start_sec, gametime)

        delete_clip(clip_path)

        if detections:
            d          = detections[0]
            jersey_str = f" #{d['jersey']}" if d.get("jersey") else ""
            desc_short = (d.get("description") or "")[:60]
            print(f"  ✓ extract({t_extract:.1f}s) detect({t_detect:.1f}s) → "
                  f"{d['action']}{jersey_str}")
            if desc_short:
                print(f"    \"{desc_short}...\"")
            print(f"    [+{n_added} to buffer]")
        else:
            print(f"  ✓ extract({t_extract:.1f}s) detect({t_detect:.1f}s) → no action")

        clip_count += 1
        start_sec  += clip_step

        # ESPN TICK
        if clip_count % espn_every == 0 or clip_count == total_clips:
            tick_count    += 1
            current_minute = start_sec / 60.0

            print(f"\n  {'▼'*30}")
            print(f"  ESPN TICK #{tick_count}  at {current_minute:.1f}'")
            print(f"  Buffer: {buffer.size()} events")

            if buffer.size() > 0:
                espn_events = scraper.get_events_up_to(current_minute + 1)
                video_events = buffer.flush()

                matched = align_buffer(
                    video_events, espn_events,
                    time_tolerance_min = 2.0,
                    roster_lookup      = roster,
                    espn_scraper       = None,
                )

                print(summarize(matched))
                print(f"\n  Ingesting into EKG:")

                for m in matched:
                    if m.match_method == "gated":
                        print(f"   GATED    {m.gametime:<12} {m.action:<10} → "
                              f"vlm_score={m.vlm_confidence_score:.2f} (low VLM confidence score, skipped)")
                        continue

                    # cross-batch dedup: skip if same action already ingested within 60s
                    _dup = any(
                        act == m.action
                        and abs(t - m.video_time) <= CROSS_DEDUP_SEC
                        for act, t in ingested_cache
                    )
                    if _dup:
                        print(f"   XDEDUP   {m.gametime:<12} {m.action:<10} → "
                              f"already ingested within 60s, skipped")
                        continue

                    ingested_cache.append((m.action, m.video_time))

                    ingest_matched_event(
                        matched      = m,
                        match_name   = match_name,
                        match_date   = match_date,
                        ekg          = ekg,
                        last_event   = last_event,
                        description  = m.description,
                        jersey       = m.jersey,
                        team_color   = m.team_color,
                        shorts_color = m.shorts_color,
                        socks_color  = m.socks_color,
                        kit_pattern  = m.kit_pattern,
                        pitch_zone   = m.pitch_zone,
                        body_part    = m.body_part,
                        outcome      = m.outcome,
                        foul_type    = m.foul_type,
                        team_side    = m.team_side,
                        ball_visible = m.ball_visible,
                    )
                    event_queue.put_nowait(m)
                    method = f"[{m.match_method}]"
                    player = m.player if m.matched else "?"
                    jersey = f" #{m.jersey}" if m.jersey else ""
                    print(f"   {'MATCHED ' if m.matched else 'UNKNOWN '} "
                          f"{m.gametime:<12} {m.action:<10} → "
                          f"{player}{jersey} {method}")

                print(f"\n  KG now: {ekg.stats()}")

            if tick_count % save_every == 0:
                ekg.save(TTL_PATH)
                print(f"  → saved snapshot: {TTL_PATH.name}")

            print(f"  {'▲'*30}\n")

    # flush any Goals still pending at end of match → downgrade to Shot
    for rdet, rorig_start, rorig_gametime, is_goal in score_validator.flush_all():
        rdet = dict(rdet)
        rdet["action"]  = "Shot"
        rdet["outcome"] = rdet.get("outcome") or "on_target"
        buffer.add_from_detections([rdet], rorig_start, rorig_gametime)

    ekg.save(TTL_PATH)
    elapsed = time.time() - t0

    print(f"\n  Tier 1 jersey matching: {roster.hit_rate_summary()}")

    return {
        "match"       : match_name,
        "clips"       : clip_count,
        "ticks"       : tick_count,
        "time_sec"    : elapsed,
        "jersey_rate" : roster.hit_rate_summary(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR — multi-match
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(
    match_folders : List[Path],
    max_clips     : int  = None,
    espn_every    : int  = 5,
    clip_duration : int  = CLIP_DURATION,
    clip_step     : int  = CLIP_STEP,
    save_every    : int  = 3,
    use_224p      : bool = False,
):
    print("\n" + "═"*70)
    print("  Soccer EKG — Real-Time Pipeline V2 (Qwen3-VL + TKG)")
    print("═"*70)

    # ── checkpoint: load registry + resume from existing TTL ──────────────
    registry = load_registry()
    ekg      = EKG_Graph()
    last_event = {}

    if registry:
        if TTL_PATH.exists():
            ekg.load(str(TTL_PATH))
            print(f"  Checkpoint: loaded {TTL_PATH.name} ({ekg.triple_count()} triples, "
                  f"{len(registry)} match(es) already processed)")
            unprocessed = [f for f in match_folders if f.name not in registry]
            skipped = len(match_folders) - len(unprocessed)
            if skipped:
                print(f"  Checkpoint: skipping {skipped} already-processed match(es):")
                for f in match_folders:
                    if f.name in registry:
                        print(f"    ✓ {f.name}")
            if not unprocessed:
                print("  All matches already processed — nothing to do.")
                return ekg
            match_folders = unprocessed
        else:
            print(f"  WARNING: registry exists ({len(registry)} entries) but "
                  f"{TTL_PATH.name} is missing — reprocessing all matches")
            registry = set()

    print(f"  Matches to process : {len(match_folders)}")
    for f in match_folders:
        print(f"    {f.name}")
    print("═"*70)

    clear_stream()
    print(f"  [viz] event stream cleared → data/kg_output/events_stream.jsonl")

    start_commentator("data/kg_output/ekg.ttl")

    print(f"\n  Loading Qwen3-VL model (once, shared across all matches)...")
    load_model()

    t0_total = time.time()
    summaries = []

    for i, folder in enumerate(match_folders, 1):
        print(f"\n\n{'█'*70}")
        print(f"  MATCH {i}/{len(match_folders)}: {folder.name}")
        print(f"{'█'*70}")

        log_match_boundary(folder.name)

        summary = run_match(
            match_folder   = folder,
            ekg            = ekg,
            last_event     = last_event,
            roster_global  = None,
            scraper_global = None,
            max_clips      = max_clips,
            espn_every     = espn_every,
            clip_duration  = clip_duration,
            clip_step      = clip_step,
            save_every     = save_every,
            use_224p       = use_224p,
        )
        if summary:
            summaries.append(summary)
            # checkpoint: TTL already saved by run_match(); update registry
            registry.add(folder.name)
            save_registry(registry)
            print(f"  Checkpoint: {folder.name} saved to registry ({len(registry)} total)")

    total_time = time.time() - t0_total
    yellow     = ekg.events_by_type("YellowCard")
    red        = ekg.events_by_type("RedCard")

    print("\n" + "═"*70)
    print("  ALL MATCHES COMPLETE")
    print("═"*70)
    for s in summaries:
        print(f"  {s['match']}")
        print(f"    clips={s['clips']}  time={s['time_sec']:.0f}s  {s['jersey_rate']}")
    print(f"{'─'*70}")
    print(f"  Total time  : {total_time:.1f}s")
    print(f"  Final EKG   : {ekg.stats()}")
    print(f"  Yellow cards: {len(yellow)}")
    print(f"  Red cards   : {len(red)}")
    print(f"  Saved to    : {TTL_PATH}")
    print("═"*70 + "\n")

    return ekg


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Soccer EKG Pipeline V2")
    parser.add_argument("--match",      type=str, default=None,
                        help="Specific match folder name (partial match OK)")
    parser.add_argument("--test",       action="store_true",
                        help="5 clips, 224p, first match only")
    parser.add_argument("--clips",      type=int, default=None)
    parser.add_argument("--espn-every", type=int, default=5)
    parser.add_argument("--clip-dur",   type=int, default=CLIP_DURATION)
    parser.add_argument("--clip-step",  type=int, default=CLIP_STEP)
    parser.add_argument("--exp-name",      type=str, default="",
                        help="Experiment name tag — passed to extract step (default: off)")
    parser.add_argument("--use-cot",       action="store_true",
                        help="Chain-of-thought commentary (sets EXP_USE_COT=1)")
    parser.add_argument("--force-history", action="store_true",
                        help="Inject player KG history before each prompt (sets EXP_FORCE_HISTORY=1)")
    args = parser.parse_args()

    # Set experiment env vars before start_commentator() — commentator.py
    # reads these lazily inside agent_commentate(), not at import time.
    if args.use_cot:
        os.environ["EXP_USE_COT"] = "1"
    if args.force_history:
        os.environ["EXP_FORCE_HISTORY"] = "1"

    log_path = setup_logging(DATA_DIR / "logs")
    print(f"  Logging to: {log_path}")

    # discover match folders
    all_folders = find_match_folders(DATA_DIR)

    if not all_folders:
        print(f"ERROR: no match folders found in {DATA_DIR}")
        print("  Expected format: data/YYYY-MM-DD - Team One - Team Two/")
        sys.exit(1)

    # filter by --match flag
    if args.match:
        all_folders = [f for f in all_folders if args.match.lower() in f.name.lower()]
        if not all_folders:
            print(f"ERROR: no match folders matching '{args.match}'")
            sys.exit(1)

    # test mode: first match only, 5 clips, 224p
    if args.test:
        all_folders = [all_folders[0]]
        print(f"  [TEST MODE] {all_folders[0].name}, 5 clips, 224p")

    run_pipeline(
        match_folders = all_folders,
        max_clips     = 5 if args.test else args.clips,
        espn_every    = args.espn_every,
        clip_duration = args.clip_dur,
        clip_step     = args.clip_step,
        use_224p      = args.test,
    )