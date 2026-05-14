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
    EKG_Graph, ingest_matched_event, prepopulate_roster, TTL_PATH
)

DATA_DIR      = BASE_DIR / "data"
CLIP_DURATION = 60
CLIP_STEP     = 30


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
    Find all valid match folders in data_dir, sorted by date (oldest first).
    A valid match folder must contain a 720p.mp4 or 224p.mp4 file.
    """
    matches = []
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        info = parse_match_folder(folder)
        if not info:
            continue
        # must have a video file
        has_video = (folder / "720p.mp4").exists() or (folder / "224p.mp4").exists()
        if not has_video:
            continue
        matches.append(folder)

    return sorted(matches, key=lambda f: f.name)


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

    buffer = EventBuffer(dedup_window_sec=30.0)

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
        recent   = buffer.get_recent(start_sec, minutes=2.0)
        recent_d = [{"action": e.action, "gametime": e.gametime} for e in recent]
        detections = detect_actions(str(clip_path), clip_start_sec=start_sec,
                                    halftime_sec=halftime_sec,
                                    recent_events=recent_d if recent_d else None)
        t_detect   = time.time() - t_d0

        n_added = buffer.add_from_detections(detections, start_sec, gametime)
        delete_clip(clip_path)

        if detections:
            d          = detections[0]
            jersey_str = f" #{d['jersey']}" if d.get("jersey") else ""
            desc_short = (d.get("description") or "")[:60]
            print(f"  ✓ extract({t_extract:.1f}s) detect({t_detect:.1f}s) → "
                  f"{d['action']}{jersey_str} conf={d['confidence']:.2f}")
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
                espn_events  = scraper.get_events_up_to(current_minute + 1)
                video_events = buffer.flush()

                matched = align_buffer(
                    video_events, espn_events,
                    time_tolerance_min = 2.0,
                    roster_lookup      = roster,
                    espn_scraper       = scraper,
                )

                print(summarize(matched))
                print(f"\n  Ingesting into EKG:")

                for m in matched:
                    if m.match_method == "gated":
                        print(f"   GATED    {m.gametime:<12} {m.action:<10} → "
                              f"conf={m.confidence:.2f} (ESPN contradicts, skipped)")
                        continue
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
                    )
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
    print(f"  Matches to process : {len(match_folders)}")
    for f in match_folders:
        print(f"    {f.name}")
    print("═"*70)

    # shared across ALL matches
    ekg        = EKG_Graph()
    last_event = {}

    print(f"\n  Loading Qwen3-VL model (once, shared across all matches)...")
    load_model()

    t0_total = time.time()
    summaries = []

    for i, folder in enumerate(match_folders, 1):
        print(f"\n\n{'█'*70}")
        print(f"  MATCH {i}/{len(match_folders)}: {folder.name}")
        print(f"{'█'*70}")

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

    # final save
    ekg.save(TTL_PATH)

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
    args = parser.parse_args()

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