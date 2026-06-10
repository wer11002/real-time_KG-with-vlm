"""
evaluate.py — Soccer EKG Real Evaluation (D2)
──────────────────────────────────────────────

Reads the ACTUAL pipeline output from ekg.ttl and compares
against real ground truth sources. No simulation.

Ground truth (per action type):
  Shot      → Labels-ball.json  (millisecond-accurate SoccerNet annotations)
  Goal      → Labels-ball.json
  Free_Kick → Labels-ball.json
  Foul      → ESPN CSV          (Labels-ball.json does not annotate fouls)
  Corner    → ESPN CSV          (Labels-ball.json does not annotate corners)

Pipeline output:
  Reads ekg.ttl → extracts all PlayerActions for Blackburn match
  Converts hasTime string → float minutes for comparison

Matching rule:
  A pipeline event matches a GT event if:
    1. action type matches exactly
    2. |pipeline_time - gt_time| <= tolerance (default 1.0 min = 60 s)
    3. GT event not already matched (greedy, one-to-one)

Run:
    python evaluate.py                         # default
    python evaluate.py --ttl path/to/ekg.ttl   # custom KG path
    python evaluate.py --tolerance 1.0          # looser
    python evaluate.py --match blackburn        # specific match
    python evaluate.py --verbose                # show TP/FP/FN detail
    python evaluate.py --csv path/to/espn.csv  # override ESPN CSV path
"""

import re
import json
import argparse
from pathlib import Path
from collections import Counter

import csv as csv_module

BASE_DIR    = Path(__file__).resolve().parent
TTL_PATH    = BASE_DIR / "data" / "kg_output" / "ekg.ttl"
LABELS_PATH = (BASE_DIR / "data" /
               "2019-10-01 - Blackburn Rovers - Nottingham Forest" /
               "Labels-ball.json")
CSV_PATH    = BASE_DIR / "data" / "blackburn_forest_2019-10-01.csv"

# Actions sourced from ESPN CSV (not in Labels-ball.json)
CSV_ACTIONS = {"Foul", "Corner"}

HALFTIME_SEC = 2764.0
KEY_ACTIONS  = {"Shot", "Goal", "Foul", "Corner", "Free_Kick"}

ACTION_NORM = {
    "Shot"     : "Shot",
    "Goal"     : "Goal",
    "Foul"     : "Foul",
    "Corner"   : "Corner",
    "FreeKick" : "Free_Kick",
    "Free_Kick": "Free_Kick",
}


# ═══════════════════════════════════════════════════════════════════════════
# TIME HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def kg_time_to_min(time_str: str) -> float:
    """"1st 23:45" → 23.75,  "2nd 07:30" → halftime + 7.5"""
    try:
        half, t = time_str.strip().split(" ", 1)
        mm, ss  = t.strip().split(":")
        secs    = int(mm) * 60 + int(ss)
        if half == "2nd":
            secs += int(HALFTIME_SEC)
        return secs / 60.0
    except Exception:
        return 0.0


def labels_time_to_min(game_time: str) -> float:
    """SoccerNet "1 - 09:14" → float minutes"""
    try:
        half_str, t = game_time.split(" - ")
        mm, ss      = t.strip().split(":")
        secs        = int(mm) * 60 + int(ss)
        if int(half_str.strip()) == 2:
            secs += int(HALFTIME_SEC)
        return secs / 60.0
    except Exception:
        return 0.0


def csv_time_to_min(time_str: str) -> float:
    """ESPN "23'" or "45+2'" → float minutes"""
    t = re.sub(r"'", "", str(time_str).strip())
    if "+" in t:
        base, extra = t.split("+", 1)
        try:
            return float(base) + float(extra)
        except Exception:
            return 0.0
    try:
        return float(t)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# LOAD KG EVENTS (pure regex parse — no rdflib needed)
# ═══════════════════════════════════════════════════════════════════════════

def load_kg_events(ttl_path: Path, match_filter: str = None) -> list:
    """
    Parse ekg.ttl line by line.
    Extracts PlayerAction blocks with action, time, matched, jersey, match.
    """
    events  = []
    current = {}

    def flush(c):
        if c.get("is_event") and c.get("action") and c.get("time_str"):
            events.append(dict(c))
        # pending but never confirmed = not an PlayerAction block, discard silently

    with open(ttl_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            stripped = line.strip()

            # new event block — starts with data:event_<any-id>
            if re.match(r"data:event_\S+\s+a\s+", stripped):
                flush(current)
                current = {"is_pending": True, "matched": False}
                continue

            # new non-event block → flush and clear
            if re.match(r"data:\S+\s+a\s+", stripped) and not re.match(r"data:event_\S+", stripped):
                flush(current)
                current = {}
                continue

            # confirm it is an PlayerAction once we see the type in the block
            if current.get("is_pending") and "ekg:PlayerAction" in stripped:
                current["is_event"] = True
                continue

            if not current.get("is_event"):
                continue

            if "ekg:hasEventType" in stripped or "dcterms:type" in stripped:
                m = re.search(r'"([^"]+)"', stripped)
                if m:
                    current["action"] = m.group(1)

            elif "ekg:hasTime " in stripped:
                m = re.search(r'"([^"]+)"', stripped)
                if m:
                    current["time_str"] = m.group(1)
                    current["time_min"] = kg_time_to_min(m.group(1))

            elif "ekg:isMatched" in stripped:
                current["matched"] = "true" in stripped.lower()

            elif "ekg:hasJerseyNumber" in stripped:
                m = re.search(r'"([^"]+)"', stripped)
                if m:
                    current["jersey"] = m.group(1)

            elif "ekg:inMatch" in stripped:
                m = re.search(r"data:(\S+?)[\s;.]", stripped)
                if m:
                    current["match"] = m.group(1)

            elif "ekg:hasDescription" in stripped or "dcterms:description" in stripped:
                m = re.search(r'"([^"]+)"', stripped)
                if m:
                    current["description"] = m.group(1)

    flush(current)

    # keep only KEY_ACTIONS
    events = [e for e in events if e.get("action") in KEY_ACTIONS]

    # filter by match
    if match_filter:
        events = [e for e in events
                  if match_filter.lower() in e.get("match", "").lower()]

    return sorted(events, key=lambda e: e.get("time_min", 0))


# ═══════════════════════════════════════════════════════════════════════════
# LOAD GROUND TRUTH
# ═══════════════════════════════════════════════════════════════════════════

# Labels-ball.json action labels → normalized KEY_ACTIONS names.
# SHOT, GOAL, and FREE KICK are present in Labels-ball.json with
# millisecond precision and are used preferentially over the ESPN CSV.
# FOUL and CORNER are not in Labels-ball.json so ESPN CSV covers those.
LABELS_ACTION_MAP = {
    "SHOT"     : "Shot",
    "GOAL"     : "Goal",
    "FREE KICK": "Free_Kick",
    "PASS"     : "Pass",
    "HIGH PASS": "Pass",
    "CROSS"    : "Pass",
}

# Actions sourced from Labels-ball.json (don't double-load from ESPN CSV)
LABELS_COVERED = set(LABELS_ACTION_MAP.values())


def load_gt_labels(labels_path: Path) -> list:
    """Load all relevant action annotations from Labels-ball.json."""
    with open(labels_path, encoding="utf-8") as f:
        data = json.load(f)
    events = []
    for ann in data.get("annotations", []):
        label  = ann.get("label", "").upper().strip()
        action = LABELS_ACTION_MAP.get(label)
        if not action:
            continue
        t = labels_time_to_min(ann.get("gameTime", ""))
        if t > 0:
            events.append({"action": action, "time_min": t,
                           "player": "", "source": "Labels-ball.json"})
    return events


def load_gt_csv(csv_path: Path, actions: set = None) -> list:
    """
    Load GT events from ESPN CSV for action types not in Labels-ball.json.
    CSV columns expected: Match, Team, Time, Player, Action_Type, Full_Text, ...
    Only loads rows where Action_Type is in `actions` (default: CSV_ACTIONS).
    """
    if actions is None:
        actions = CSV_ACTIONS
    events = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            raw_action = row.get("Action_Type", "").strip()
            action     = ACTION_NORM.get(raw_action, raw_action)
            if action not in actions:
                continue
            t = csv_time_to_min(row.get("Time", "0"))
            if t > 0:
                events.append({
                    "action" : action,
                    "time_min": t,
                    "player" : row.get("Player", ""),
                    "source" : "ESPN CSV",
                })
    return events


def load_ground_truth(labels_path: Path, csv_path: Path = None) -> list:
    """
    Load GT from Labels-ball.json (Shot, Goal, Free_Kick) and optionally
    from ESPN CSV (Foul, Corner — not covered by Labels-ball.json).
    """
    gt = load_gt_labels(labels_path)
    if csv_path and csv_path.exists():
        gt += load_gt_csv(csv_path, actions=CSV_ACTIONS)
    elif csv_path:
        print(f"  WARNING: ESPN CSV not found at {csv_path} — Foul/Corner GT = 0")
    gt.sort(key=lambda e: e["time_min"])
    return gt


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATE
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(pipeline: list, gt: list,
             tolerance: float = 1.0,
             matched_only: bool = False) -> dict:

    det = [e for e in pipeline
           if e.get("action") in KEY_ACTIONS
           and (not matched_only or e.get("matched"))]

    gt_f = [e for e in gt if e.get("action") in KEY_ACTIONS]

    gt_used  = set()
    det_used = set()
    matches  = []

    for di, d in enumerate(det):
        best_j, best_diff = None, float("inf")
        for gi, g in enumerate(gt_f):
            if gi in gt_used:
                continue
            if d["action"] != g["action"]:
                continue
            diff = abs(d["time_min"] - g["time_min"])
            if diff <= tolerance and diff < best_diff:
                best_diff, best_j = diff, gi
        if best_j is not None:
            gt_used.add(best_j)
            det_used.add(di)
            matches.append((di, best_j, best_diff))

    per_action = {}
    for action in sorted(KEY_ACTIONS):
        det_a = [i for i, e in enumerate(det) if e["action"] == action]
        gt_a  = [i for i, e in enumerate(gt_f) if e["action"] == action]
        tp    = sum(1 for di, gi, _ in matches
                    if det[di]["action"] == action)
        fp    = len(det_a) - tp
        fn    = len(gt_a)  - tp
        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1    = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        src = "ESPN CSV" if action in CSV_ACTIONS else "Labels-ball.json"
        per_action[action] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 3),
            "recall"   : round(rec,  3),
            "f1"       : round(f1,   3),
            "n_det"    : len(det_a),
            "n_gt"     : len(gt_a),
            "source"   : src,
        }

    total_tp = len(det_used)
    total_fp = len(det) - total_tp
    total_fn = len(gt_f) - total_tp
    op   = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    orec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    of1  = 2*op*orec / (op+orec) if (op+orec) > 0 else 0.0

    return {
        "per_action"        : per_action,
        "overall_precision" : round(op,   3),
        "overall_recall"    : round(orec, 3),
        "overall_f1"        : round(of1,  3),
        "total_tp"          : total_tp,
        "total_fp"          : total_fp,
        "total_fn"          : total_fn,
        "n_det"             : len(det),
        "n_gt"              : len(gt_f),
        "matches"           : matches,
        "det"               : det,
        "gt"                : gt_f,
    }


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

def print_results(results: dict, label: str = ""):
    w = 76
    print(f"\n{'═'*w}")
    if label:
        print(f"  {label}")
    print(f"{'═'*w}")
    print(f"  Pipeline detections : {results['n_det']}")
    print(f"  Ground truth events : {results['n_gt']}")
    print(f"  TP={results['total_tp']}  "
          f"FP={results['total_fp']}  "
          f"FN={results['total_fn']}")
    print(f"{'─'*w}")
    print(f"  {'Action':<12} {'Prec':>7} {'Rec':>7} {'F1':>7}  "
          f"{'TP':>4} {'FP':>4} {'FN':>4}  "
          f"{'Det':>5} {'GT':>4}  Source")
    print(f"  {'─'*12} {'─'*7} {'─'*7} {'─'*7}  "
          f"{'─'*4} {'─'*4} {'─'*4}  {'─'*5} {'─'*4}")
    for action, m in sorted(results["per_action"].items()):
        print(f"  {action:<12} {m['precision']:>7.3f} {m['recall']:>7.3f} "
              f"{m['f1']:>7.3f}  {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}  "
              f"{m['n_det']:>5} {m['n_gt']:>4}  {m['source']}")
    print(f"  {'─'*12} {'─'*7} {'─'*7} {'─'*7}")
    print(f"  {'OVERALL':<12} {results['overall_precision']:>7.3f} "
          f"{results['overall_recall']:>7.3f} "
          f"{results['overall_f1']:>7.3f}")
    print(f"{'═'*w}")


def print_verbose(results: dict):
    det = results["det"]
    gt  = results["gt"]
    matched_det = {di for di, gi, _ in results["matches"]}
    matched_gt  = {gi for di, gi, _ in results["matches"]}

    print(f"\n{'─'*76}")
    print(f"  TRUE POSITIVES ({results['total_tp']})")
    print(f"{'─'*76}")
    for di, gi, diff in sorted(results["matches"],
                                key=lambda x: det[x[0]]["time_min"]):
        d = det[di]
        g = gt[gi]
        print(f"  ✓ {d.get('time_str','?'):<12} {d['action']:<10} "
              f"pipeline={d['time_min']:.1f}'  "
              f"GT={g['time_min']:.1f}'  Δ={diff:.2f}min"
              f"{'  ⚠ large Δt' if diff > 1.0 else ''}  "
              f"[{g['source']}]")

    print(f"\n{'─'*76}")
    print(f"  FALSE POSITIVES ({results['total_fp']}) — pipeline detected, not in GT")
    print(f"{'─'*76}")
    for di, d in enumerate(det):
        if di not in matched_det:
            jersey = f"#{d['jersey']}" if d.get("jersey") else "   "
            print(f"  ✗ {d.get('time_str','?'):<12} {d['action']:<10} "
                  f"{jersey:<5} matched={d.get('matched')}")

    print(f"\n{'─'*76}")
    print(f"  FALSE NEGATIVES ({results['total_fn']}) — in GT, pipeline missed")
    print(f"{'─'*76}")
    for gi, g in enumerate(gt):
        if gi not in matched_gt:
            player = g.get("player", "?") or "?"
            print(f"  ✗ {g['time_min']:>6.1f}'  {g['action']:<10} "
                  f"{player:<28} [{g['source']}]")


def print_pipeline_summary(events: list, gt: list, match_filter: str = ""):
    total   = len(events)
    matched = sum(1 for e in events if e.get("matched"))
    counts  = Counter(e["action"] for e in events)
    gt_cnt  = Counter(e["action"] for e in gt)

    print(f"\n{'─'*76}")
    print(f"  PIPELINE OUTPUT SUMMARY (match: {match_filter})")
    print(f"{'─'*76}")
    print(f"  Total KG events   : {total}")
    print(f"  Matched (T1+T2)   : {matched}")
    print(f"  Unmatched         : {total - matched}")
    print(f"\n  {'Action':<12} {'KG total':>10} {'Matched':>9} {'GT':>6}")
    print(f"  {'─'*12} {'─'*10} {'─'*9} {'─'*6}")
    for action in sorted(KEY_ACTIONS):
        n = counts.get(action, 0)
        m = sum(1 for e in events
                if e.get("action") == action and e.get("matched"))
        g = gt_cnt.get(action, 0)
        print(f"  {action:<12} {n:>10} {m:>9} {g:>6}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main(args):
    print("\n─── Soccer EKG — Real Evaluation (actual pipeline output) ───")
    print(f"  KG file   : {args.ttl}")
    print(f"  Tolerance : ±{args.tolerance} min")

    ttl_path = Path(args.ttl)
    if not ttl_path.exists():
        print(f"\nERROR: {ttl_path} not found — run python main.py first")
        return

    match_filter = args.match or "blackburn"
    kg_events = load_kg_events(ttl_path, match_filter=match_filter)
    print(f"\n  Loaded {len(kg_events)} events from KG "
          f"(filter: '{match_filter}')")

    if not kg_events:
        print("  No events found — check --match filter or run the pipeline")
        return

    csv_path = Path(args.csv) if args.csv else CSV_PATH
    labels_path = Path(args.labels) if args.labels else LABELS_PATH
    gt = load_ground_truth(labels_path, csv_path=csv_path)
    if args.coverage_min is not None:
        before = len(gt)
        gt = [e for e in gt if e["time_min"] <= args.coverage_min]
        print(f"\n  Coverage filter: ≤ {args.coverage_min} min "
              f"({len(gt)}/{before} GT events in window)")

    gt_counts = Counter(e["action"] for e in gt)
    csv_note  = f" + ESPN CSV ({csv_path.name})" if csv_path.exists() else " (no ESPN CSV)"

    print(f"\n  Ground truth: {len(gt)} events total "
          f"(Labels-ball.json{csv_note})")
    for action in sorted(KEY_ACTIONS):
        print(f"    {action:<12} {gt_counts.get(action,0):>3}")

    print_pipeline_summary(kg_events, gt, match_filter=match_filter)

    # eval 1: all pipeline events
    r_all = evaluate(kg_events, gt,
                     tolerance=args.tolerance, matched_only=False)
    print_results(r_all,
                  "ALL pipeline events (matched + unmatched)")

    # eval 2: matched only
    r_mat = evaluate(kg_events, gt,
                     tolerance=args.tolerance, matched_only=True)
    print_results(r_mat,
                  "MATCHED events only (Tier 1 jersey + Tier 2 time)")

    # summary comparison
    print(f"\n{'─'*76}")
    print(f"  SUMMARY")
    print(f"{'─'*76}")
    print(f"  {'':22} {'Precision':>10} {'Recall':>8} {'F1':>8}  {'Events':>8}")
    print(f"  {'─'*22} {'─'*10} {'─'*8} {'─'*8}  {'─'*8}")
    print(f"  {'All events (TP+FP)':22} "
          f"{r_all['overall_precision']:>10.3f} "
          f"{r_all['overall_recall']:>8.3f} "
          f"{r_all['overall_f1']:>8.3f}  "
          f"{r_all['n_det']:>8}")
    print(f"  {'Matched only':22} "
          f"{r_mat['overall_precision']:>10.3f} "
          f"{r_mat['overall_recall']:>8.3f} "
          f"{r_mat['overall_f1']:>8.3f}  "
          f"{r_mat['n_det']:>8}")
    print(f"{'─'*76}")

    if args.verbose:
        print("\n\n  === VERBOSE DETAIL: ALL EVENTS ===")
        print_verbose(r_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttl",       default=str(TTL_PATH),
                        help="Path to ekg.ttl")
    parser.add_argument("--tolerance", type=float, default=1.0,
                        help="Time tolerance in minutes (default 1.0 = 60s)")
    parser.add_argument("--match",     type=str,   default=None,
                        help="Match filter string (default: blackburn)")
    parser.add_argument("--verbose",      action="store_true",
                        help="Show all TP/FP/FN details")
    parser.add_argument("--coverage-min", type=float, default=None,
                        dest="coverage_min",
                        help="Exclude GT events after this minute (e.g. 35 for 70-clip test)")
    parser.add_argument("--csv",          type=str,   default=None,
                        help="Path to ESPN CSV for Foul/Corner GT (default: data/blackburn_forest_2019-10-01.csv)")
    parser.add_argument("--labels",       type=str,   default=None,
                        help="Path to Labels-ball.json (default: Blackburn match)")
    args = parser.parse_args()
    main(args)