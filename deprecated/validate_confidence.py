"""
validate_confidence.py — Confidence Formula Validation
───────────────────────────────────────────────────────
Validates the VLM confidence scoring formula against SoccerNet / ESPN ground truth.

Outputs:
  data/validation/score_distribution.png
  data/validation/precision_recall_curve.png
  data/validation/threshold_f1_curve.png

Run:
    python validate_confidence.py
    python validate_confidence.py --ttl path/to/ekg.ttl
    python validate_confidence.py --threshold 0.65
"""

import re
import json
import csv
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR    = Path(__file__).resolve().parent
TTL_PATH    = BASE_DIR / "data" / "kg_output" / "ekg.ttl"
LABELS_PATH = (BASE_DIR / "data" /
               "2019-10-01 - Blackburn Rovers - Nottingham Forest" /
               "Labels-ball.json")
CSV_PATH    = BASE_DIR / "data" / "blackburn_forest_2019-10-01.csv"
OUT_DIR     = BASE_DIR / "data" / "validation"

HALFTIME_SEC  = 2764.0
KEY_ACTIONS   = {"Shot", "Goal", "Foul", "Free_Kick"}
UNKNOWN       = {"unknown", "null", None, "", "none"}
GT_TOLERANCE  = 1.0   # minutes

LABELS_ACTION_MAP = {
    "SHOT"     : "Shot",
    "GOAL"     : "Goal",
    "FREE KICK": "Free_Kick",
}
CSV_ACTIONS = {"Foul"}


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — PARSE TTL
# ═══════════════════════════════════════════════════════════════════════════

def parse_ttl(ttl_path: Path) -> list[dict]:
    """
    Parse ekg.ttl line-by-line. Extract one dict per PlayerAction block.
    Handles multi-type declarations (new TTL format).
    """
    events  = []
    current = {}

    def flush(c):
        if c.get("is_event") and c.get("action") and c.get("time_str"):
            events.append(dict(c))

    with open(ttl_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            # new data:event_NNNN block
            if re.match(r"data:event_\d+\s+a\s+", stripped):
                flush(current)
                current = {"is_pending": True}
                continue

            # non-event data: block → flush and clear
            if re.match(r"data:\S+\s+a\s+", stripped) and not re.match(r"data:event_\d+", stripped):
                flush(current)
                current = {}
                continue

            # promote pending → confirmed PlayerAction
            if current.get("is_pending") and "ekg:PlayerAction" in stripped:
                current["is_event"] = True
                continue

            if not current.get("is_event"):
                continue

            # ── property extraction ───────────────────────────────────────
            def _quoted(s):
                m = re.search(r'"([^"]*)"', s)
                return m.group(1) if m else None

            if "ekg:hasConfidence" in stripped:
                v = _quoted(stripped)
                if v:
                    try:
                        current["base_conf"] = float(v)
                    except ValueError:
                        pass

            elif "ekg:hasEventType" in stripped or "dcterms:type" in stripped:
                v = _quoted(stripped)
                if v:
                    current["action"] = v

            elif "ekg:hasTime " in stripped:
                v = _quoted(stripped)
                if v:
                    current["time_str"] = v

            elif "ekg:detectedJersey" in stripped:
                v = _quoted(stripped)
                if v:
                    current["jersey"] = v

            elif "ekg:hasPitchZone" in stripped:
                v = _quoted(stripped)
                if v:
                    current["pitch_zone"] = v

            elif "ekg:hasBodyPart" in stripped:
                v = _quoted(stripped)
                if v:
                    current["body_part"] = v

            elif "ekg:hasOutcome" in stripped:
                v = _quoted(stripped)
                if v:
                    current["outcome"] = v

            elif "ekg:hasFoulType" in stripped:
                v = _quoted(stripped)
                if v:
                    current["foul_type"] = v

            elif "ekg:inMatch" in stripped:
                m = re.search(r"data:(\S+?)[\s;.]", stripped)
                if m:
                    current["match"] = m.group(1)

    flush(current)
    return [e for e in events if e.get("action") in KEY_ACTIONS]


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — RECOMPUTE CONFIDENCE COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

def recompute_confidence(event: dict) -> dict:
    """
    Recompute the 4 formula components from raw event fields.
    team_color is not stored in TTL → treated as unknown (max completeness = 4/5).
    """
    base         = float(event.get("base_conf") or 0.5)
    jersey       = str(event.get("jersey",     "") or "").lower().strip()
    pitch_zone   = str(event.get("pitch_zone", "") or "").lower().strip()
    body_part    = str(event.get("body_part",  "") or "").lower().strip()
    outcome      = str(event.get("outcome",    "") or "").lower().strip()
    foul_type    = str(event.get("foul_type",  "") or "").lower().strip()
    action       = event.get("action", "")

    # completeness — team_color not in TTL so always counts as unknown
    field_vals = [jersey, "", pitch_zone, body_part, outcome]
    filled     = sum(1 for v in field_vals if v not in UNKNOWN)
    completeness = filled / 5.0

    # jersey bonus
    jersey_bonus = 0.10 if jersey not in UNKNOWN else -0.05

    # semantic consistency
    if action == "Shot":
        consistency = 0.0 if ("own_half" in pitch_zone or outcome in UNKNOWN) else 1.0
    elif action == "Goal":
        consistency = 1.0 if outcome in {"scored", "goal"} else 0.5
    elif action == "Foul":
        consistency = 1.0 if foul_type not in UNKNOWN else 0.5
    else:
        consistency = 1.0

    score = (0.50 * base
             + 0.30 * completeness
             + 0.10 * jersey_bonus
             + 0.10 * consistency)
    score = round(min(max(score, 0.0), 1.0), 4)

    return {
        "base"        : base,
        "completeness": completeness,
        "jersey_bonus": jersey_bonus,
        "consistency" : consistency,
        "score"       : score,
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — GROUND TRUTH
# ═══════════════════════════════════════════════════════════════════════════

def _kg_time_to_min(time_str: str) -> float:
    try:
        half, t = time_str.strip().split(" ", 1)
        mm, ss  = t.strip().split(":")
        secs    = int(mm) * 60 + int(ss)
        if half == "2nd":
            secs += int(HALFTIME_SEC)
        return secs / 60.0
    except Exception:
        return 0.0


def _labels_time_to_min(game_time: str) -> float:
    try:
        half_str, t = game_time.split(" - ")
        mm, ss      = t.strip().split(":")
        secs        = int(mm) * 60 + int(ss)
        if int(half_str.strip()) == 2:
            secs += int(HALFTIME_SEC)
        return secs / 60.0
    except Exception:
        return 0.0


def _csv_time_to_min(time_str: str) -> float:
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


def load_ground_truth(labels_path: Path, csv_path: Path) -> list[dict]:
    gt = []

    if labels_path.exists():
        with open(labels_path, encoding="utf-8") as f:
            data = json.load(f)
        for ann in data.get("annotations", []):
            label  = ann.get("label", "").upper().strip()
            action = LABELS_ACTION_MAP.get(label)
            if not action:
                continue
            t = _labels_time_to_min(ann.get("gameTime", ""))
            if t > 0:
                gt.append({"action": action, "time_min": t})
    else:
        print(f"  WARNING: Labels-ball.json not found — Shot/Goal/Free_Kick GT = 0")

    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                raw = row.get("Action_Type", "").strip()
                if raw not in CSV_ACTIONS:
                    continue
                t = _csv_time_to_min(row.get("Time", "0"))
                if t > 0:
                    gt.append({"action": raw, "time_min": t})
    else:
        print(f"  WARNING: ESPN CSV not found — Foul GT = 0")

    gt.sort(key=lambda e: e["time_min"])
    return gt


def label_events(events: list[dict], gt: list[dict], tolerance: float) -> list[int]:
    """Greedy one-to-one matching. Returns list of 1 (TP) or 0 (FP) per event."""
    labels   = [0] * len(events)
    gt_used  = set()

    for ei, ev in enumerate(events):
        best_j, best_diff = None, float("inf")
        for gi, g in enumerate(gt):
            if gi in gt_used:
                continue
            if ev["action"] != g["action"]:
                continue
            diff = abs(ev["time_min"] - g["time_min"])
            if diff <= tolerance and diff < best_diff:
                best_diff, best_j = diff, gi
        if best_j is not None:
            gt_used.add(best_j)
            labels[ei] = 1

    return labels


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def make_plots(scores: list[float], labels: list[int],
               threshold: float, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = np.array(scores)
    labels = np.array(labels)
    tp_scores = scores[labels == 1]
    fp_scores = scores[labels == 0]

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: score distribution ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(tp_scores, bins=bins, color="green", alpha=0.6,
            label=f"TP (n={len(tp_scores)})")
    ax.hist(fp_scores, bins=bins, color="red",   alpha=0.6,
            label=f"FP (n={len(fp_scores)})")
    ax.axvline(threshold, color="black", linestyle="--",
               label=f"threshold={threshold}")
    ax.set_title("Confidence Score Distribution: TP vs FP")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "score_distribution.png", dpi=120)
    plt.close(fig)

    # ── Plot 2: precision-recall curve ────────────────────────────────────
    thresholds = np.arange(0.0, 1.01, 0.02)
    precisions, recalls = [], []
    best_f1, best_thr, best_pr, best_rec = 0.0, 0.0, 0.0, 0.0

    for thr in thresholds:
        pred  = (scores >= thr).astype(int)
        tp    = int(np.sum((pred == 1) & (labels == 1)))
        fp    = int(np.sum((pred == 1) & (labels == 0)))
        fn    = int(np.sum((pred == 0) & (labels == 1)))
        prec  = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        if f1 > best_f1:
            best_f1, best_thr, best_pr, best_rec = f1, thr, prec, rec

    # mark current threshold point
    pred_cur  = (scores >= threshold).astype(int)
    tp_cur    = int(np.sum((pred_cur == 1) & (labels == 1)))
    fp_cur    = int(np.sum((pred_cur == 1) & (labels == 0)))
    fn_cur    = int(np.sum((pred_cur == 0) & (labels == 1)))
    prec_cur  = tp_cur / (tp_cur + fp_cur) if (tp_cur + fp_cur) > 0 else 1.0
    rec_cur   = tp_cur / (tp_cur + fn_cur) if (tp_cur + fn_cur) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recalls, precisions, color="steelblue", linewidth=2)
    ax.scatter([rec_cur], [prec_cur], color="red",    s=80, zorder=5,
               label=f"current thr={threshold:.2f}")
    ax.scatter([best_rec], [best_pr], color="gold", marker="*", s=180, zorder=5,
               label=f"best F1={best_f1:.3f} @ thr={best_thr:.2f}")
    ax.set_title("Precision-Recall Curve (threshold sweep)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "precision_recall_curve.png", dpi=120)
    plt.close(fig)

    # ── Plot 3: F1 vs threshold ───────────────────────────────────────────
    f1_scores = []
    for prec, rec in zip(precisions, recalls):
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1_scores, color="steelblue", linewidth=2)
    ax.axvline(threshold, color="red", linestyle="--",
               label=f"current threshold={threshold:.2f}")
    ax.set_title("F1 Score vs Confidence Threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_f1_curve.png", dpi=120)
    plt.close(fig)

    return {
        "thresholds": thresholds,
        "precisions": precisions,
        "recalls"   : recalls,
        "f1_scores" : f1_scores,
        "best_f1"   : best_f1,
        "best_thr"  : best_thr,
        "cur_prec"  : prec_cur,
        "cur_rec"   : rec_cur,
        "cur_f1"    : 2 * prec_cur * rec_cur / (prec_cur + rec_cur)
                      if (prec_cur + rec_cur) > 0 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — LOGISTIC REGRESSION WEIGHT COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

HAND_WEIGHTS = {
    "base"        : 0.50,
    "completeness": 0.30,
    "jersey_bonus": 0.10,
    "consistency" : 0.10,
}


def lr_weight_comparison(features: list[list], labels: list[int]) -> dict:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {}

    tp_count = sum(labels)
    fp_count = len(labels) - tp_count
    if tp_count < 5 or fp_count < 5:
        return {"skip": True, "reason": f"too few samples (TP={tp_count} FP={fp_count})"}

    X = np.array(features)
    y = np.array(labels)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    # normalize coefficients so max abs = 1.0 for direction comparison
    coefs = model.coef_[0]
    max_abs = np.max(np.abs(coefs)) or 1.0
    normed = coefs / max_abs

    feature_names = list(HAND_WEIGHTS.keys())
    return {
        "skip"   : False,
        "coefs"  : dict(zip(feature_names, coefs)),
        "normed" : dict(zip(feature_names, normed)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main(args):
    ttl_path  = Path(args.ttl)
    threshold = args.threshold

    print("\n─── Confidence Formula Validation ───")
    print(f"  TTL       : {ttl_path}")
    print(f"  Threshold : {threshold}")

    if not ttl_path.exists():
        print(f"\nERROR: {ttl_path} not found")
        return

    # Step 1 — parse TTL
    raw_events = parse_ttl(ttl_path)
    print(f"\n  Events loaded from TTL : {len(raw_events)}")

    # Step 2 — recompute components
    for ev in raw_events:
        comps = recompute_confidence(ev)
        ev.update(comps)
        ev["time_min"] = _kg_time_to_min(ev.get("time_str", ""))

    # Step 3 — label TP/FP
    gt     = load_ground_truth(LABELS_PATH, CSV_PATH)
    labels = label_events(raw_events, gt, tolerance=GT_TOLERANCE)

    scores   = [ev["score"]        for ev in raw_events]
    feats    = [[ev["base"], ev["completeness"],
                 ev["jersey_bonus"], ev["consistency"]]
                for ev in raw_events]

    tp_total = sum(labels)
    fp_total = len(labels) - tp_total

    # Step 4 — plots
    plot_stats = make_plots(scores, labels, threshold, OUT_DIR)

    # Step 5 — LR comparison
    lr = lr_weight_comparison(feats, labels)

    # ── print results ─────────────────────────────────────────────────────
    W = 54
    print(f"\n{'═'*W}")
    print(f"  Confidence Formula — Weight Validation")
    print(f"{'═'*W}")

    if lr.get("skip"):
        print(f"  LR skipped: {lr.get('reason')}")
    elif lr:
        normed = lr["normed"]
        print(f"  {'Feature':<16} {'Hand-picked':>12} {'Learned (LR)':>14}  {'Match?':>6}")
        print(f"  {'─'*16} {'─'*12} {'─'*14}  {'─'*6}")
        for feat, hand in HAND_WEIGHTS.items():
            lr_val = normed.get(feat, 0.0)
            match  = "✓" if lr_val > 0 else "✗"
            print(f"  {feat:<16} {hand:>12.2f} {lr_val:>14.3f}  {match:>6}")
        print(f"  {'─'*16} {'─'*12} {'─'*14}  {'─'*6}")
        print(f"  Note: LR weights are direction-normalized (sign only),")
        print(f"        not directly comparable in magnitude.")

    print(f"\n{'═'*W}")
    print(f"  Total events   : {len(raw_events)}")
    print(f"  TP             : {tp_total}  (GT-matched)")
    print(f"  FP             : {fp_total}  (no GT match)")
    print(f"  Current threshold ({threshold:.2f}):")
    print(f"    Precision : {plot_stats['cur_prec']:.3f}")
    print(f"    Recall    : {plot_stats['cur_rec']:.3f}")
    print(f"    F1        : {plot_stats['cur_f1']:.3f}")
    print(f"  Best threshold : {plot_stats['best_thr']:.2f} "
          f"(F1 = {plot_stats['best_f1']:.3f})")
    print(f"{'═'*W}")
    print(f"\n  Plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttl",       default=str(TTL_PATH))
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()
    main(args)
