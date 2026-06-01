"""
Commentary Evaluation — Human vs AI Agent
Outputs a labeled side-by-side report with scores A, B, C per event
and a summary table of mismatches.

Single-match (legacy, text log):
    python src/commentator/evaluate_commentary.py \
        --ai-log  data/commentator_output/commentary_log.txt \
        --human-json data/human_commentary.json \
        --espn-csv   data/blackburn_forest_2019-10-01.csv

Single-match (JSON files):
    python src/commentator/evaluate_commentary.py \
        --gt-file  ground_truth_commentary.json \
        --ai-file  ai_commentary.json \
        --match-dir "data/2019-10-01 - Blackburn Rovers - Nottingham Forest"

All matches:
    python src/commentator/evaluate_commentary.py --all
    python src/commentator/evaluate_commentary.py --all \
        --gt-file ground_truth_commentary.json \
        --ai-file ai_commentary.json
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = BASE_DIR / "data"
OUT_DIR   = BASE_DIR / "data" / "commentator_output"


def _has_cjk(text: str) -> bool:
    return any(
        (unicodedata.category(c) in ('Lo',) and '一' <= c <= '鿿')
        or '　' <= c <= '〿'
        or '＀' <= c <= '￯'
        for c in text
    )

PAST_REF_KEYWORDS = [
    "again", "second", "third", "another", "has been", "continues",
    "following", "always", "still", "once more", "earlier",
    "keeps the pressure", "first time", "pressure mounting",
]


# ── Parse AI commentary log ────────────────────────────────────────────────

def parse_ai_log(path):
    events  = []
    pattern = re.compile(r"\[(\d+)(?:st|nd)\s+(\d+):(\d+)\]\s+(\w+)\s+\|(.+)")
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        half, mins, _, etype, text = m.groups()
        if _has_cjk(text):
            continue
        events.append({
            "half"      : int(half),
            "minute"    : int(mins),
            "event_type": etype.strip(),
            "full_text" : text.strip(),
        })
    return events


# ── Find closest AI match (±2 min, same half + event type) ────────────────

def find_match(human, ai_events, tol=2.0):
    def abs_min(e):
        return e["minute"] + (45 if e["half"] == 2 else 0)

    human_abs = human["minute"] + (45 if human["half"] == 2 else 0)
    same = [
        e for e in ai_events
        if e["event_type"].lower() == human["event_type"].lower()
        and abs(abs_min(e) - human_abs) <= tol
    ]
    return min(same, key=lambda e: abs(abs_min(e) - human_abs)) if same else None


# ── Metric A — BLEU ────────────────────────────────────────────────────────

def metric_a_bleu(ref, hyp):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    if not hyp:
        return 0.0
    return round(
        sentence_bleu(
            [ref.lower().split()], hyp.lower().split(),
            weights=(0.5, 0.5),
            smoothing_function=SmoothingFunction().method1,
        ), 3,
    )


# ── Metric B — Fact Overlap ────────────────────────────────────────────────

def metric_b_fact(human, ai_text):
    from thefuzz import fuzz
    t          = ai_text.lower()
    player_ok  = any(p in t for p in human["player"].lower().split())
    team_ok    = any(fuzz.partial_ratio(w, t) > 75 for w in human["team"].lower().split())
    OUTCOMES = {
        "Goal"        : ["goal", "scores", "scored", "fires home", "slots",
                         "nets", "buries", "puts away"],
        "Shot"        : ["saved", "blocked", "missed", "wide", "over",
                         "attempt", "shot"],
        "Foul"        : ["foul", "tackle", "challenge", "card", "brings down"],
        "Corner"      : ["corner"],
        "Free_Kick"   : ["free kick", "freekick"],
        "Substitution": ["substitut", "replac", "comes on", "comes off"],
        "Offside"     : ["offside"],
    }
    keys       = OUTCOMES.get(human["event_type"], [])
    outcome_ok = any(k in t for k in keys)
    return {
        "player" : player_ok,
        "team"   : team_ok,
        "outcome": outcome_ok,
        "score"  : int(player_ok) + int(team_ok) + int(outcome_ok),
    }


# ── Metric C — BERTScore ──────────────────────────────────────────────────

def metric_c_bert(refs, hyps):
    try:
        from bert_score import score as bscore
        _, _, F1 = bscore(hyps, refs, lang="en", verbose=False,
                          model_type="distilbert-base-uncased")
        return [round(f.item(), 3) for f in F1]
    except ImportError:
        print("WARNING: bert-score not installed. Run: pip install bert-score")
        return [None] * len(refs)


# ── Parse ESPN CSV ────────────────────────────────────────────────────────

def parse_espn_csv(path: str) -> list[dict]:
    """
    Real CSV columns: Time, Player, Team, Action_Type,
                      Yellow_Card, Red_Card, Full_Text
    Time is a float in minutes (e.g. 63.5 = 63rd minute).
    """
    import csv as _csv
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
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
                "event_type" : action,
                "player"     : row.get("Player", "").strip(),
                "team"       : row.get("Team",   "").strip(),
                "description": row.get("Full_Text", "").strip(),
            })
    return sorted(rows, key=lambda r: r["minute"])


# ── Parse JSON commentary files (GT or AI) ────────────────────────────────

def parse_json_commentary(path: str | Path) -> list[dict]:
    """
    Load ground_truth_commentary.json or ai_commentary.json.
    Expected fields: minute, half, event_type, player, team, human_text
    Falls back to 'text' or 'commentary' if 'human_text' is absent.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    events = []
    for e in data:
        text = (e.get("human_text") or e.get("text") or
                e.get("commentary") or e.get("full_text") or "")
        events.append({
            "minute"    : float(e.get("minute", 0)),
            "half"      : int(e.get("half", 1)),
            "event_type": str(e.get("event_type", "")),
            "player"    : str(e.get("player", "")),
            "team"      : str(e.get("team", "")),
            "human_text": text,
        })
    return sorted(events, key=lambda e: e["minute"] + (45 if e["half"] == 2 else 0))


# ── Multi-match event matching ─────────────────────────────────────────────

def match_events(
    gt_events : list[dict],
    ai_events : list[dict],
    tolerance_min: float = 1.0,
) -> tuple[list[tuple], list[dict], list[dict]]:
    """
    Greedy 1-to-1 matching: for each GT event find the closest AI event
    of the same type within tolerance_min absolute minutes.
    Returns:
        matched      — list of (gt_event, ai_event) pairs
        unmatched_gt — GT events with no AI counterpart
        unmatched_ai — AI events that never matched any GT event
    """
    def abs_min(e):
        return float(e["minute"]) + (45.0 if int(e["half"]) == 2 else 0.0)

    ai_pool = list(ai_events)   # consume from this
    matched, unmatched_gt = [], []

    for gt in gt_events:
        gt_abs = abs_min(gt)
        candidates = [
            (abs(abs_min(a) - gt_abs), a)
            for a in ai_pool
            if a["event_type"].lower() == gt["event_type"].lower()
            and abs(abs_min(a) - gt_abs) <= tolerance_min
        ]
        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, best_ai = candidates[0]
            matched.append((gt, best_ai))
            ai_pool.remove(best_ai)
        else:
            unmatched_gt.append(gt)

    return matched, unmatched_gt, ai_pool   # ai_pool remainder = unmatched AI


# ── Per-match evaluation (JSON mode) ──────────────────────────────────────

def evaluate_match_json(
    gt_events: list[dict],
    ai_events: list[dict],
    match_name: str = "",
    tolerance_min: float = 1.0,
    verbose: bool = True,
) -> dict:
    """
    Run all metrics on one match worth of GT vs AI JSON commentary.
    Returns a summary dict suitable for the aggregate table.
    """
    matched_pairs, unmatched_gt, unmatched_ai = match_events(
        gt_events, ai_events, tolerance_min)

    n_gt      = len(gt_events)
    n_ai      = len(ai_events)
    n_matched = len(matched_pairs)

    precision  = n_matched / n_ai      if n_ai   > 0 else 0.0
    recall     = n_matched / n_gt      if n_gt   > 0 else 0.0
    f1         = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)

    gt_texts   = [gt["human_text"] for gt, _ in matched_pairs]
    ai_texts   = [ai["human_text"] for _, ai in matched_pairs]

    avg_bleu   = (round(sum(metric_a_bleu(g, a) for g, a in zip(gt_texts, ai_texts))
                        / n_matched, 3)
                  if n_matched else 0.0)

    bert_scores = metric_c_bert(gt_texts, ai_texts) if n_matched else []
    valid_bert  = [s for s in bert_scores if s is not None]
    avg_bert    = round(sum(valid_bert) / len(valid_bert), 3) if valid_bert else None

    crr_gt  = crr([gt["human_text"] for gt in gt_events])
    crr_ai  = crr([ai["human_text"] for ai in ai_events])

    corp_bleu     = corpus_bleu_score(gt_texts, ai_texts)
    coverage_rate = recall
    fact_rate_raw = 0.0
    if matched_pairs:
        fact_scores = [
            metric_b_fact(gt, ai["human_text"])["score"] / 3
            for gt, ai in matched_pairs
        ]
        fact_rate_raw = sum(fact_scores) / len(fact_scores)

    bert_val  = avg_bert if avg_bert is not None else 0.0
    crr_ratio = (crr_ai / crr_gt / 100) if crr_gt > 0 else 0.0
    mos       = match_overall_score(coverage_rate, fact_rate_raw, bert_val, crr_ratio)

    if verbose:
        W = 70
        def bar(v, max_val=1.0, width=20):
            if v is None:
                return "not computed"
            filled = int((v / max_val) * width)
            return f"[{'█'*filled}{'░'*(width-filled)}] {v:.3f}"

        print(f"\n{'='*W}")
        print(f"  MATCH: {match_name}")
        print(f"  GT events: {n_gt}  |  AI events: {n_ai}  |  Matched: {n_matched}")
        print(f"{'─'*W}")
        print(f"  Precision : {precision:.3f}   Recall : {recall:.3f}   F1 : {f1:.3f}")
        print(f"  BLEU avg  : {bar(avg_bleu)}")
        print(f"  BERTScore : {bar(avg_bert)}")
        print(f"  CRR  GT   : {crr_gt:.1f}%     CRR AI : {crr_ai:.1f}%")
        print(f"{'─'*W}")

        mos_b = int(mos)
        print(f"  Corpus BLEU   : {bar(corp_bleu)}")
        print(f"  Match Overall : [{'█'*mos_b}{'░'*(10-mos_b)}] {mos}/10")
        print(f"    Coverage {round(coverage_rate*100,1)}%  |  "
              f"Semantic {round(bert_val*100,1)}%  |  "
              f"Factual {round(fact_rate_raw*100,1)}%  |  "
              f"Context {round(crr_ratio*100,1)}%")

        if unmatched_gt:
            print(f"\n  Unmatched GT ({len(unmatched_gt)}):")
            for e in unmatched_gt[:5]:
                print(f"    ✗ {int(e['minute'])}'  {e['event_type']}  {e.get('player','?')}")
            if len(unmatched_gt) > 5:
                print(f"    ... and {len(unmatched_gt)-5} more")

    return {
        "match"    : match_name,
        "n_gt"     : n_gt,
        "n_ai"     : n_ai,
        "matched"  : n_matched,
        "precision": round(precision, 3),
        "recall"   : round(recall, 3),
        "f1"       : round(f1, 3),
        "bleu"     : avg_bleu,
        "bert"     : avg_bert,
        "crr_gt"   : crr_gt,
        "crr_ai"   : crr_ai,
        "corp_bleu": corp_bleu,
        "mos"      : mos,
    }


# ── Full ESPN coverage analysis ────────────────────────────────────────────

def full_coverage_analysis(espn_events, ai_events, tol_min=1.5):
    """
    For every ESPN event, check if AI has a matching commentary
    within tol_min minutes and same event_type.
    Returns per-type breakdown + overall stats.
    """
    TYPE_MAP = {
        "Shot"        : "Shot",
        "Goal"        : "Goal",
        "Corner"      : "Corner",
        "Foul"        : "Foul",
        "Free_Kick"   : "Free_Kick",
        "Substitution": "Substitution",
        "Offside"     : "Offside",
        "shot"        : "Shot",
        "goal"        : "Goal",
        "corner"      : "Corner",
        "foul"        : "Foul",
        "free_kick"   : "Free_Kick",
        "substitution": "Substitution",
        "offside"     : "Offside",
    }

    type_stats = {}

    for espn in espn_events:
        raw_type = espn.get("event_type", "").lower()
        ai_type  = TYPE_MAP.get(raw_type)
        if not ai_type:
            continue

        if ai_type not in type_stats:
            type_stats[ai_type] = {"espn": 0, "matched": 0, "missed": []}

        type_stats[ai_type]["espn"] += 1

        espn_min  = float(espn.get("minute", 0))
        espn_half = int(espn.get("half", 1))
        hit = any(
            e["event_type"] == ai_type
            and abs((e["minute"] + (45 if e["half"] == 2 else 0)) - espn_min) <= tol_min
            for e in ai_events
        )

        if hit:
            type_stats[ai_type]["matched"] += 1
        else:
            type_stats[ai_type]["missed"].append(
                f"{espn_half}H {espn_min:.0f}'  {espn.get('player','?')} ({espn.get('team','?')})"
            )

    return type_stats


# ── Corpus BLEU (match-level document similarity) ─────────────────────────

def corpus_bleu_score(human_texts: list[str], ai_texts: list[str]) -> float:
    """
    NLTK corpus_bleu: human sentences are references, AI are hypotheses.
    Each human sentence is one reference set for the corresponding AI sentence.
    Measures overall linguistic similarity across the whole match, not per event.
    """
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    if not human_texts or not ai_texts:
        return 0.0
    refs  = [[t.lower().split()] for t in human_texts]
    hyps  = [t.lower().split()   for t in ai_texts]
    return round(
        corpus_bleu(refs, hyps,
                    weights=(0.5, 0.5),
                    smoothing_function=SmoothingFunction().method1),
        3,
    )


# ── Match Overall Score (MOS) ──────────────────────────────────────────────

def match_overall_score(
    coverage_rate : float,   # matched / total  (0–1)
    fact_rate     : float,   # avg fact score / 3  (0–1)
    bert_avg      : float,   # avg BERTScore  (0–1)
    crr_ratio     : float,   # ai_crr / human_crr  (0–1, capped at 1)
) -> float:
    """
    Weighted 0–10 composite:
      40% coverage  — did the AI notice the events at all?
      30% semantics — does the sentence mean the same thing? (BERTScore)
      20% factual   — right player / team / outcome?
      10% context   — does it reference past events like a human would?
    """
    crr_ratio = min(crr_ratio, 1.0)
    raw = (0.40 * coverage_rate +
           0.30 * bert_avg      +
           0.20 * fact_rate     +
           0.10 * crr_ratio)
    return round(raw * 10, 2)


# ── CRR ───────────────────────────────────────────────────────────────────

def crr(texts):
    if not texts:
        return 0.0
    hits = sum(1 for t in texts
               if any(k in t.lower() for k in PAST_REF_KEYWORDS))
    return round(hits / len(texts) * 100, 1)


# ── Shared report builder ──────────────────────────────────────────────────

def _build_legacy_report(results, ai_events, espn_events, match_title="") -> tuple[str, dict]:
    """
    Build the existing per-event report from the legacy (text-log) workflow.
    Returns (report_string, summary_dict).
    """
    W = 70

    def bar(value, max_val=1.0, width=20):
        if value is None:
            return "not computed"
        filled = int((value / max_val) * width)
        return f"[{'█'*filled}{'░'*(width-filled)}] {value}"

    out = []
    title = match_title or "HUMAN vs AI COMMENTARY EVALUATION"
    out.append("=" * W)
    out.append(f"  {title}")
    out.append("=" * W)

    matched_count = sum(1 for r in results if r["matched"])
    missed_count  = len(results) - matched_count

    out.append(f"\n{'─'*W}")
    out.append("  EVENT-BY-EVENT COMPARISON")
    out.append(f"{'─'*W}")

    for r in results:
        out.append(f"\n┌─ {r['label']}")
        out.append(f"│  HUMAN : {r['human_text']}")
        if r["matched"]:
            out.append(f"│  AI    : {r['ai_text']}")
            p = "✓" if r["b_fact"]["player"]  else "✗"
            t = "✓" if r["b_fact"]["team"]    else "✗"
            o = "✓" if r["b_fact"]["outcome"] else "✗"
            c = "✓" if r["has_context"]       else "✗"
            out.append("│")
            out.append(f"│  [A] BLEU       : {bar(r['a_bleu'])}")
            out.append(f"│  [B] Fact       : {r['b_fact']['score']}/3  "
                       f"Player:{p}  Team:{t}  Outcome:{o}")
            out.append(f"│  [C] BERTScore  : {bar(r['c_bert'])}")
            out.append(f"│  [+] Uses past context: {c}")
        else:
            out.append("│  AI    : ✗  NO MATCH FOUND (AI missed this event)")
        out.append(f"└{'─'*(W-2)}")

    out.append(f"\n{'─'*W}")
    out.append("  MISMATCH SUMMARY")
    out.append(f"{'─'*W}")
    out.append(f"  Total human moments sampled : {len(results)}")
    out.append(f"  AI matched                  : {matched_count}  ✓")
    out.append(f"  AI missed entirely          : {missed_count}   ✗")

    wrong_player  = sum(1 for r in results if r["matched"] and not r["b_fact"]["player"])
    wrong_team    = sum(1 for r in results if r["matched"] and not r["b_fact"]["team"])
    wrong_outcome = sum(1 for r in results if r["matched"] and not r["b_fact"]["outcome"])
    out.append("\n  Of matched events:")
    out.append(f"    Wrong player  : {wrong_player}/{matched_count}")
    out.append(f"    Wrong team    : {wrong_team}/{matched_count}")
    out.append(f"    Wrong outcome : {wrong_outcome}/{matched_count}")

    if espn_events:
        cov = full_coverage_analysis(espn_events, ai_events)
        out.append(f"\n{'─'*W}")
        out.append("  FULL ESPN EVENT COVERAGE  (all events, not just sampled)")
        out.append(f"{'─'*W}")
        out.append(f"  {'Action':<16} {'ESPN':>6} {'AI matched':>10} {'Missed':>8}  Coverage")
        out.append(f"  {'─'*14} {'─'*6} {'─'*10} {'─'*8}  {'─'*24}")

        total_espn = total_matched = 0
        for atype in ["Goal", "Shot", "Corner", "Foul", "Free_Kick", "Substitution", "Offside"]:
            s   = cov.get(atype, {"espn": 0, "matched": 0, "missed": []})
            pct = round(s["matched"] / s["espn"] * 100) if s["espn"] > 0 else 0
            bar_str = f"{'█' * int(pct/5)}{'░' * (20 - int(pct/5))}"
            out.append(f"  {atype:<16} {s['espn']:>6} {s['matched']:>10} "
                       f"{s['espn']-s['matched']:>8}  {pct:>3}%  {bar_str}")
            total_espn    += s["espn"]
            total_matched += s["matched"]

        overall_pct = round(total_matched / total_espn * 100) if total_espn else 0
        out.append(f"  {'─'*14} {'─'*6} {'─'*10} {'─'*8}")
        out.append(f"  {'TOTAL':<16} {total_espn:>6} {total_matched:>10} "
                   f"{total_espn-total_matched:>8}  {overall_pct}% overall coverage")
        out.append("\n  Missed events by type:")
        for atype, s in cov.items():
            if s["missed"]:
                out.append(f"  {atype}:")
                for m in s["missed"][:5]:
                    out.append(f"    ✗  {m}")
                if len(s["missed"]) > 5:
                    out.append(f"    ... and {len(s['missed'])-5} more")

    matched    = [r for r in results if r["matched"]]
    avg_bleu   = round(sum(r["a_bleu"] for r in matched) / len(matched), 3) if matched else 0
    avg_fact   = round(sum(r["b_fact"]["score"] for r in matched) / len(matched), 2) if matched else 0
    valid_bert = [r["c_bert"] for r in matched if r["c_bert"] is not None]
    avg_bert   = round(sum(valid_bert) / len(valid_bert), 3) if valid_bert else None

    ai_crr    = crr([r["ai_text"]    for r in results if r["matched"]])
    human_crr = crr([r["human_text"] for r in results])

    matched_human_texts = [r["human_text"] for r in results if r["matched"]]
    matched_ai_texts    = [r["ai_text"]    for r in results if r["matched"]]
    corp_bleu     = corpus_bleu_score(matched_human_texts, matched_ai_texts)
    coverage_rate = matched_count / len(results) if results else 0.0
    fact_rate     = (avg_fact / 3) if matched else 0.0
    bert_val      = avg_bert if avg_bert is not None else 0.0
    crr_ratio     = (ai_crr / human_crr / 100) if human_crr > 0 else 0.0
    mos           = match_overall_score(coverage_rate, fact_rate, bert_val, crr_ratio)

    out.append(f"\n{'─'*W}")
    out.append("  AGGREGATE SCORES  (matched events only)")
    out.append(f"{'─'*W}")
    out.append(f"  [A] BLEU avg      : {bar(avg_bleu)}")
    out.append(f"  [B] Fact avg      : {avg_fact}/3  ({round(avg_fact / 3 * 100, 1)}%)")
    out.append(f"  [C] BERTScore avg : {bar(avg_bert)}")

    out.append(f"\n{'─'*W}")
    out.append("  CONTEXTUAL REFERENCE RATE (CRR)")
    out.append(f"{'─'*W}")
    out.append("  ESPN baseline     :  0.0%  (no history, by design)")
    out.append(f"  Human commentary  : {human_crr:>5}%")
    out.append(f"  AI commentary     : {ai_crr:>5}%")
    out.append(f"  KG contribution   : +{ai_crr}% over ESPN baseline")
    if human_crr > 0:
        out.append(f"  AI reaches        : {round(ai_crr / human_crr * 100, 1)}% of human CRR")

    # ── Match-level scores (above CONCLUSION) ────────────────────────────
    mos_bar_filled = int(mos)
    mos_bar = f"[{'█'*mos_bar_filled}{'░'*(10-mos_bar_filled)}] {mos}/10"

    out.append(f"\n{'─'*W}")
    out.append("  MATCH-LEVEL SCORES")
    out.append(f"{'─'*W}")
    out.append(f"  Corpus BLEU   : {bar(corp_bleu)}")
    out.append(f"    └─ Linguistic similarity across all matched event pairs.")
    out.append(f"  Match Overall : {mos_bar}")
    out.append(f"    └─ 40% coverage  ({round(coverage_rate*100,1)}%)  "
               f"30% semantic ({round(bert_val*100,1)}%)  "
               f"20% factual ({round(fact_rate*100,1)}%)  "
               f"10% context ({round(crr_ratio*100,1)}%)")

    out.append(f"\n{'='*W}")
    out.append("  CONCLUSION")
    out.append(f"{'='*W}")
    out.append(f"  Coverage  : AI detected {matched_count}/{len(results)} sampled moments "
               f"({round(matched_count / len(results) * 100)}%)")
    out.append(f"  Accuracy  : {round(avg_fact / 3 * 100, 1)}% factual correctness on detected events")
    out.append(f"  Context   : KG raises CRR from 0% → {ai_crr}% (human benchmark: {human_crr}%)")
    if avg_bert is not None:
        quality = "partial match" if avg_bert < 0.65 else "good match"
        out.append(f"  Semantics : BERTScore {avg_bert} — {quality} with human phrasing")
    out.append("")

    summary = {
        "match"    : title,
        "matched"  : matched_count,
        "total"    : len(results),
        "precision": round(matched_count / len(results), 3) if results else 0,
        "recall"   : round(matched_count / len(results), 3) if results else 0,
        "f1"       : 0.0,
        "bleu"     : avg_bleu,
        "bert"     : avg_bert,
        "crr_gt"   : human_crr,
        "crr_ai"   : ai_crr,
        "corp_bleu": corp_bleu,
        "mos"      : mos,
    }
    return "\n".join(out), summary


# ── Aggregate summary table ────────────────────────────────────────────────

def _print_aggregate_table(summaries: list[dict], out_path: Path):
    W   = 100
    COL = 28
    lines = []
    lines.append(f"\n{'='*W}")
    lines.append("  AGGREGATE SUMMARY — ALL MATCHES")
    lines.append(f"{'='*W}")
    hdr = (f"  {'Match':<{COL}} {'P':>5} {'R':>5} {'F1':>5} "
           f"{'BLEU':>6} {'BERT':>6} {'CRR_gt':>7} {'CRR_ai':>7} {'MOS':>5}")
    lines.append(hdr)
    lines.append(f"  {'─'*COL} {'─'*5} {'─'*5} {'─'*5} "
                 f"{'─'*6} {'─'*6} {'─'*7} {'─'*7} {'─'*5}")

    for s in summaries:
        name = s["match"][:COL]
        bert = f"{s['bert']:.3f}" if s["bert"] is not None else "  N/A"
        lines.append(
            f"  {name:<{COL}} "
            f"{s['precision']:>5.3f} {s['recall']:>5.3f} {s['f1']:>5.3f} "
            f"{s['bleu']:>6.3f} {bert:>6} "
            f"{s['crr_gt']:>6.1f}% {s['crr_ai']:>6.1f}% "
            f"{s['mos']:>5.2f}"
        )

    # averages
    def avg(key):
        vals = [s[key] for s in summaries if s[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    lines.append(f"  {'─'*COL} {'─'*5} {'─'*5} {'─'*5} "
                 f"{'─'*6} {'─'*6} {'─'*7} {'─'*7} {'─'*5}")
    bert_avg_str = f"{avg('bert'):.3f}"
    lines.append(
        f"  {'AVERAGE':<{COL}} "
        f"{avg('precision'):>5.3f} {avg('recall'):>5.3f} {avg('f1'):>5.3f} "
        f"{avg('bleu'):>6.3f} {bert_avg_str:>6} "
        f"{avg('crr_gt'):>6.1f}% {avg('crr_ai'):>6.1f}% "
        f"{avg('mos'):>5.2f}"
    )
    lines.append("")

    table = "\n".join(lines)
    print(table)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table, encoding="utf-8")
    print(f"✓ Aggregate table saved to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate AI commentary against ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # legacy single-match (text log)
    ap.add_argument("--ai-log",     help="AI commentary text log")
    ap.add_argument("--human-json", help="Human/GT commentary JSON")
    ap.add_argument("--espn-csv",   help="ESPN CSV for coverage analysis")
    ap.add_argument("--output",     default="evaluation_report.txt",
                    help="Output path for single-match report")

    # JSON-based single match
    ap.add_argument("--match-dir",  help="Match folder containing GT and AI JSON files")

    # multi-match
    ap.add_argument("--all",        action="store_true",
                    help="Loop every match folder under data/ that has both files")
    ap.add_argument("--gt-file",    default="ground_truth_commentary.json",
                    help="GT filename inside each match folder (default: ground_truth_commentary.json)")
    ap.add_argument("--ai-file",    default="ai_commentary.json",
                    help="AI filename inside each match folder (default: ai_commentary.json)")
    ap.add_argument("--tolerance",  type=float, default=1.0,
                    help="Match tolerance in minutes for JSON mode (default: 1.0)")

    args = ap.parse_args()

    # ── MODE A: --all  ───────────────────────────────────────────────────
    if args.all:
        folders = sorted(
            f for f in DATA_DIR.iterdir()
            if f.is_dir()
            and (f / args.gt_file).exists()
            and (f / args.ai_file).exists()
        )
        if not folders:
            print(f"No match folders found under {DATA_DIR} with both "
                  f"'{args.gt_file}' and '{args.ai_file}'.")
            return

        print(f"Found {len(folders)} match folder(s) with both files.\n")
        summaries = []
        for folder in folders:
            gt_events = parse_json_commentary(folder / args.gt_file)
            ai_events = parse_json_commentary(folder / args.ai_file)
            summary   = evaluate_match_json(
                gt_events, ai_events,
                match_name   = folder.name,
                tolerance_min= args.tolerance,
                verbose      = True,
            )
            summaries.append(summary)

        out_path = OUT_DIR / "evaluation_report_all.txt"
        _print_aggregate_table(summaries, out_path)
        return

    # ── MODE B: --match-dir  (JSON single match) ─────────────────────────
    if args.match_dir:
        folder    = Path(args.match_dir)
        gt_path   = folder / args.gt_file
        ai_path   = folder / args.ai_file
        if not gt_path.exists():
            print(f"GT file not found: {gt_path}")
            return
        if not ai_path.exists():
            print(f"AI file not found: {ai_path}")
            return

        gt_events = parse_json_commentary(gt_path)
        ai_events = parse_json_commentary(ai_path)
        evaluate_match_json(
            gt_events, ai_events,
            match_name   = folder.name,
            tolerance_min= args.tolerance,
            verbose      = True,
        )
        return

    # ── MODE C: legacy --ai-log / --human-json  ──────────────────────────
    if not args.ai_log or not args.human_json:
        ap.print_help()
        return

    ai_events    = parse_ai_log(args.ai_log)
    human_events = json.loads(Path(args.human_json).read_text())
    espn_events  = parse_espn_csv(args.espn_csv) if args.espn_csv else []

    results                  = []
    bert_refs, bert_hyps, bert_idx = [], [], []

    for h in human_events:
        ai      = find_match(h, ai_events)
        ai_text = ai["full_text"] if ai else ""

        bleu = metric_a_bleu(h["human_text"], ai_text)
        fact = metric_b_fact(h, ai_text)
        has_context = (any(k in ai_text.lower() for k in PAST_REF_KEYWORDS)
                       if ai_text else False)

        if ai_text:
            bert_refs.append(h["human_text"])
            bert_hyps.append(ai_text)
            bert_idx.append(len(results))

        results.append({
            "label"      : (f"{h['half']}H {h['minute']:02d}'  "
                            f"{h['event_type']:<14} {h.get('player','')}"),
            "human_text" : h["human_text"],
            "ai_text"    : ai_text,
            "matched"    : ai is not None,
            "a_bleu"     : bleu,
            "b_fact"     : fact,
            "c_bert"     : None,
            "has_context": has_context,
        })

    bert_scores = metric_c_bert(bert_refs, bert_hyps)
    for i, score in zip(bert_idx, bert_scores):
        results[i]["c_bert"] = score

    report, _ = _build_legacy_report(
        results, ai_events, espn_events,
        match_title="HUMAN vs AI COMMENTARY EVALUATION",
    )
    print(report)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report)
    print(f"\n✓ Saved to {args.output}")


if __name__ == "__main__":
    main()
