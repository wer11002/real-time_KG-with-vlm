"""
Commentary Evaluation Script
Compares AI commentary against ESPN ground truth and human samples.

Metrics:
  A. BLEU score          — word-level overlap with human text
  B. Fact Overlap score  — player / team / outcome match (0-3 per event)
  C. BERTScore           — semantic similarity to human text
  + Pinpoint match %     — did AI comment at the same moment as human?
  + CRR                  — contextual reference rate vs ESPN baseline

Usage:
  python evaluate_commentary.py \
      --ai-log  data/commentator_output/commentary_log.txt \
      --espn-csv data/blackburn_forest_2019-10-01.csv \
      --human-json data/human_commentary.json \
      --output data/commentator_output/evaluation_report.txt
"""

import argparse
import csv
import json
import re
from pathlib import Path


# ── Step 1: Parse AI commentary log ────────────────────────────────────────

def parse_ai_log(path: str) -> list[dict]:
    """
    Returns list of:
      { half, minute, event_type, full_text }
    Skips lines that contain non-ASCII characters (Chinese text bug).
    """
    events = []
    pattern = re.compile(r"\[(\d+)(?:st|nd)\s+(\d+):(\d+)\]\s+(\w+)\s+\|(.+)")
    for line in Path(path).read_text().splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        half, mins, secs, etype, text = m.groups()
        if not text.isascii():
            continue
        events.append({
            "half"      : int(half),
            "minute"    : int(mins),
            "event_type": etype.strip(),
            "full_text" : text.strip(),
        })
    return events


# ── Step 2: Parse ESPN CSV ──────────────────────────────────────────────────

def parse_espn_csv(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "half"       : int(row.get("half", 1)),
                "minute"     : float(row.get("minute", 0)),
                "event_type" : row.get("event_type", "").strip(),
                "player"     : row.get("player", "").strip(),
                "team"       : row.get("team", "").strip(),
                "description": row.get("description", "").strip(),
            })
    return rows


# ── Step 3: Match AI event to human sample (±2 min, same event type) ───────

def find_ai_match(human_event: dict, ai_events: list[dict],
                  tol_min: float = 2.0) -> dict | None:
    same_type = [
        e for e in ai_events
        if e["event_type"].lower() == human_event["event_type"].lower()
        and e["half"] == human_event["half"]
        and abs(e["minute"] - human_event["minute"]) <= tol_min
    ]
    if not same_type:
        return None
    return min(same_type, key=lambda e: abs(e["minute"] - human_event["minute"]))


# ── Step 4: Metric A — BLEU score ──────────────────────────────────────────

def bleu(reference: str, hypothesis: str) -> float:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    smoothie   = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens,
                         weights=(0.5, 0.5),
                         smoothing_function=smoothie)


# ── Step 5: Metric B — Fact Overlap score (0–3) ────────────────────────────

def fact_overlap(human: dict, ai_text: str) -> dict:
    from thefuzz import fuzz
    text_lower   = ai_text.lower()
    player_parts = human["player"].lower().split()
    team_parts   = human["team"].lower().split()

    player_hit = any(p in text_lower for p in player_parts)
    team_hit   = any(fuzz.partial_ratio(t, text_lower) > 75 for t in team_parts)

    OUTCOME_KEYWORDS = {
        "Goal"        : ["goal", "scores", "scored", "puts away", "fires home",
                         "slots", "nets", "buries"],
        "Shot"        : ["saved", "blocked", "missed", "wide", "over",
                         "shot", "attempt"],
        "Foul"        : ["foul", "tackle", "card", "challenge", "brings down"],
        "Corner"      : ["corner"],
        "Free_Kick"   : ["free kick", "freekick"],
        "Substitution": ["substitut", "replac", "comes on", "comes off"],
        "Offside"     : ["offside"],
    }
    keywords    = OUTCOME_KEYWORDS.get(human["event_type"], [])
    outcome_hit = any(k in text_lower for k in keywords)

    score = int(player_hit) + int(team_hit) + int(outcome_hit)
    return {
        "player" : player_hit,
        "team"   : team_hit,
        "outcome": outcome_hit,
        "score"  : score,
    }


# ── Step 6: Metric C — BERTScore ───────────────────────────────────────────

def bertscore_f1(references: list[str], hypotheses: list[str]) -> list[float]:
    try:
        from bert_score import score as bt_score
        _, _, F1 = bt_score(hypotheses, references,
                            lang="en", verbose=False,
                            model_type="distilbert-base-uncased")
        return F1.tolist()
    except ImportError:
        print("WARNING: bert-score not installed. Run: pip install bert-score")
        return [None] * len(references)


# ── Step 7: Pinpoint match % and CRR ───────────────────────────────────────

PAST_REF_KEYWORDS = [
    "again", "second", "third", "another", "has been",
    "continues", "following", "always", "still", "once more",
    "earlier", "first time", "pressure mounting", "keeps the pressure",
]


def crr(ai_events: list[dict]) -> float:
    if not ai_events:
        return 0.0
    hits = sum(
        1 for e in ai_events
        if any(k in e["full_text"].lower() for k in PAST_REF_KEYWORDS)
    )
    return hits / len(ai_events) * 100


def pinpoint_match_rate(human_events: list[dict],
                        ai_events: list[dict]) -> float:
    matched = sum(
        1 for h in human_events if find_ai_match(h, ai_events) is not None
    )
    return matched / len(human_events) * 100 if human_events else 0.0


# ── Step 8: Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai-log",     required=True)
    parser.add_argument("--espn-csv",   required=True)
    parser.add_argument("--human-json", required=True)
    parser.add_argument("--output",     default="evaluation_report.txt")
    args = parser.parse_args()

    ai_events    = parse_ai_log(args.ai_log)
    espn_events  = parse_espn_csv(args.espn_csv)
    human_events = json.loads(Path(args.human_json).read_text())

    pinpoint  = pinpoint_match_rate(human_events, ai_events)
    ai_crr    = crr(ai_events)
    human_crr = crr([{"full_text": h["human_text"]} for h in human_events])

    rows        = []
    bleu_scores = []
    fact_scores = []
    bert_refs, bert_hyps, bert_indices = [], [], []

    for h in human_events:
        ai_match = find_ai_match(h, ai_events)
        ai_text  = ai_match["full_text"] if ai_match else ""

        b  = bleu(h["human_text"], ai_text) if ai_text else 0.0
        fo = (fact_overlap(h, ai_text) if ai_text
              else {"player": False, "team": False, "outcome": False, "score": 0})

        bleu_scores.append(b)
        fact_scores.append(fo["score"])

        if ai_text and h["human_text"] != "FILL FROM YOUTUBE":
            bert_refs.append(h["human_text"])
            bert_hyps.append(ai_text)
            bert_indices.append(len(rows))

        rows.append({
            "minute"    : f"{h['half']}H {h['minute']}'",
            "event_type": h["event_type"],
            "human_text": h["human_text"][:80],
            "ai_text"   : ai_text[:80] if ai_text else "(no match found)",
            "bleu"      : round(b, 3),
            "fact_score": fo["score"],
            "player_ok" : "✓" if fo["player"]  else "✗",
            "team_ok"   : "✓" if fo["team"]    else "✗",
            "outcome_ok": "✓" if fo["outcome"] else "✗",
            "bert_f1"   : None,
        })

    bert_f1s = bertscore_f1(bert_refs, bert_hyps)
    for idx, f1 in zip(bert_indices, bert_f1s):
        rows[idx]["bert_f1"] = round(f1, 3) if f1 is not None else None

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_fact = sum(fact_scores) / len(fact_scores) if fact_scores else 0.0
    valid_bert = [r["bert_f1"] for r in rows if r["bert_f1"] is not None]
    avg_bert   = sum(valid_bert) / len(valid_bert) if valid_bert else None

    n_matched = sum(1 for h in human_events if find_ai_match(h, ai_events) is not None)

    sep = "=" * 60
    lines = [
        sep,
        "  COMMENTARY EVALUATION REPORT",
        "  Blackburn Rovers 1-1 Nottingham Forest (Oct 1 2019)",
        sep,
        "",
        "PINPOINT MATCH RATE",
        f"  Human moments sampled : {len(human_events)}",
        f"  AI matched within ±2m : {n_matched}",
        f"  Pinpoint match rate   : {pinpoint:.1f}%",
        "",
        "CONTEXTUAL REFERENCE RATE (CRR)",
        "  ESPN baseline CRR     : 0.0%  (no history by design)",
        f"  Human CRR             : {human_crr:.1f}%",
        f"  AI CRR                : {ai_crr:.1f}%",
        f"  KG contribution       : +{ai_crr:.1f}% over ESPN baseline",
        "",
        "METRIC A — BLEU SCORE  (word overlap, 0-1)",
        f"  Average BLEU          : {avg_bleu:.3f}",
        "  Interpretation        : <0.1 low | 0.1-0.3 partial | >0.3 good",
        "",
        "METRIC B — FACT OVERLAP  (player + team + outcome, 0-3 per event)",
        f"  Average fact score    : {avg_fact:.2f} / 3",
        f"  As percentage         : {avg_fact / 3 * 100:.1f}%",
        "",
        "METRIC C — BERTScore F1  (semantic similarity, 0-1)",
        (f"  Average BERTScore F1  : {avg_bert:.3f}"
         if avg_bert is not None
         else "  BERTScore             : not computed (fill human_commentary.json first)"),
        "  Interpretation        : <0.5 low | 0.5-0.7 partial | >0.7 good",
        "",
        sep,
        "  SIDE-BY-SIDE KEY MOMENTS",
        sep,
    ]

    for r in rows:
        lines += [
            f"\n[{r['minute']}] {r['event_type']}",
            f"  Human : {r['human_text']}",
            f"  AI    : {r['ai_text']}",
            f"  BLEU={r['bleu']}  Fact={r['fact_score']}/3 "
            f"(P:{r['player_ok']} T:{r['team_ok']} O:{r['outcome_ok']})  "
            f"BERT={r['bert_f1'] if r['bert_f1'] is not None else 'N/A'}",
        ]

    lines += [
        "",
        sep,
        "  CONCLUSION",
        sep,
        f"  The KG history system raises CRR from 0% (ESPN) to {ai_crr:.1f}%,",
        (f"  achieving {ai_crr / human_crr * 100:.0f}% of human-level contextual richness."
         if human_crr > 0 else ""),
        f"  Pinpoint coverage: {pinpoint:.1f}% of human-sampled moments detected.",
        f"  Factual accuracy on detected events: {avg_fact / 3 * 100:.1f}%.",
        (f"  Semantic similarity (BERTScore): {avg_bert:.3f}."
         if avg_bert is not None else ""),
        "",
    ]

    report = "\n".join(lines)
    print(report)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report)
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
