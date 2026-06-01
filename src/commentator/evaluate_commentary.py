"""
Commentary Evaluation — Human vs AI Agent
Outputs a labeled side-by-side report with scores A, B, C per event
and a summary table of mismatches.

Usage:
    python src/commentator/evaluate_commentary.py \
        --ai-log  data/commentator_output/commentary_log.txt \
        --human-json data/human_commentary.json \
        --output  data/commentator_output/evaluation_report.txt
"""

import argparse
import json
import re
from pathlib import Path

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
        if not text.strip().isascii():
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
    same = [
        e for e in ai_events
        if e["half"] == human["half"]
        and e["event_type"].lower() == human["event_type"].lower()
        and abs(e["minute"] - human["minute"]) <= tol
    ]
    return min(same, key=lambda e: abs(e["minute"] - human["minute"])) if same else None


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


# ── CRR ───────────────────────────────────────────────────────────────────

def crr(texts):
    if not texts:
        return 0.0
    hits = sum(1 for t in texts
               if any(k in t.lower() for k in PAST_REF_KEYWORDS))
    return round(hits / len(texts) * 100, 1)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai-log",     required=True)
    ap.add_argument("--human-json", required=True)
    ap.add_argument("--output",     default="evaluation_report.txt")
    args = ap.parse_args()

    ai_events    = parse_ai_log(args.ai_log)
    human_events = json.loads(Path(args.human_json).read_text())

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
            "label"      : f"{h['half']}H {h['minute']:02d}'  {h['event_type']:<14} {h['player']}",
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

    # ── report ────────────────────────────────────────────────────────────
    W = 70

    def bar(value, max_val=1.0, width=20):
        if value is None:
            return "not computed"
        filled = int((value / max_val) * width)
        return f"[{'█'*filled}{'░'*(width-filled)}] {value}"

    out = []

    out.append("=" * W)
    out.append("  HUMAN vs AI COMMENTARY EVALUATION")
    out.append("  Blackburn Rovers 1-1 Nottingham Forest | Oct 1 2019")
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

    # ── mismatch summary ──────────────────────────────────────────────────
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

    # ── aggregate scores ──────────────────────────────────────────────────
    matched    = [r for r in results if r["matched"]]
    avg_bleu   = round(sum(r["a_bleu"] for r in matched) / len(matched), 3) if matched else 0
    avg_fact   = round(sum(r["b_fact"]["score"] for r in matched) / len(matched), 2) if matched else 0
    valid_bert = [r["c_bert"] for r in matched if r["c_bert"] is not None]
    avg_bert   = round(sum(valid_bert) / len(valid_bert), 3) if valid_bert else None

    ai_crr    = crr([r["ai_text"]    for r in results if r["matched"]])
    human_crr = crr([r["human_text"] for r in results])

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

    report = "\n".join(out)
    print(report)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report)
    print(f"\n✓ Saved to {args.output}")


if __name__ == "__main__":
    main()
