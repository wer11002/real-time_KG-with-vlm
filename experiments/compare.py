"""
experiments/compare.py — Compare all experiment levels from results.json

Reads experiments/results.json and prints a side-by-side table across all
three levels (baseline, cot, kg_history) for every metric and match.

Usage:
    python experiments/compare.py
    python experiments/compare.py --metric bleu_4 meteor cider
    python experiments/compare.py --match "Blackburn"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.config import METRICS, RESULTS

LEVEL_ORDER = ["baseline", "cot", "kg_history"]
LEVEL_LABEL = {
    "baseline"   : "L1 Baseline",
    "cot"        : "L2 CoT",
    "kg_history" : "L3 KG-Hist",
}


def _avg(match_results: dict, metric: str) -> float:
    vals = [v.get(metric, 0.0) or 0.0 for v in match_results.values()]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", nargs="+", default=METRICS,
                    help=f"Metrics to show (default: {' '.join(METRICS)})")
    ap.add_argument("--match",  default=None,
                    help="Show only rows containing this substring")
    args = ap.parse_args()

    if not RESULTS.exists():
        print(f"ERROR: {RESULTS} not found — run level1/2/3 first.")
        sys.exit(1)

    all_results: dict = json.loads(RESULTS.read_text())

    present_levels = [l for l in LEVEL_ORDER if l in all_results]
    if not present_levels:
        print("No experiment results found in results.json.")
        sys.exit(1)

    metrics = [m for m in args.metric if m in METRICS + ["f1", "precision", "recall"]]

    # ── per-match table ──────────────────────────────────────────────────────
    # Collect all match names across all levels
    all_matches: set[str] = set()
    for level in present_levels:
        all_matches |= set(all_results[level].keys())
    match_list = sorted(all_matches)
    if args.match:
        match_list = [m for m in match_list if args.match.lower() in m.lower()]

    col_w = 10
    level_cols = [LEVEL_LABEL.get(l, l) for l in present_levels]

    for metric in metrics:
        print(f"\n{'═'*80}")
        print(f"  Metric: {metric}")
        print(f"{'═'*80}")

        header = f"  {'Match':<45}" + "".join(f"{lbl:>{col_w}}" for lbl in level_cols)
        print(header)
        print(f"  {'-'*45}" + "-"*col_w*len(present_levels))

        for match in match_list:
            row = f"  {match[:44]:<45}"
            for level in present_levels:
                val = all_results[level].get(match, {}).get(metric)
                if val is None:
                    row += f"{'—':>{col_w}}"
                else:
                    row += f"{val:>{col_w}.3f}"
            print(row)

        # aggregate row
        print(f"  {'─'*45}" + "─"*col_w*len(present_levels))
        agg_row = f"  {'AVERAGE':<45}"
        for level in present_levels:
            avg = _avg(all_results[level], metric)
            agg_row += f"{avg:>{col_w}.3f}"
        print(agg_row)

    # ── summary table (all metrics × all levels) ─────────────────────────────
    print(f"\n{'═'*80}")
    print("  SUMMARY — macro-average across all matches")
    print(f"{'═'*80}")
    header = f"  {'Metric':<16}" + "".join(f"{lbl:>{col_w}}" for lbl in level_cols)
    print(header)
    print(f"  {'-'*16}" + "-"*col_w*len(present_levels))
    for metric in metrics:
        row = f"  {metric:<16}"
        for level in present_levels:
            avg = _avg(all_results[level], metric)
            row += f"{avg:>{col_w}.3f}"
        print(row)

    # ── best level per metric ────────────────────────────────────────────────
    print(f"\n  {'Best level per metric':}")
    for metric in metrics:
        avgs = {level: _avg(all_results[level], metric) for level in present_levels}
        best = max(avgs, key=lambda l: avgs[l])
        print(f"    {metric:<16}  → {LEVEL_LABEL.get(best, best)}  ({avgs[best]:.3f})")


if __name__ == "__main__":
    main()
