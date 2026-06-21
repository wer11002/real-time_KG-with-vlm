"""
experiments/level1_baseline.py — Level 1: Baseline (event-anchored, no extras)

Runs the standard commentator.py with fix 047 settings:
  - SYSTEM_PROMPT: JAIST SN-Long 5-example prompt
  - No chain-of-thought prefix
  - No player history injection
  - No tools passed to LLM

Usage:
    # All matches
    python experiments/level1_baseline.py

    # Single match (smoke test)
    python experiments/level1_baseline.py --match "Blackburn"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.runner import aggregate, run_experiment

NAME = "baseline"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", default=None,
                    help="Partial match folder name (run one match only)")
    args = ap.parse_args()

    print(f"═══ Experiment: {NAME} ═══")
    results = run_experiment(
        name=NAME,
        match_filter=args.match,
        extra_main_args=[],
    )

    if not results:
        print("No results — check that human_commentary.json exists in data/sn_long/*/")
        sys.exit(1)

    agg = aggregate(results)
    print(f"\n── Aggregate ({agg['n_matches']} matches) ──")
    print(f"  BLEU-4   : {agg.get('bleu_4', 0):.3f}")
    print(f"  METEOR   : {agg.get('meteor', 0):.3f}")
    print(f"  ROUGE-L  : {agg.get('rouge_l', 0):.3f}")
    print(f"  CIDEr    : {agg.get('cider', 0):.3f}")
    print(f"  BERTScore: {agg.get('bertscore', 0):.3f}")
    print(f"  CRR      : {agg.get('crr', 0):.1f}%")
    print(f"  F1       : {agg.get('f1', 0):.3f}")


if __name__ == "__main__":
    main()
