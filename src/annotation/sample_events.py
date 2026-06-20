"""
sample_events.py — Randomly sample events from KG for annotation.
50 per match, stratified by event type.
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

# Stratification ratios (matches research norms)
STRATIFICATION = {
    "Shot":          0.35,   # 35% shots
    "Goal":          0.10,   # 10% goals (rare but high-value)
    "Foul":          0.15,
    "YellowCard":    0.05,
    "Corner":        0.15,
    "Free_Kick":     0.10,
    "Substitution":  0.05,
    "Offside":       0.05,
}

def sample_events_per_match(match_dir: Path, n: int = 50, seed: int = 42) -> list:
    """Sample n events from a match, stratified by event type."""
    ai_file = match_dir / "ai_commentary.json"
    if not ai_file.exists():
        print(f"  [skip] no ai_commentary.json in {match_dir.name}")
        return []
    
    events = json.load(open(ai_file))
    by_type = defaultdict(list)
    for ev in events:
        by_type[ev["event_type"]].append(ev)
    
    sampled = []
    rng = random.Random(seed)
    for event_type, ratio in STRATIFICATION.items():
        bucket = by_type.get(event_type, [])
        n_to_sample = max(1, int(n * ratio))
        n_to_sample = min(n_to_sample, len(bucket))
        sampled.extend(rng.sample(bucket, n_to_sample))
    
    # Trim or pad to exactly n
    rng.shuffle(sampled)
    return sampled[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="+", required=True,
                    help="List of match folder names to sample from")
    ap.add_argument("--n-per-match", type=int, default=50)
    ap.add_argument("--out", default="data/annotation/events_to_rate.json")
    args = ap.parse_args()
    
    DATA = Path("data")
    OUT  = Path(args.out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    
    all_events = []
    rng = random.Random(42)
    
    for match_name in args.matches:
        match_dir = DATA / match_name
        if not match_dir.exists():
            print(f"  [skip] {match_dir} not found")
            continue
        sampled = sample_events_per_match(match_dir, args.n_per_match)
        for ev in sampled:
            ev["_match"] = match_name        # hidden from rater UI
            ev["_match_dir"] = str(match_dir)
        all_events.extend(sampled)
        print(f"  [ok] {match_name}: {len(sampled)} events sampled")
    
    # Assign rating_id and shuffle so events from different matches are interleaved
    rng.shuffle(all_events)
    for i, ev in enumerate(all_events):
        ev["rating_id"] = f"r_{i+1:04d}"
    
    json.dump(all_events, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"\n  Saved {len(all_events)} events to {OUT}")
    print(f"  Each rater will see all {len(all_events)} in this shuffled order.")


if __name__ == "__main__":
    main()
