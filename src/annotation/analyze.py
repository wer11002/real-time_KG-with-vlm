"""analyze.py — Compute inter-rater agreement after collection."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa
from itertools import combinations

ratings_dir = Path("data/annotation/ratings")
files = list(ratings_dir.glob("ratings_*.csv"))
print(f"Found {len(files)} rater files: {[f.stem for f in files]}")

# Load all
dfs = {f.stem.replace("ratings_", ""): pd.read_csv(f) for f in files}

# Merge on rating_id
merged = None
for rater, df in dfs.items():
    df = df[["rating_id", "accuracy", "completeness", "depth"]].copy()
    df.columns = ["rating_id"] + [f"{c}_{rater}" for c in ["acc", "comp", "depth"]]
    merged = df if merged is None else merged.merge(df, on="rating_id")

print(f"\n{len(merged)} events rated by ALL {len(dfs)} annotators")

# Drop skipped events
merged = merged[(merged.filter(regex="acc_") >= 1).all(axis=1)]
print(f"{len(merged)} events after excluding skipped")

# Pairwise Cohen's κ
print("\n=== Pairwise Cohen's κ ===")
raters = list(dfs.keys())
for dim in ["acc", "comp", "depth"]:
    print(f"\n  {dim}:")
    kappas = []
    for r1, r2 in combinations(raters, 2):
        k = cohen_kappa_score(merged[f"{dim}_{r1}"], merged[f"{dim}_{r2}"])
        print(f"    {r1} ↔ {r2}: κ = {k:.3f}")
        kappas.append(k)
    print(f"    MEAN: {np.mean(kappas):.3f} (need ≥ 0.6)")

# Fleiss' κ (all raters at once)
print("\n=== Fleiss' κ (3 raters) ===")
for dim in ["acc", "comp", "depth"]:
    # Build (n_events × 5_scores) matrix
    mat = np.zeros((len(merged), 5))
    for _, row in merged.iterrows():
        for r in raters:
            score = int(row[f"{dim}_{r}"])
            mat[merged.index.get_loc(_)][score - 1] += 1
    k = fleiss_kappa(mat)
    print(f"  {dim}: κ = {k:.3f}")

# Mean ratings per dimension
print("\n=== Mean ratings ===")
for dim in ["acc", "comp", "depth"]:
    cols = [f"{dim}_{r}" for r in raters]
    mean_per_event = merged[cols].mean(axis=1)
    print(f"  {dim}: {mean_per_event.mean():.2f} ± {mean_per_event.std():.2f}")
