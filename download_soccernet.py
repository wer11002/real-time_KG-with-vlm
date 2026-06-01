"""
download_soccernet.py — Download SoccerNet match videos into data/
Skips matches that already exist in the data/ folder.

Usage:
    # Step 1: peek at dataset structure first (no download)
    python download_soccernet.py --peek

    # Step 2: download everything new
    python download_soccernet.py

    # Step 3: download specific repo
    python download_soccernet.py --repo SoccerNet/SN-ALS-2025
"""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, list_repo_files

DATA_DIR = Path("data")

ALREADY_HAVE = {
    "2019-10-01 - Blackburn Rovers - Nottingham Forest",
    "2019-10-01 - Brentford - Bristol City",
}

def get_existing_matches() -> set[str]:
    if not DATA_DIR.exists():
        return set()
    return {p.name for p in DATA_DIR.iterdir() if p.is_dir()}

def peek(repo_id: str):
    """Print first 30 files in the repo to understand structure."""
    print(f"\nPeeking at {repo_id} structure...\n")
    files = list(list_repo_files(repo_id, repo_type="dataset"))
    for f in files[:30]:
        print(f"  {f}")
    print(f"\n  ... {len(files)} total files")
    print("\nCheck if these are full match videos (720p.mp4) or clips before downloading.")

def download(repo_id: str):
    existing = get_existing_matches()
    skip     = ALREADY_HAVE | existing

    print(f"\nRepo     : {repo_id}")
    print(f"Data dir : {DATA_DIR.resolve()}")
    print(f"Skipping : {len(skip)} already-existing matches")
    for s in sorted(skip):
        print(f"  - {s}")

    ignore = [f"{name}/*" for name in skip]

    print(f"\nDownloading new matches to {DATA_DIR}...\n")
    snapshot_download(
        repo_id         = repo_id,
        repo_type       = "dataset",
        revision        = "main",
        local_dir       = str(DATA_DIR),
        ignore_patterns = ignore,
    )
    print("\nDone.")

    new_matches = get_existing_matches() - existing - ALREADY_HAVE
    print(f"\nNew matches downloaded: {len(new_matches)}")
    for m in sorted(new_matches):
        print(f"  + {m}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo",  default="SoccerNet/SN-BAS-2025",
                    help="HuggingFace dataset repo ID")
    ap.add_argument("--peek",  action="store_true",
                    help="Print repo file structure without downloading")
    args = ap.parse_args()

    if args.peek:
        peek(args.repo)
    else:
        download(args.repo)

if __name__ == "__main__":
    main()
