#!/usr/bin/env python3
"""
download_soccernet.py — Download SoccerNet matches for the EKG pipeline
────────────────────────────────────────────────────────────────────────

Converts SoccerNet's league/season/match directory tree into the flat
per-match folders that main.py expects:

    data/YYYY-MM-DD - HomeTeam - AwayTeam/
        Labels-ball.json   ← renamed from Labels-v2.json
        720p.mp4           ← merged from 1_720p.mkv + 2_720p.mkv
        224p.mp4           ← merged from 1_224p.mkv + 2_224p.mkv

Usage:
    # Labels only — no password needed
    python download_soccernet.py --labels-only

    # Full download (NDA password required for videos)
    python download_soccernet.py --password "your_nda_password"

    # Specific league, limit matches for quick test
    python download_soccernet.py --labels-only --leagues england_epl --max 10

    # Already downloaded raw data — just convert to project format
    python download_soccernet.py --convert-only --raw-dir /path/to/raw

NDA password:
    Register at https://www.soccer-net.org/data, fill in the form,
    and the password arrives by email (check spam).
    You only need it for videos; labels download without any password.

Requires:
    pip install SoccerNet
    ffmpeg  (for video merging — brew install ffmpeg / apt install ffmpeg)
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ── SoccerNet folder name ────────────────────────────────────────────────────
# Format: "YYYY-MM-DD - HH-MM HomeTeam score - score AwayTeam"
# Example: "2014-08-16 - 17-00 Arsenal 2 - 1 Crystal Palace"
_FOLDER_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}) - \d{2}-\d{2} (.+?) \d+ - \d+ (.+)$"
)


def parse_sn_folder(name: str):
    """Return (date, home_team, away_team) or None."""
    m = _FOLDER_RE.match(name.strip())
    if not m:
        return None
    return m.group(1), m.group(2).strip(), m.group(3).strip()


def target_name(date: str, home: str, away: str) -> str:
    return f"{date} - {home} - {away}"


# ── Video merging ────────────────────────────────────────────────────────────

def _check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def merge_halves(src_dir: Path, res: str, out_file: Path) -> bool:
    """
    Merge 1_{res}.mkv + 2_{res}.mkv → out_file using ffmpeg concat.
    Returns True on success.
    """
    h1 = src_dir / f"1_{res}.mkv"
    h2 = src_dir / f"2_{res}.mkv"

    available = [f for f in (h1, h2) if f.exists()]
    if not available:
        return False

    concat_txt = src_dir / f"_concat_{res}.txt"
    try:
        with open(concat_txt, "w") as f:
            for half in available:
                f.write(f"file '{half.resolve()}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c", "copy",
            str(out_file),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(f"    [ffmpeg error] {result.stderr.decode()[-300:]}")
            return False
        return True
    finally:
        concat_txt.unlink(missing_ok=True)


# ── Per-match processing ─────────────────────────────────────────────────────

def process_match(src_dir: Path, data_dir: Path, do_videos: bool, resolutions: list) -> bool:
    """
    Convert one SoccerNet match directory to project format.
    Returns True if anything was written.
    """
    info = parse_sn_folder(src_dir.name)
    if not info:
        print(f"  [skip — unrecognised folder name] {src_dir.name}")
        return False

    date, home, away = info
    dst = data_dir / target_name(date, home, away)
    dst.mkdir(parents=True, exist_ok=True)
    wrote_anything = False

    # Labels — accept Labels-ball.json (ball-action-spotting task) or Labels-v2.json
    labels_dst = dst / "Labels-ball.json"
    labels_src = next(
        (src_dir / f for f in ("Labels-ball.json", "Labels-v2.json") if (src_dir / f).exists()),
        None,
    )
    if labels_src:
        if not labels_dst.exists():
            shutil.copy2(labels_src, labels_dst)
            print(f"  labels  ✓  {dst.name}/Labels-ball.json  (from {labels_src.name})")
        else:
            print(f"  labels  —  already exists, skipped")
        wrote_anything = True
    else:
        print(f"  labels  ✗  no label file found in {src_dir.name}")

    # Videos ─────────────────────────────────────────────────────────────────
    if do_videos:
        for res in resolutions:
            out = dst / f"{res}.mp4"
            if out.exists():
                print(f"  {res}     —  {res}.mp4 already exists, skipped")
                continue
            ok = merge_halves(src_dir, res, out)
            if ok:
                size_mb = out.stat().st_size / 1_048_576
                print(f"  {res}     ✓  {dst.name}/{res}.mp4  ({size_mb:.0f} MB)")
                wrote_anything = True
            else:
                print(f"  {res}     ✗  no halves found for {res}")

    return wrote_anything


# ── Download ─────────────────────────────────────────────────────────────────

def download(raw_dir: Path, password: str, labels_only: bool,
             splits: list, resolutions: list):
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        print("SoccerNet library not installed.\nRun: pip install SoccerNet")
        sys.exit(1)

    dl = SoccerNetDownloader(LocalDirectory=str(raw_dir))
    if password:
        dl.password = password

    print(f"Task   : ball-action-spotting")
    print(f"Splits : {splits}")
    print()

    # Labels (no password needed)
    print("─── Downloading labels ───")
    dl.downloadDataTask(
        task="ball-action-spotting",
        split=splits,
    )

    # Videos (password required)
    if not labels_only:
        for res in resolutions:
            print(f"─── Downloading {res} videos ───")
            dl.downloadDataTask(
                task="ball-action-spotting",
                split=splits,
                version=res,
            )


# ── Walk raw directory ────────────────────────────────────────────────────────

def iter_match_dirs(raw_dir: Path):
    """Yield all leaf match directories (3 levels deep: league/season/match)."""
    for league_dir in sorted(raw_dir.iterdir()):
        if not league_dir.is_dir() or league_dir.name.startswith("."):
            continue
        for season_dir in sorted(league_dir.iterdir()):
            if not season_dir.is_dir():
                continue
            for match_dir in sorted(season_dir.iterdir()):
                if match_dir.is_dir():
                    yield match_dir


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download SoccerNet matches and convert to EKG pipeline format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--password", default="",
        help="NDA password for video downloads (labels don't need it)",
    )
    parser.add_argument(
        "--labels-only", action="store_true",
        help="Download labels only — no password required",
    )
    parser.add_argument(
        "--convert-only", action="store_true",
        help="Skip download; just convert whatever is in --raw-dir",
    )
    parser.add_argument(
        "--leagues", nargs="+",
        default=["england_epl"],
        choices=[
            "england_epl",
            "europe_uefa-champions-league",
            "france_ligue-1",
            "germany_bundesliga",
            "spain_laliga",
        ],
        help="Leagues to download (default: england_epl)",
    )
    parser.add_argument(
        "--splits", nargs="+",
        default=["train", "valid", "test"],
        help="Dataset splits (default: all)",
    )
    parser.add_argument(
        "--resolutions", nargs="+",
        default=["720p", "224p"],
        choices=["720p", "224p"],
        help="Video resolutions to download (default: both)",
    )
    parser.add_argument(
        "--max", type=int, default=None, metavar="N",
        help="Stop after N matches (useful for quick tests)",
    )
    parser.add_argument(
        "--raw-dir", default=str(DATA_DIR / "soccernet_raw"),
        help="Staging directory for raw SoccerNet downloads (deleted after conversion)",
    )
    parser.add_argument(
        "--data-dir", default=str(DATA_DIR),
        help="Output data directory (default: data/)",
    )
    parser.add_argument(
        "--keep-raw", action="store_true",
        help="Keep the raw staging directory after conversion (default: delete it)",
    )

    args = parser.parse_args()

    raw_dir  = Path(args.raw_dir)
    data_dir = Path(args.data_dir)
    do_videos = not args.labels_only and bool(args.password)

    if do_videos and not _check_ffmpeg():
        print("ffmpeg not found — needed for video merging.")
        print("Install: brew install ffmpeg  OR  sudo apt install ffmpeg")
        sys.exit(1)

    # ── download phase ───────────────────────────────────────────────────────
    if not args.convert_only:
        raw_dir.mkdir(parents=True, exist_ok=True)
        download(
            raw_dir      = raw_dir,
            password     = args.password,
            labels_only  = args.labels_only,
            splits       = args.splits,
            resolutions  = args.resolutions,
        )

    # ── convert phase ────────────────────────────────────────────────────────
    print("\n─── Converting to project format ───")
    data_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"Raw directory not found: {raw_dir}")
        print("Run without --convert-only first to download data.")
        sys.exit(1)

    processed = 0
    skipped   = 0

    for match_dir in iter_match_dirs(raw_dir):
        if args.max and processed >= args.max:
            print(f"\nReached --max {args.max}, stopping.")
            break

        league = match_dir.parent.parent.name
        season = match_dir.parent.name
        print(f"\n[{processed + skipped + 1}] {league}/{season}/{match_dir.name}")

        ok = process_match(match_dir, data_dir, do_videos, args.resolutions)
        if ok:
            processed += 1
        else:
            skipped += 1

    # ── cleanup staging dir ──────────────────────────────────────────────────
    if not args.keep_raw and not args.convert_only and raw_dir.exists():
        print(f"\nCleaning up staging directory: {raw_dir}")
        shutil.rmtree(raw_dir)
        print("  Deleted.")

    print(f"\n{'─'*60}")
    print(f"Done. {processed} matches written to {data_dir}")
    if skipped:
        print(f"      {skipped} matches skipped (no labels found)")
    print()
    print("Next steps:")
    print(f"  python main.py --match <name>    # run pipeline on one match")
    print(f"  python main.py                   # run all matches")
    print(f"  python evaluate.py               # evaluate KG vs ground truth")


if __name__ == "__main__":
    main()
