"""
download_sn_long.py — Download ONLY the 47 JAIST SN-Long matches.
"""

import argparse
import shutil
import subprocess
from pathlib import Path
from SoccerNet.Downloader import SoccerNetDownloader

JAIST_ROOT = Path.home() / "work/s2616011/Augmented_Soccer/Dataset/long"
RAW_DIR    = Path("data/sn_long_videos")
OUT_DIR    = Path("data/sn_long")


def discover_matches(leagues_filter=None):
    matches = []
    for league_dir in sorted(JAIST_ROOT.iterdir()):
        if not league_dir.is_dir():
            continue
        if leagues_filter and league_dir.name not in leagues_filter:
            continue
        # league_dir.name = "england_epl_2014-2015"  (e.g.)
        # We want league = "england_epl", season = "2014-2015"
        parts = league_dir.name.rsplit("_", 1)
        league = parts[0]              # "england_epl"
        season = parts[1]              # "2014-2015"
        for match_dir in sorted(league_dir.iterdir()):
            if not match_dir.is_dir():
                continue
            matches.append({
                "league": league,
                "season": season,
                "match_name": match_dir.name,
                "jaist_path": match_dir,
                # SoccerNet path: "england_epl/2014-2015/<match>"
                "rel_path": f"{league}/{season}/{match_dir.name}",
            })
    return matches


def download_match(match, resolution, dl):
    print(f"\n[download] {match['rel_path']}")
    files = [f"1_{resolution}.mkv", f"2_{resolution}.mkv", "Labels-caption.json"]
    try:
        dl.downloadGame(match["rel_path"], files=files)
        return True
    except Exception as e:
        print(f"  [err] {e}")
        return False


def prep_and_cleanup(match, resolution):
    src_dir = RAW_DIR / match["rel_path"]
    out_name = f"{match['season']} - {match['match_name']}"
    out_dir = OUT_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    half1 = src_dir / f"1_{resolution}.mkv"
    half2 = src_dir / f"2_{resolution}.mkv"
    out_video = out_dir / f"{resolution}.mp4"

    if not half1.exists() or not half2.exists():
        print(f"  [skip] missing halves at {src_dir}")
        return False

    if not out_video.exists():
        concat_list = out_dir / "concat.txt"
        concat_list.write_text(f"file '{half1.resolve()}'\nfile '{half2.resolve()}'\n")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "warning",
                 "-f", "concat", "-safe", "0",
                 "-i", str(concat_list), "-c", "copy",
                 str(out_video)],
                check=True,
            )
        except subprocess.CalledProcessError:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "warning",
                 "-f", "concat", "-safe", "0",
                 "-i", str(concat_list),
                 "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-an",
                 str(out_video)],
                check=True,
            )
        concat_list.unlink()
        print(f"  [concat] {out_video}")

    for half in (1, 2):
        src = match["jaist_path"] / f"{half}_long-term.json"
        if src.exists():
            shutil.copy(src, out_dir / f"jaist_gt_{half}.json")
    sn_cap = src_dir / "Labels-caption.json"
    if sn_cap.exists():
        shutil.copy(sn_cap, out_dir / "soccernet_caption.json")

    for f in (half1, half2, sn_cap):
        if f.exists():
            f.unlink()
    for p in [src_dir, src_dir.parent, src_dir.parent.parent]:
        if p.exists() and not any(p.iterdir()):
            p.rmdir()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="+")
    ap.add_argument("--resolution", default="720p", choices=["224p", "720p"])
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--max", type=int)
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    matches = discover_matches(args.leagues)
    if args.max:
        matches = matches[:args.max]

    print(f"Found {len(matches)} JAIST SN-Long matches")
    for m in matches:
        print(f"  - {m['rel_path']}")
    if args.list_only:
        return

    est_gb = len(matches) * (3 if args.resolution == "720p" else 0.5)
    print(f"\nEstimated FINAL disk usage: ~{est_gb:.0f} GB")
    if input("Continue? (y/n) ").lower() != "y":
        return

    dl = SoccerNetDownloader(LocalDirectory=str(RAW_DIR))
    dl.password = "s0cc3rn3t"

    ok, fail = 0, 0
    for i, m in enumerate(matches, 1):
        print(f"\n[{i}/{len(matches)}] {m['match_name']}")
        if download_match(m, args.resolution, dl):
            if prep_and_cleanup(m, args.resolution):
                ok += 1
                continue
        fail += 1

    print(f"\n========================================")
    print(f"  Done. {ok} ready, {fail} failed")
    print(f"========================================")


if __name__ == "__main__":
    main()
