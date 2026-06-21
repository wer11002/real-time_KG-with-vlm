import re
from pathlib import Path

SN_LONG = Path("data/sn_long")
PATTERN = re.compile(
    r"^(\d{4}-\d{4}) - (\d{4}-\d{2}-\d{2}) - \d{2}-\d{2} (.+?) \d+ - \d+ (.+)$"
)

renamed = skipped = 0
for folder in sorted(SN_LONG.iterdir()):
    if not folder.is_dir():
        continue
    m = PATTERN.match(folder.name)
    if not m:
        print(f"SKIP: {folder.name}")
        skipped += 1
        continue
    season, date, team1, team2 = m.groups()
    new_name = f"{date} - {team1.strip()} - {team2.strip()}"
    new_path = folder.parent / new_name
    if new_path.exists():
        print(f"EXISTS: {new_name}")
        skipped += 1
        continue
    print(f"  {folder.name}")
    print(f"-> {new_name}")
    folder.rename(new_path)
    renamed += 1

print(f"\nDone: renamed {renamed}, skipped {skipped}")
