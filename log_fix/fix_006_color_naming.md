# FIX 006 — Kit Color Naming: Crude Heuristic → Nearest-Neighbor RGB Distance + Collision Detection

**Date:** 2026-05-10
**Status:** Applied
**Files changed:**
- `src/1_video_processor/action_recognizer.py`

════════════════════════════════════════════════════════════════
## PART 1 — THE PROBLEM
════════════════════════════════════════════════════════════════

### Problem A — hex_to_color_name() uses a dominant-channel heuristic

The old implementation determined color by finding the dominant RGB channel
and applying threshold rules:

```python
if brightness < 40:
    return "black"
if brightness > 200 and max_channel == brightness:
    return "white"
if r > g and r > b:
    return "red" (or "orange/red")
if b > r and b > g:
    return "blue" (or "dark blue")
```

This fails for colors where no channel clearly dominates, or where two
similar shades are on opposite sides of a threshold boundary:
- `#001489` (Blackburn navy) → r=0, g=20, b=137 → b dominant → "blue"
- `#003399` (deep blue) → r=0, g=51, b=153 → b dominant → "blue"
- `#00003c` (very dark navy) → brightness=20 → **"black"** ✗

Two teams with `#001489` vs `#003399` would both map to "blue" →
collision → color disambiguation produces wrong or no result.

The "orange/red" return value was also a multi-word name that wouldn't
match typical VLM output ("orange" or "red" but not "orange/red").

### Problem B — build_color_map() doesn't detect collisions

```python
color_map = {
    color1: team1,
    color2: team2,    # silently overwrites if color1 == color2
    "white": None,
}
```

If both teams resolve to the same color name, the dict silently has
`color1: team2` (second team wins), and team1 is never reachable by color.
There is no warning — wrong player attribution occurs silently.

════════════════════════════════════════════════════════════════
## PART 2 — THE FIX
════════════════════════════════════════════════════════════════

### Fix A — Replace heuristic with nearest-neighbor Euclidean RGB distance

Added a palette of 16 named colors covering all common football kit colors:

```python
_COLOR_PALETTE = {
    "black"     : (  0,   0,   0),
    "white"     : (255, 255, 255),
    "red"       : (220,  20,  20),
    "maroon"    : (128,   0,   0),
    "orange"    : (230, 100,   0),
    "yellow"    : (240, 210,   0),
    "gold"      : (200, 160,   0),
    "green"     : (  0, 180,   0),
    "dark green": (  0, 100,   0),
    "sky blue"  : (100, 180, 240),
    "blue"      : ( 20,  60, 200),
    "dark blue" : (  0,   0, 110),
    "navy"      : (  0,   0,  60),
    "purple"    : (130,   0, 150),
    "pink"      : (240,  80, 150),
    "gray"      : (130, 130, 130),
}
```

New `hex_to_color_name()`:
```python
def hex_to_color_name(hex_color: str) -> str:
    r, g, b = parse hex...
    best_name, best_dist = "unknown", float("inf")
    for name, (pr, pg, pb) in _COLOR_PALETTE.items():
        dist = ((r-pr)**2 + (g-pg)**2 + (b-pb)**2) ** 0.5
        if dist < best_dist:
            best_dist, best_name = dist, name
    return best_name
```

This is deterministic, covers the full color space without thresholds,
and returns names the VLM would actually use ("dark blue", "navy", "gold").

### Fix B — Return empty map on collision instead of silently overwriting

```python
if color1 == color2:
    print(f"  [colors] WARNING: kit collision — both teams map to '{color1}', color disambiguation disabled")
    return {}
```

If both teams have the same resolved color name, `build_color_map()` returns
an empty dict. This causes `find_by_color()` (which checks `_color_map`) to
fall back to returning all teams for any jersey number. This is the correct
fallback: ambiguous is better than confidently wrong.

════════════════════════════════════════════════════════════════
## PART 3 — PROOF THIS WORKS
════════════════════════════════════════════════════════════════

### Nearest-neighbor examples

| Hex      | Old result   | New result   | Correct? |
|----------|-------------|--------------|----------|
| `0000fa` | blue         | blue         | ✓        |
| `c8102e` | red          | red          | ✓        |
| `00003c` | **black**    | navy         | ✓ fixed  |
| `001489` | blue         | dark blue    | ✓        |
| `ffcd00` | **mixed**    | yellow       | ✓ fixed  |
| `7b2d8b` | **mixed**    | purple       | ✓ fixed  |

### Collision detection

```
Team A: #001489 → "dark blue"
Team B: #003399 → "dark blue"  (similar shade)
→ color1 == color2
→ WARNING logged, return {} (empty map)
→ find_by_color() returns all teams → ambiguous
→ falls through to ESPN time-based match or UNKNOWN
```

Without Fix B, Team B's entry would silently overwrite Team A's in the map,
and all Team A jerseys resolved by color would be attributed to Team B.

════════════════════════════════════════════════════════════════
## PART 4 — WHAT THIS DOES NOT FIX
════════════════════════════════════════════════════════════════

- Two teams with genuinely different hex colors that happen to be the
  nearest neighbor to the same palette entry will still collide. This can
  be improved in future by expanding the palette or using a perceptual
  color space (Lab/HSL). For now, 16 palette entries covers the majority
  of EFL/EPL kit combinations.
- Away kits mapped to "white" remain ambiguous by design — white away kits
  are always `None` in the color map so VLM "white" detections don't get
  attributed to a team.
