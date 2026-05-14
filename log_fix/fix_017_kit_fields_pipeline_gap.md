# Fix 017 — Kit Fields Pipeline Gap: VideoEvent Missing shorts/socks/kit_pattern

**Status:** Applied  
**Files changed:** `src/3_buffer_matcher/buffer.py`, `src/3_buffer_matcher/align.py`

---

## Problem

`VideoEvent` dataclass in `buffer.py` was missing `shorts_color`, `socks_color`,
and `kit_pattern`. These three fields were added to `action_recognizer.py` output
(Fix 011) and to `MatchedEvent` / KG schema, but were never wired into the buffer
stage. Result: the values were silently dropped as soon as `add_from_detections()`
was called, so kit disambiguation never actually worked end-to-end despite
appearing to be complete.

---

## Fix

**buffer.py — VideoEvent dataclass:**
```python
shorts_color : Optional[str] = None
socks_color  : Optional[str] = None
kit_pattern  : Optional[str] = None
```

**buffer.py — make_video_event():**
```python
shorts_color = detected.get("shorts_color"),
socks_color  = detected.get("socks_color"),
kit_pattern  = detected.get("kit_pattern"),
```

**align.py — align_buffer() v_dict conversion:**
```python
"shorts_color": getattr(v, "shorts_color", None),
"socks_color" : getattr(v, "socks_color",  None),
"kit_pattern" : getattr(v, "kit_pattern",  None),
```

---

## Result

Kit detail fields now flow fully: VLM output → buffer → align → KG triples.
The full pipeline for `hasDetectedShortsColor`, `hasDetectedSocksColor`, and
`hasKitPattern` is now operational.
