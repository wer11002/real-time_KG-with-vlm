"""
buffer.py
---------
A rolling buffer that holds video-detected actions until the ESPN scrape
arrives. When ESPN data comes in, the buffer is flushed for matching.

What changed in this version:
    - VideoEvent now includes jersey, team, team_color, description
      fields from Qwen2-VL output
    - make_video_event passes all VLM fields through
    - dedup_window_sec default raised to 30.0 to catch overlapping clips

Quick test:
    python buffer.py
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VideoEvent:
    """One action detected in the video by Qwen2-VL."""
    video_time   : float            # absolute seconds from match start
    action       : str              # "Shot", "Foul", "Corner"...
    confidence   : float            # 0.0 to 1.0
    gametime     : str              # e.g. "1st 09:14"
    clip_start   : float            # start_sec of the clip this came from
    # VLM fields — from Qwen2-VL output
    jersey       : Optional[str] = None   # jersey number e.g. "7"
    team         : Optional[str] = None   # resolved team name
    team_color   : Optional[str] = None   # raw color e.g. "blue/white"
    shorts_color : Optional[str] = None
    socks_color  : Optional[str] = None
    kit_pattern  : Optional[str] = None
    pitch_zone   : Optional[str] = None
    body_part    : Optional[str] = None
    outcome      : Optional[str] = None   # Shot/Goal result e.g. "saved_low", "wide_right"
    foul_type    : Optional[str] = None   # Foul sub-type e.g. "tackle", "handball"
    team_side    : Optional[str] = None   # "home" or "away" — VLM-inferred
    ball_visible : Optional[bool] = None  # False triggers quality flag
    description  : Optional[str] = None   # VLM natural language description
    detected_at  : str = field(default_factory=lambda: datetime.now().isoformat())


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY HELPER
# ═══════════════════════════════════════════════════════════════════════════

def make_video_event(detected: dict, clip_start_sec: float, gametime: str) -> VideoEvent:
    """
    Convert one action_recognizer output dict into a VideoEvent.

    detected keys from Qwen2-VL:
        action, confidence, video_time, time_in_clip,
        jersey, team, team_color, description
    """
    return VideoEvent(
        video_time  = detected["video_time"],
        action      = detected["action"],
        confidence  = detected["confidence"],
        gametime    = detected.get("gametime", gametime),
        clip_start  = clip_start_sec,
        jersey       = detected.get("jersey"),
        team         = detected.get("team"),
        team_color   = detected.get("team_color"),
        shorts_color = detected.get("shorts_color"),
        socks_color  = detected.get("socks_color"),
        kit_pattern  = detected.get("kit_pattern"),
        pitch_zone   = detected.get("pitch_zone"),
        body_part    = detected.get("body_part"),
        outcome      = detected.get("outcome"),
        foul_type    = detected.get("foul_type"),
        team_side    = detected.get("team_side"),
        ball_visible = detected.get("ball_visible"),
        description  = detected.get("description"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# BUFFER CLASS
# ═══════════════════════════════════════════════════════════════════════════

class EventBuffer:
    """
    Holds video events until ESPN data arrives.
    Deduplicates overlapping detections automatically on add.
    """

    def __init__(self, dedup_window_sec: float = 30.0):
        """
        Args:
            dedup_window_sec : if same action is seen within this many seconds
                               of an existing entry, skip it (keep higher confidence)
                               Default 30s catches overlapping 60s clips (step=30s)
        """
        self._events      : List[VideoEvent] = []
        self.dedup_window : float = dedup_window_sec

    # ── add events ─────────────────────────────────────────────────────────

    def add(self, event: VideoEvent) -> bool:
        """
        Add a new event. Returns True if added, False if duplicate.
        If duplicate with higher confidence → replaces the old one.
        """
        duplicate = self._find_duplicate(event)

        if duplicate is None:
            self._events.append(event)
            return True

        if event.confidence > duplicate.confidence:
            self._events.remove(duplicate)
            self._events.append(event)
            return True

        return False

    def add_many(self, events: List[VideoEvent]) -> int:
        """Add multiple events. Returns how many added after dedup."""
        added = 0
        for e in events:
            if self.add(e):
                added += 1
        return added

    def add_from_detections(
        self,
        detections     : List[dict],
        clip_start_sec : float,
        gametime       : str,
    ) -> int:
        """
        Convert Qwen2-VL output dicts → VideoEvents and add to buffer.
        All VLM fields (jersey, team, team_color, description) are preserved.
        """
        events = [make_video_event(d, clip_start_sec, gametime) for d in detections]
        return self.add_many(events)

    # ── dedup ──────────────────────────────────────────────────────────────

    def _find_duplicate(self, new: VideoEvent) -> Optional[VideoEvent]:
        for e in self._events:
            if e.action != new.action:
                continue
            # Both conditions must hold: video times must be close AND the
            # clips must be adjacent (clip_start within dedup_window).
            # OR would incorrectly deduplicate different events in adjacent
            # clips (Fix 003 — see log_fix/fix_003_dedup_or_logic.md).
            time_close   = abs(e.video_time  - new.video_time)  <= self.dedup_window
            clip_overlap = abs(e.clip_start  - new.clip_start)  <= self.dedup_window
            if time_close and clip_overlap:
                return e
        return None

    # ── access & flush ─────────────────────────────────────────────────────

    def get_all(self) -> List[VideoEvent]:
        return sorted(self._events, key=lambda e: e.video_time)

    def get_recent(self, current_sec: float, minutes: float = 2.0) -> List[VideoEvent]:
        """Return events from the last `minutes` relative to current_sec."""
        cutoff = current_sec - minutes * 60.0
        return [e for e in self._events if e.video_time >= cutoff]

    def size(self) -> int:
        return len(self._events)

    def flush(self) -> List[VideoEvent]:
        events       = self.get_all()
        self._events = []
        return events

    def clear(self):
        self._events = []

    def __repr__(self):
        return f"<EventBuffer size={len(self._events)}>"

    def summary(self) -> str:
        if not self._events:
            return "buffer is empty"
        lines = [f"buffer has {len(self._events)} events:"]
        for e in self.get_all():
            jersey = f" #{e.jersey}" if e.jersey else ""
            lines.append(
                f"  {e.gametime:<12} {e.action:<10} "
                f"conf={e.confidence:.2f}  t={e.video_time:.1f}s{jersey}"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# quick self-test
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("─── EventBuffer self-test ───\n")

    buf = EventBuffer(dedup_window_sec=30.0)

    print("→ clip 1: Shot at 14s  jersey=#7")
    buf.add(VideoEvent(
        video_time=14.2, action="Shot", confidence=0.85,
        gametime="1st 00:14", clip_start=0.0,
        jersey="7", team="Blackburn Rovers", team_color="blue",
        description="Player #7 takes a shot from outside the box",
    ))

    print("→ clip 2: Shot at 14s (DUPLICATE — should be skipped)")
    buf.add(VideoEvent(
        video_time=14.5, action="Shot", confidence=0.82,
        gametime="1st 00:14", clip_start=30.0,
        jersey="7", team="Blackburn Rovers",
    ))

    print("→ clip 2: Foul at 42s  jersey=#23")
    buf.add(VideoEvent(
        video_time=42.8, action="Foul", confidence=0.73,
        gametime="1st 00:42", clip_start=30.0,
        jersey="23", team="Nottingham Forest",
    ))

    print("→ clip 3: Shot at 75s (different time → new event)")
    buf.add(VideoEvent(
        video_time=75.1, action="Shot", confidence=0.78,
        gametime="1st 01:15", clip_start=60.0,
    ))

    print("\n" + buf.summary())

    # test add_from_detections with VLM fields
    print("\n── Testing add_from_detections with VLM fields ──")
    fake_detections = [
        {
            "action": "Goal", "confidence": 0.92,
            "video_time": 3780.0, "time_in_clip": 30.0,
            "jersey": "7", "team": "Blackburn Rovers",
            "team_color": "blue",
            "description": "Player #7 scores from close range",
        },
    ]
    added = buf.add_from_detections(fake_detections, clip_start_sec=3750.0, gametime="1st 63:00")
    print(f"  added {added} event(s)")

    # verify VLM fields preserved
    events = buf.get_all()
    goal   = [e for e in events if e.action == "Goal"][0]
    print(f"  jersey preserved: #{goal.jersey}")
    print(f"  team preserved: {goal.team}")
    print(f"  description preserved: {goal.description[:40]}...")

    print("\n→ flushing buffer...")
    flushed = buf.flush()
    print(f"  flushed {len(flushed)} events")
    print(f"  buffer empty: {buf.size() == 0}")

    print("\n✓ all good!")