"""
app.py — Streamlit annotation interface.
Annotators rate AI commentary against video clips.
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="Soccer Commentary Evaluation",
    layout="wide",
)

EVENTS_FILE  = Path("data/annotation/events_to_rate.json")
RATINGS_DIR  = Path("data/annotation/ratings")
RATINGS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Sidebar: who's rating?
# ──────────────────────────────────────────────────────────────────────
st.sidebar.title("Soccer Commentary Evaluation")
rater_name = st.sidebar.text_input(
    "Your name (used to save your ratings)",
    value=st.session_state.get("rater_name", ""),
)
if not rater_name:
    st.warning("Please enter your name in the sidebar to start.")
    st.stop()
st.session_state.rater_name = rater_name

ratings_file = RATINGS_DIR / f"ratings_{rater_name.lower().replace(' ', '_')}.csv"

# Load existing ratings (resumable!)
if ratings_file.exists():
    saved_df = pd.read_csv(ratings_file)
    saved = {row["rating_id"]: row for _, row in saved_df.iterrows()}
else:
    saved = {}

# Load events
events = json.load(open(EVENTS_FILE))
total = len(events)
done = len(saved)
remaining = [e for e in events if e["rating_id"] not in saved]

# ──────────────────────────────────────────────────────────────────────
# Progress
# ──────────────────────────────────────────────────────────────────────
st.sidebar.markdown(f"### Progress")
st.sidebar.progress(done / total if total else 0.0)
st.sidebar.markdown(f"**{done} / {total}** rated")
st.sidebar.markdown(f"**{len(remaining)} remaining**")

if not remaining:
    st.success("🎉 You've rated all events! Thank you.")
    st.markdown(f"Your ratings are saved at `{ratings_file}`.")
    st.markdown("**Please send this file to the researcher.**")
    st.stop()

# ──────────────────────────────────────────────────────────────────────
# Show guidelines (collapsible)
# ──────────────────────────────────────────────────────────────────────
with st.sidebar.expander("📖 Rating guidelines"):
    st.markdown("""
**Accuracy** (1-5): *Did the commentary describe what actually happened?*
- 5: Every fact matches (player, action, location, outcome)
- 4: One minor detail off
- 3: Main event correct, multiple minor errors
- 2: Main event partially wrong
- 1: Main event completely wrong

**Completeness** (1-5): *Did it cover the important parts?*
- 5: Mentions player + action + outcome + relevant context
- 4: Player + action + outcome
- 3: Action + outcome, missing player
- 2: Action only
- 1: Doesn't describe the event

**Depth** (1-5): *Does it explain WHY this matters?*
- 5: Tactical insight or strong narrative ("his third attempt")
- 4: References past events meaningfully
- 3: Adds basic context (score/atmosphere)
- 2: Pure description, no insight
- 1: Generic ("good shot")

**Edge cases:**
- If you can't see the action clearly, lower scores
- Ignore grammar/typos
- Rate what's WRITTEN, not what you'd want it to say
    """)

# ──────────────────────────────────────────────────────────────────────
# Main rating panel
# ──────────────────────────────────────────────────────────────────────
event = remaining[0]
st.markdown(f"### Event {done + 1} of {total}")

col_video, col_text = st.columns([1, 1])
with col_video:
    st.markdown("**Watch this clip:**")
    video_path = Path(event["_clip"])
    if video_path.exists():
        with open(video_path, "rb") as f:
            st.video(f.read())
    else:
        st.error(f"Clip not found: {video_path}")
        st.write("Skip this event if you cannot see it.")

with col_text:
    st.markdown("**AI commentary for this moment:**")
    st.info(event["human_text"])

st.divider()

# Rating sliders
st.markdown("### Rate the commentary above")

c1, c2, c3 = st.columns(3)
with c1:
    accuracy = st.radio(
        "**Accuracy** — facts correct?",
        options=[1, 2, 3, 4, 5],
        index=2,
        horizontal=True,
        key=f"acc_{event['rating_id']}",
    )
with c2:
    completeness = st.radio(
        "**Completeness** — covers the event?",
        options=[1, 2, 3, 4, 5],
        index=2,
        horizontal=True,
        key=f"comp_{event['rating_id']}",
    )
with c3:
    depth = st.radio(
        "**Depth** — explains why it matters?",
        options=[1, 2, 3, 4, 5],
        index=2,
        horizontal=True,
        key=f"dep_{event['rating_id']}",
    )

notes = st.text_input("Notes (optional)",
                      key=f"notes_{event['rating_id']}")

col_back, col_skip, col_next = st.columns([1, 1, 2])
with col_skip:
    if st.button("⏭️ Skip (no clip)"):
        accuracy = completeness = depth = -1
        # Fall through to save logic

with col_next:
    if st.button("✅ Save & Next", type="primary"):
        new_row = {
            "rating_id": event["rating_id"],
            "rater": rater_name,
            "timestamp": datetime.now().isoformat(),
            "accuracy": accuracy,
            "completeness": completeness,
            "depth": depth,
            "notes": notes,
        }
        # Append to CSV
        if ratings_file.exists():
            df = pd.read_csv(ratings_file)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])
        df.to_csv(ratings_file, index=False)
        st.rerun()
