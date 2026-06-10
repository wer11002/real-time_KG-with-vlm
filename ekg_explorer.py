"""
ekg_explorer.py — Interactive EKG Schema & Data Explorer
─────────────────────────────────────────────────────────
Run:
    pip install streamlit
    streamlit run ekg_explorer.py
    streamlit run ekg_explorer.py -- --ttl data/kg_output/ekg.ttl
"""

import sys
import argparse
import tempfile
from pathlib import Path
from collections import defaultdict

import streamlit as st
import streamlit.components.v1 as components
from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef

# ── namespaces ────────────────────────────────────────────────────────────────

EKG_NS   = "http://soccerekg.org/ontology#"
INST_NS  = "http://soccerekg.org/data#"

EKG    = Namespace(EKG_NS)
INST   = Namespace(INST_NS)
FOAF   = Namespace("http://xmlns.com/foaf/0.1/")
SCHEMA = Namespace("https://schema.org/")
PROV   = Namespace("http://www.w3.org/ns/prov#")

BASE_DIR     = Path(__file__).resolve().parent
DEFAULT_TTL  = BASE_DIR / "data" / "kg_output" / "ekg.ttl"

# ── helpers ───────────────────────────────────────────────────────────────────

def short(uri) -> str:
    s = str(uri)
    if "#" in s:
        return s.split("#")[-1]
    return s.split("/")[-1]


def ns_prefix(uri) -> str:
    s = str(uri)
    if s.startswith(EKG_NS):    return f"ekg:{short(uri)}"
    if s.startswith(INST_NS):   return f"data:{short(uri)}"
    if "foaf" in s:             return f"foaf:{short(uri)}"
    if "schema.org" in s:       return f"schema:{short(uri)}"
    if "prov" in s:             return f"prov:{short(uri)}"
    if "c4dm" in s:             return f"event:{short(uri)}"
    if "wgs84" in s:            return f"wgs84:{short(uri)}"
    if "skos" in s:             return f"skos:{short(uri)}"
    if "dcterms" in s:          return f"dcterms:{short(uri)}"
    return short(uri)


@st.cache_resource(show_spinner="Loading ontology…")
def load_graph(path: str) -> Graph:
    g = Graph()
    g.parse(path, format="turtle")
    return g


def ekg_classes(g) -> list[URIRef]:
    return sorted(
        [s for s in g.subjects(RDF.type, OWL.Class)
         if isinstance(s, URIRef) and str(s).startswith(EKG_NS)],
        key=short,
    )


def all_properties(g) -> list[tuple[str, URIRef]]:
    result = []
    for s in g.subjects(RDF.type, OWL.ObjectProperty):
        if isinstance(s, URIRef) and str(s).startswith(EKG_NS):
            result.append(("object", s))
    for s in g.subjects(RDF.type, OWL.DatatypeProperty):
        if isinstance(s, URIRef) and str(s).startswith(EKG_NS):
            result.append(("datatype", s))
    return sorted(result, key=lambda x: short(x[1]))


def ancestors(g, cls: URIRef) -> list[URIRef]:
    chain, visited = [], set()
    cur = cls
    while True:
        visited.add(cur)
        parents = [p for p in g.objects(cur, RDFS.subClassOf)
                   if isinstance(p, URIRef) and p not in visited]
        if not parents:
            break
        cur = parents[0]
        chain.append(cur)
    return chain          # nearest first → [PlayerAction, Event, ...]


def descendants(g, cls: URIRef, depth=0) -> list[tuple[URIRef, int]]:
    result = []
    for sub in g.subjects(RDFS.subClassOf, cls):
        if isinstance(sub, URIRef) and str(sub).startswith(EKG_NS):
            result.append((sub, depth))
            result.extend(descendants(g, sub, depth + 1))
    return sorted(result, key=lambda x: (x[1], short(x[0])))


def domain_properties(g, cls: URIRef) -> list[dict]:
    props = []
    for prop in g.subjects(RDFS.domain, cls):
        if not isinstance(prop, URIRef):
            continue
        ptype  = "Object"  if (prop, RDF.type, OWL.ObjectProperty)  in g else "Datatype"
        ranges = [ns_prefix(r) for r in g.objects(prop, RDFS.range)
                  if isinstance(r, URIRef)]
        subs   = [ns_prefix(s) for s in g.objects(prop, RDFS.subPropertyOf)
                  if isinstance(s, URIRef)]
        props.append({
            "name"   : short(prop),
            "type"   : ptype,
            "range"  : ", ".join(ranges) or "—",
            "aligns" : ", ".join(subs)   or "—",
        })
    return sorted(props, key=lambda p: p["name"])


def instances_of(g, cls: URIRef) -> list[URIRef]:
    return [s for s in g.subjects(RDF.type, cls)
            if isinstance(s, URIRef) and str(s).startswith(INST_NS)]


# ── full T-Box flow diagram ───────────────────────────────────────────────────

def build_tbox_flow_graph(g, highlight_cls: URIRef = None, height: int = 600) -> str | None:
    """
    Full T-Box schema flowchart with manual x,y positions (physics off).
    Layout:
      Row 0 — Match / Team / Player / Venue / League
      Row 1 — Event (base class)
      Row 2 — PlayerAction (left)  Card (right)
      Row 3 — PlayerAction subtypes (2 rows)  +  YellowCard / RedCard
    Solid gray  = subClassOf (parent → child)
    Dashed blue = key object properties
    """
    try:
        from pyvis.network import Network
    except ImportError:
        return None

    net = Network(height=f"{height}px", width="100%",
                  bgcolor="#ffffff", font_color="#222222", directed=True)
    net.set_options("""{
      "physics": { "enabled": false },
      "edges": { "smooth": { "enabled": true, "type": "cubicBezier" } }
    }""")

    GX, GY = 130, 108   # grid unit x, y

    # ── PlayerAction anchor and children layout ────────────────────────────────
    AC = -2 * GX          # PlayerAction center x
    CC =  5 * GX          # Card  center x

    # PlayerAction has 9 children — split into two rows of 5 and 4
    # Row 1: 5 children centered at AC
    R1 = [AC + (i - 2) * GX for i in range(5)]   # offsets -2,-1,0,1,2 from AC
    # Row 2: 4 children centered at AC
    R2 = [AC + (i - 1.5) * GX for i in range(4)]

    POSITIONS = {
        # Row 0 — domain root entities
        "Match":             (  0,        0),
        "Team":              (-3 * GX,    0),
        "Player":            ( 3 * GX,    0),
        "Venue":             (-5.5 * GX,  0),
        "League":            ( 5.5 * GX,  0),
        # Row 1 — base event class
        "Event":             (  0,      GY),
        # Row 2 — mid-level event classes
        "PlayerAction":       (AC,    2 * GY),
        "Card":         (CC,    2 * GY),
        # Row 3a — first 5 PlayerAction children
        "Goal":         (R1[0], 3 * GY),
        "Shot":         (R1[1], 3 * GY),
        "Foul":         (R1[2], 3 * GY),
        "Corner":       (R1[3], 3 * GY),
        "OffsideCalled":      (R1[4], 3 * GY),
        # Row 3b — next 4 PlayerAction children (offset row)
        "FreeKick":     (R2[0], 4 * GY),
        "Substitution": (R2[1], 4 * GY),
        "PenaltyEvent":      (R2[2], 4 * GY),
        "PassEvent":         (R2[3], 4 * GY),
        # Row 3a — Card children (same row as first action subtypes)
        "YellowCard":   (CC - GX,  3 * GY),
        "RedCard":      (CC + GX,  3 * GY),
    }

    # ── color scheme ──────────────────────────────────────────────────────────
    CLASS_COLORS = {
        "Match":           ("#4A90D9", "#2c6fad", "#ffffff"),
        "Team":            ("#E74C3C", "#b03a2e", "#ffffff"),
        "Player":          ("#2ECC71", "#1a8a4a", "#ffffff"),
        "Event":           ("#8E44AD", "#6c3483", "#ffffff"),
        "PlayerAction":     ("#F0A500", "#c47d00", "#ffffff"),
        "Card":       ("#E67E22", "#ca6f1e", "#ffffff"),
        "YellowCard": ("#F9E400", "#c0a000", "#333333"),
        "RedCard":    ("#C0392B", "#922b21", "#ffffff"),
    }
    DEFAULT_C   = ("#FDE8C8", "#E8A020", "#333333")
    HIGHLIGHT_C = ("#FF6B35", "#cc4400", "#ffffff")

    def get_col(cls_uri):
        if highlight_cls and str(cls_uri) == str(highlight_cls):
            return HIGHLIGHT_C
        return CLASS_COLORS.get(short(cls_uri), DEFAULT_C)

    added_nodes: set = set()
    added_edges: set = set()

    # ── add nodes with manual positions ──────────────────────────────────────
    for cls in ekg_classes(g):
        name = short(cls)
        pos  = POSITIONS.get(name)
        if pos is None:
            continue  # class not in layout map → skip
        uid = str(cls)
        bg, border, fc = get_col(cls)
        is_hl = highlight_cls and str(cls) == str(highlight_cls)
        net.add_node(
            uid,
            label=ns_prefix(cls),
            x=int(pos[0]), y=int(pos[1]),
            color={"background": bg, "border": border},
            font={"color": fc, "size": 11, "bold": bool(is_hl)},
            shape="box",
            physics=False,
        )
        added_nodes.add(uid)

    def add_e(src, dst, label, color, dashes=False, width=1.8):
        key = (str(src), str(dst), label)
        if key in added_edges:
            return
        if str(src) not in added_nodes or str(dst) not in added_nodes:
            return
        added_edges.add(key)
        net.add_edge(str(src), str(dst), label=label, color=color,
                     arrows="to", width=width, dashes=dashes,
                     font={"size": 9, "color": "#666"})

    # ── subClassOf edges: EKG parent → child ──────────────────────────────────
    for cls in ekg_classes(g):
        if str(cls) not in added_nodes:
            continue
        for parent in g.objects(cls, RDFS.subClassOf):
            if isinstance(parent, URIRef) and str(parent).startswith(EKG_NS):
                add_e(parent, cls, "", "#BBBBBB", dashes=False, width=2.2)

    # ── key object-property edges (dashed, colored) ───────────────────────────
    PROP_COLORS = {
        "hasHomeTeam":     "#4A90D9",
        "hasAwayTeam":     "#4A90D9",
        "playsFor":       "#2ECC71",
        "performed":       "#E74C3C",
        "isPerformedBy": "#E74C3C",
        "involvedTeam":     "#E74C3C",
        "inMatch":        "#9B59B6",
        "triggered":       "#E67E22",
    }
    for ptype, prop in all_properties(g):
        if ptype != "object":
            continue
        pname = short(prop)
        col   = PROP_COLORS.get(pname)
        if not col:
            continue
        for dom in g.objects(prop, RDFS.domain):
            for rng in g.objects(prop, RDFS.range):
                if isinstance(dom, URIRef) and isinstance(rng, URIRef):
                    add_e(dom, rng, pname, col, dashes=True, width=1.4)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        return Path(f.name).read_text()


# ── page ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="EKG Explorer", layout="wide", page_icon="⚽")

# sidebar: file selector
with st.sidebar:
    st.header("⚽ EKG Explorer")
    ttl_input = st.text_input("TTL file", value=str(DEFAULT_TTL))
    st.caption("Path to ekg.ttl — reload page after changing")

ttl_path = Path(ttl_input)
if not ttl_path.exists():
    st.error(f"TTL file not found: `{ttl_path}`")
    st.stop()

g = load_graph(str(ttl_path))

classes = ekg_classes(g)
props   = all_properties(g)

# header metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Classes",    len(classes))
c2.metric("Properties", len(props))
c3.metric("Triples",    len(g))
total_inst = sum(1 for s in g.subjects(RDF.type, None)
                 if isinstance(s, URIRef) and str(s).startswith(INST_NS))
c4.metric("Instances",  total_inst)

st.divider()

# session state for node navigation
if "node" not in st.session_state:
    st.session_state.node = None

def goto(uri: str):
    st.session_state.node = uri
    st.session_state._jump_node = True

# tabs
tab_search, tab_schema, tab_graph, tab_instances, tab_node = st.tabs([
    "🔍 Search",
    "📋 Full Schema",
    "🕸️ Graph",
    "🗂️ Instances",
    "🔎 Node",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — SEARCH
# ═══════════════════════════════════════════════════════════════════════════

with tab_search:
    query = st.text_input(
        "Search classes and properties",
        placeholder="e.g.  Goal   hasTime   Shot   pitch_zone",
        label_visibility="collapsed",
    )

    if not query:
        st.caption("Type a class or property name above — partial matches shown. Case-insensitive.")
    else:
        q = query.strip().lower()

        def class_matches(cls):
            # match own name
            if q in short(cls).lower():
                return True
            # match any RDFS label
            if any(q in str(lbl).lower() for lbl in g.objects(cls, RDFS.label)):
                return True
            # match any ancestor name (e.g. "person" → Player because Player ⊆ foaf:Person)
            if any(q in short(a).lower() for a in ancestors(g, cls)):
                return True
            return False

        def prop_matches(prop):
            if q in short(prop).lower():
                return True
            if any(q in str(lbl).lower() for lbl in g.objects(prop, RDFS.label)):
                return True
            return False

        hit_classes = [c for c in classes if class_matches(c)]
        hit_props   = [(t, p) for t, p in props if prop_matches(p)]

        if not hit_classes and not hit_props:
            st.warning(f"No matches for **{query}**")
        else:
            # ── matched classes ───────────────────────────────────────────
            if hit_classes:
                st.subheader(f"Classes  ({len(hit_classes)})")
                for cls in hit_classes:
                    anc_match = [a for a in ancestors(g, cls) if q in short(a).lower()]
                    via = f"  —  via `{ns_prefix(anc_match[0])}`" if anc_match and q not in short(cls).lower() else ""
                    with st.expander(f"🏷️  {short(cls)}{via}", expanded=True):
                        # ── top row: properties + instances ──────────────────
                        left, right = st.columns([1, 1])

                        with left:
                            st.markdown("##### Properties on this class")
                            dp = domain_properties(g, cls)
                            if dp:
                                for p in dp:
                                    icon = "🔗" if p["type"] == "Object" else "📝"
                                    st.markdown(
                                        f"{icon} **`{p['name']}`** → `{p['range']}`"
                                        + (f"  ·  *aligns: {p['aligns']}*" if p["aligns"] != "—" else "")
                                    )
                            else:
                                st.caption("No direct properties declared on this class.")

                        with right:
                            inst = instances_of(g, cls)
                            st.markdown(f"##### Instances: **{len(inst)}**")
                            if inst:
                                PREVIEW = 6
                                show_key = f"_show_all_{short(cls)}"
                                if show_key not in st.session_state:
                                    st.session_state[show_key] = False
                                sorted_inst = sorted(inst, key=lambda x: short(x).lower())
                                visible = sorted_inst if st.session_state[show_key] else sorted_inst[:PREVIEW]
                                for i in visible:
                                    lbl = next((str(o) for o in g.objects(i, RDFS.label)), None) \
                                          or next((str(o) for o in g.objects(i, FOAF.name)), None) \
                                          or short(i)
                                    if st.button(f"• {lbl}", key=f"ins_{short(cls)}_{short(i)}", use_container_width=True):
                                        goto(str(i))
                                        st.info(f"**{lbl}** selected — click the **Node** tab to inspect.")
                                if len(inst) > PREVIEW:
                                    toggle_label = f"▲ Show less" if st.session_state[show_key] else f"▼ Show all {len(inst)}"
                                    if st.button(toggle_label, key=f"_toggle_{short(cls)}"):
                                        st.session_state[show_key] = not st.session_state[show_key]
                                        st.rerun()

                        # ── full-width T-Box flow diagram ─────────────────────
                        st.markdown("##### T-Box Schema Flow")
                        anc_list = ancestors(g, cls)
                        depth    = len(anc_list)
                        st.caption(
                            "Solid = subClassOf  ·  Dashed = object property  ·  "
                            + ("  ›  ".join(ns_prefix(a) for a in reversed(anc_list))
                               + f"  ›  **{ns_prefix(cls)}**" if anc_list else "Root class")
                        )
                        tbox_html = build_tbox_flow_graph(g, cls, height=600)
                        if tbox_html:
                            components.html(tbox_html, height=610, scrolling=False)
                        else:
                            st.caption("pyvis not installed — run `pip install pyvis`")

            # ── matched properties ────────────────────────────────────────
            if hit_props:
                st.subheader(f"Properties  ({len(hit_props)})")
                for ptype, prop in hit_props:
                    icon   = "🔗" if ptype == "object" else "📝"
                    dom    = [ns_prefix(d) for d in g.objects(prop, RDFS.domain)
                              if isinstance(d, URIRef)]
                    rng    = [ns_prefix(r) for r in g.objects(prop, RDFS.range)
                              if isinstance(r, URIRef)]
                    subs   = [ns_prefix(s) for s in g.objects(prop, RDFS.subPropertyOf)
                              if isinstance(s, URIRef)]
                    with st.expander(f"{icon}  {short(prop)}", expanded=True):
                        st.markdown(f"**Type:** `{'ObjectProperty' if ptype == 'object' else 'DatatypeProperty'}`")
                        st.markdown(f"**Domain:** {', '.join(dom) or '—'}")
                        st.markdown(f"**Range:**  {', '.join(rng) or '—'}")
                        if subs:
                            st.markdown(f"**Aligns to:** {', '.join(subs)}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — FULL SCHEMA
# ═══════════════════════════════════════════════════════════════════════════

with tab_schema:
    col_cls, col_prop = st.columns([1, 1])

    with col_cls:
        st.subheader("Class Hierarchy")
        # Build tree rooted at EKG.Event
        roots = [c for c in classes if not any(
            isinstance(p, URIRef) and str(p).startswith(EKG_NS)
            for p in g.objects(c, RDFS.subClassOf)
        )]

        def render_branch(cls, depth=0):
            anc  = ancestors(g, cls)
            inst = instances_of(g, cls)
            indent = "  " * depth
            badge  = f"  `{len(inst)} inst`" if inst else ""
            st.markdown(f"{indent}**{short(cls)}**{badge}")
            for sub, _ in descendants(g, cls, 0):
                # only direct children
                direct_parents = [p for p in g.objects(sub, RDFS.subClassOf)
                                  if isinstance(p, URIRef)]
                if cls in direct_parents:
                    render_branch(sub, depth + 1)

        for root in roots:
            render_branch(root)

    with col_prop:
        st.subheader("Object Properties")
        for ptype, prop in props:
            if ptype != "object":
                continue
            dom = ", ".join(short(d) for d in g.objects(prop, RDFS.domain) if isinstance(d, URIRef))
            rng = ", ".join(short(r) for r in g.objects(prop, RDFS.range)  if isinstance(r, URIRef))
            st.markdown(f"🔗 **`{short(prop)}`**  `{dom}` → `{rng}`")

        st.subheader("Datatype Properties")
        for ptype, prop in props:
            if ptype != "datatype":
                continue
            rng = ", ".join(short(r) for r in g.objects(prop, RDFS.range) if isinstance(r, URIRef))
            dom = ", ".join(short(d) for d in g.objects(prop, RDFS.domain) if isinstance(d, URIRef))
            st.markdown(f"📝 **`{short(prop)}`**  domain: `{dom or '—'}` · range: `{rng or '—'}`")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — INSTANCES
# ═══════════════════════════════════════════════════════════════════════════

with tab_instances:
    st.subheader("Instances by Class")

    for cls in classes:
        inst = instances_of(g, cls)
        if not inst:
            continue
        with st.expander(f"**{short(cls)}** — {len(inst)} instances"):
            for i in inst[:30]:
                label_val = next((str(o) for o in g.objects(i, RDFS.label)), None) or \
                            next((str(o) for o in g.objects(i, FOAF.name)), None)
                display = label_val or short(i)
                col_lbl, col_btn = st.columns([5, 1])
                col_lbl.markdown(f"**{display}**  `{short(i)}`")
                if col_btn.button("Inspect →", key=f"ins_{short(cls)}_{str(i)}"):
                    goto(str(i))
                    st.rerun()
            if len(inst) > 30:
                st.caption(f"…{len(inst) - 30} more not shown")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — GRAPH
# ═══════════════════════════════════════════════════════════════════════════

# Node colors by type
NODE_COLORS = {
    "Match"              : "#4A90D9",   # blue
    "Team"               : "#E74C3C",   # red
    "Player"             : "#2ECC71",   # green
    "Goal"          : "#F1C40F",   # gold
    "Shot"          : "#F39C12",   # orange
    "Foul"          : "#E67E22",   # dark orange
    "Corner"        : "#D4AC0D",   # olive
    "FreeKick"      : "#CA6F1E",   # brown-orange
    "Substitution"  : "#8E44AD",   # purple
    "OffsideCalled"       : "#1ABC9C",   # teal
    "YellowCard"    : "#F9E400",   # yellow
    "RedCard"       : "#C0392B",   # dark red
    "PlayerAction"        : "#F0A500",   # amber (fallback)
    "default"            : "#BDC3C7",   # grey
}

def node_color(g, uri: URIRef) -> str:
    for t in g.objects(uri, RDF.type):
        name = short(t)
        if name in NODE_COLORS:
            return NODE_COLORS[name]
    return NODE_COLORS["default"]

def node_size(g, uri: URIRef) -> int:
    type_sizes = {"Match": 35, "Team": 30, "Player": 18}
    for t in g.objects(uri, RDF.type):
        name = short(t)
        if name in type_sizes:
            return type_sizes[name]
    return 14

def node_label(g, uri: URIRef) -> str:
    for lbl in g.objects(uri, RDFS.label):
        return str(lbl)
    s = short(uri)
    # shorten event IDs like "event_0042" → "ev_42"
    if s.startswith("event_"):
        return "ev_" + s.split("_")[-1].lstrip("0") or "0"
    return s


CLASS_LEVELS = {
    "Match"             : 0,
    "Team"              : 1,
    "Player"            : 2,
    "Person"            : 2,
    "PlayerAction"       : 3,
    "Event"             : 3,
    "Goal"         : 4,
    "Shot"         : 4,
    "Foul"         : 4,
    "Corner"       : 4,
    "FreeKick"     : 4,
    "Substitution" : 4,
    "OffsideCalled"      : 4,
    "PassEvent"         : 4,
    "PenaltyEvent"      : 4,
    "Card"         : 5,
    "YellowCard"   : 6,
    "RedCard"      : 6,
}


def build_pyvis(g, mode: str, max_nodes: int) -> str:
    try:
        from pyvis.network import Network
    except ImportError:
        return None

    net = Network(height="700px", width="100%", bgcolor="#ffffff",
                  font_color="#222222", directed=True)
    net.barnes_hut(gravity=-12000, central_gravity=0.3,
                   spring_length=120, spring_strength=0.04)

    added_nodes = set()
    added_edges = set()

    def add_node(uri, label=None, color=None, size=14, title=None, **kwargs):
        uid = str(uri)
        if uid in added_nodes:
            return
        if len(added_nodes) >= max_nodes:
            return
        added_nodes.add(uid)
        net.add_node(uid,
                     label=label or short(uri),
                     color=color or node_color(g, uri),
                     size=size,
                     title=title or uid,
                     font={"size": 11},
                     **kwargs)

    def add_edge(src, dst, label="", color="#aaaaaa"):
        key = (str(src), str(dst), label)
        if key in added_edges:
            return
        added_edges.add(key)
        net.add_edge(str(src), str(dst),
                     label=label, color=color,
                     font={"size": 9, "color": "#555555"},
                     arrows="to", width=1.2)

    # ── SCHEMA mode — T-Box classes + object properties ─────────────────────
    if mode == "schema":
        net.set_options("""
{
  "layout": {
    "hierarchical": {
      "enabled": true,
      "direction": "UD",
      "sortMethod": "directed",
      "levelSeparation": 130,
      "nodeSpacing": 150
    }
  },
  "physics": { "enabled": false }
}
""")
        for cls in ekg_classes(g):
            lbl = short(cls)
            add_node(cls,
                     label=lbl,
                     color=NODE_COLORS.get(lbl, "#7FB3D3"),
                     size=22,
                     shape="box",
                     level=CLASS_LEVELS.get(lbl, 4))
        for cls in ekg_classes(g):
            for parent in g.objects(cls, RDFS.subClassOf):
                if isinstance(parent, URIRef) and str(parent).startswith(EKG_NS):
                    key = (str(cls), str(parent), "subClassOf")
                    if key not in added_edges:
                        added_edges.add(key)
                        net.add_edge(str(cls), str(parent),
                                     label="subClassOf",
                                     color="#CCCCCC",
                                     arrows="",
                                     dashes=False,
                                     width=1.0,
                                     font={"size": 8, "color": "#AAAAAA"})

        SCHEMA_DRAW_PROPS = {
            "hasHomeTeam": "#FF8C00",
            "hasAwayTeam": "#FF8C00",
            "playsFor"  : "#8E44AD",
            "member"     : "#8E44AD",
            "performed"  : "#2ECC71",
            "triggered"  : "#F39C12",
        }
        for ptype, prop in all_properties(g):
            if ptype != "object":
                continue
            pname = short(prop)
            if pname not in SCHEMA_DRAW_PROPS:
                continue
            doms = list(g.objects(prop, RDFS.domain))
            rngs = list(g.objects(prop, RDFS.range))
            for d in doms:
                for r in rngs:
                    if not isinstance(d, URIRef) or not isinstance(r, URIRef):
                        continue
                    is_trigger = pname == "triggered"
                    key = (str(d), str(r), pname)
                    if key not in added_edges:
                        added_edges.add(key)
                        net.add_edge(str(d), str(r),
                                     label=pname,
                                     color=SCHEMA_DRAW_PROPS[pname],
                                     dashes=is_trigger,
                                     arrows="to",
                                     width=1.5,
                                     font={"size": 9, "color": "#555555"})

    # ── INSTANCE mode — A-Box data ───────────────────────────────────────────
    else:
        obj_props = {str(p) for _, p in all_properties(g) if _ == "object"}
        # also allow foaf:member (Team→Player) and hasHomeTeam/hasAwayTeam
        obj_props |= {
            str(FOAF.member),
            str(EKG.hasHomeTeam),
            str(EKG.hasAwayTeam),
        }

        # Add Match and Team nodes first (always visible anchors)
        for uri in g.subjects(RDF.type, EKG.Match):
            if isinstance(uri, URIRef):
                lbl = next((str(o) for o in g.objects(uri, RDFS.label)), short(uri))
                add_node(uri, label=lbl,
                         color=NODE_COLORS["Match"], size=35,
                         title=f"Match: {lbl}")
        for uri in g.subjects(RDF.type, EKG.Team):
            if isinstance(uri, URIRef):
                lbl = next((str(o) for o in g.objects(uri, RDFS.label)), short(uri))
                add_node(uri, label=lbl,
                         color=NODE_COLORS["Team"], size=30,
                         title=f"Team: {lbl}")
        for uri in g.subjects(RDF.type, EKG.Player):
            if isinstance(uri, URIRef):
                lbl = next((str(o) for o in g.objects(uri, RDFS.label)), short(uri))
                add_node(uri, label=lbl,
                         color=NODE_COLORS["Player"], size=18,
                         title=f"Player: {lbl}")

        # Events — up to max_nodes
        for uri in g.subjects(RDF.type, EKG.PlayerAction):
            if not isinstance(uri, URIRef):
                continue
            etype = next((short(t) for t in g.objects(uri, RDF.type)
                          if short(t).endswith("Event") and short(t) != "PlayerAction"), "PlayerAction")
            color = NODE_COLORS.get(etype, NODE_COLORS["PlayerAction"])
            time_val = next((str(o) for o in g.objects(uri, EKG.hasTime)), "")
            add_node(uri,
                     label=f"{etype.replace('Event','')}\n{time_val}",
                     color=color, size=14,
                     title=f"{etype} @ {time_val}")

        # Edges — object properties only
        for s, p, o in g:
            if not isinstance(s, URIRef) or not isinstance(o, URIRef):
                continue
            if str(p) not in obj_props:
                continue
            if str(s) not in added_nodes or str(o) not in added_nodes:
                continue
            prop_name = short(p)
            DRAW_PROPS = {
                "hasHomeTeam": "#FF8C00",
                "hasAwayTeam": "#FF8C00",
                "member"     : "#8E44AD",
                "playsFor"  : "#8E44AD",
                "performed"  : "#2ECC71",
                "precededBy": "#888888",
                "triggered"  : "#F39C12",
                "assistedBy": "#1ABC9C",
            }
            if prop_name not in DRAW_PROPS:
                continue
            add_edge(s, o, prop_name, DRAW_PROPS[prop_name])

    # write to temp file and return HTML
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        return Path(f.name).read_text()


with tab_graph:
    col_mode, col_max = st.columns([2, 1])
    with col_mode:
        graph_mode = st.radio(
            "View",
            ["Instance graph (A-Box)", "Schema graph (T-Box)"],
            horizontal=True,
        )
    with col_max:
        max_nodes = st.slider("Max nodes", 20, 300, 120, step=10)

    mode_key = "schema" if "Schema" in graph_mode else "instance"

    html = build_pyvis(g, mode_key, max_nodes)
    if html is None:
        st.error("pyvis not installed. Run: `pip install pyvis`")
    else:
        components.html(html, height=720, scrolling=False)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — NODE INSPECTOR
# ═══════════════════════════════════════════════════════════════════════════

def find_path_to_match(g, start_uri: str) -> list:
    """BFS from start_uri outward; return shortest edge path to a Match node."""
    from collections import deque
    queue   = deque([(start_uri, [])])
    visited = {start_uri}
    match_uris = {str(u) for u in g.subjects(RDF.type, EKG.Match)}

    while queue:
        cur, path = queue.popleft()
        if cur in match_uris:
            return path

        cur_uri = URIRef(cur)
        # outgoing edges
        for pred, obj in g.predicate_objects(cur_uri):
            if not isinstance(obj, URIRef):
                continue
            obj_s = str(obj)
            if obj_s not in visited:
                visited.add(obj_s)
                queue.append((obj_s, path + [(cur, f"→ {short(pred)} →", obj_s)]))
        # incoming edges
        for subj, pred in g.subject_predicates(cur_uri):
            if not isinstance(subj, URIRef):
                continue
            subj_s = str(subj)
            if subj_s not in visited:
                visited.add(subj_s)
                queue.append((subj_s, path + [(cur, f"← {short(pred)} ←", subj_s)]))
    return []


def build_neighborhood_graph(g, center_uri: str) -> str:
    """
    Focused hierarchy: Match → (Home Team, Away Team) → Player → Events.
    Queries specific relationships only — no BFS, no clutter.
    """
    TYPE_BG = {
        "Match":  "#3A86FF",
        "Team":   "#E63946",
        "Player": "#F4A261",
        "Event":  "#2DC653",
    }
    EDGE_C = {
        "hasHomeTeam": "#3A86FF",
        "hasAwayTeam": "#3A86FF",
        "playsFor":   "#9B59B6",
        "performed":   "#2DC653",
    }

    def glabel(u):
        u = URIRef(u) if isinstance(u, str) else u
        lbl = next((str(o) for o in g.objects(u, RDFS.label)), None) or \
              next((str(o) for o in g.objects(u, FOAF.name)), None)
        if lbl:
            return lbl
        s = short(str(u))
        if s.startswith("event_"):
            tail  = s.split("_")[-1].lstrip("0") or "0"
            etype = next((short(str(t)) for t in g.objects(u, RDF.type)
                          if str(t).startswith(EKG_NS)), "Event")
            return f"{etype} #{tail}"
        return s

    def js(s):
        return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")

    player = URIRef(center_uri)

    # ── 1. find match ──────────────────────────────────────────────────────
    match = next(g.objects(player, EKG.participatedIn), None)
    if not match:
        for ev in g.objects(player, EKG.performed):
            match = next(g.objects(ev, EKG.inMatch), None)
            if match:
                break

    # ── 2. teams ───────────────────────────────────────────────────────────
    home_team = g.value(match, EKG.hasHomeTeam) if match else None
    away_team = g.value(match, EKG.hasAwayTeam) if match else None

    # ── 3. which team does this player belong to? ──────────────────────────
    player_team = None
    for team in (home_team, away_team):
        if team and (team, FOAF.member, player) in g:
            player_team = team
            break
    if not player_team:
        for team in (home_team, away_team):
            if team and (player, EKG.playsFor, team) in g:
                player_team = team
                break

    # ── 4. events performed (max 6, sorted by time) ────────────────────────
    ev_raw = list(g.objects(player, EKG.performed))
    def ev_time(e):
        t = g.value(e, EKG.hasTime)
        return str(t) if t else ""
    ev_raw.sort(key=ev_time)
    events = ev_raw[:6]

    # ── 5. build node/edge lists with fixed positions ──────────────────────
    nodes = []   # {id, label, x, y, bg}
    edges = []   # {from, to, label, color}
    added = set()

    def node(uid, label, x, y, bg, is_center=False):
        uid = str(uid)
        if uid in added:
            return
        added.add(uid)
        color = "#FF6B35" if is_center else bg
        nodes.append({"id": uid, "label": label, "x": x, "y": y, "bg": color})

    def edge(src, dst, label, color):
        edges.append({"from": str(src), "to": str(dst),
                      "label": label, "color": color})

    # Match — top center
    if match:
        node(match, glabel(match), 0, 0, TYPE_BG["Match"])

    # Teams — left (home) and right (away), y=170
    if home_team:
        node(home_team, glabel(home_team), -280, 170, TYPE_BG["Team"])
        if match:
            edge(match, home_team, "hasHomeTeam", EDGE_C["hasHomeTeam"])
    if away_team and away_team != home_team:
        node(away_team, glabel(away_team),  280, 170, TYPE_BG["Team"])
        if match:
            edge(match, away_team, "hasAwayTeam", EDGE_C["hasAwayTeam"])

    # Player — below their team
    px = -280 if player_team == home_team else (280 if player_team == away_team else 0)
    node(player, glabel(player), px, 340, TYPE_BG["Player"], is_center=True)
    if player_team:
        edge(player_team, player, "playsFor", EDGE_C["playsFor"])

    # Events — spread below player
    n = len(events)
    ev_gap = 160
    ev_x0  = px - (n - 1) * ev_gap / 2
    for i, ev in enumerate(events):
        ex = ev_x0 + i * ev_gap
        node(ev, glabel(ev), ex, 500, TYPE_BG["Event"])
        edge(player, ev, "performed", EDGE_C["performed"])

    # ── 6. render vis.js ───────────────────────────────────────────────────
    node_js = ",\n    ".join(
        f'{{id:"{js(n["id"])}",label:"{js(n["label"])}",x:{n["x"]},y:{n["y"]},'
        f'color:{{background:"{n["bg"]}",border:"rgba(255,255,255,0.25)",'
        f'highlight:{{background:"{n["bg"]}",border:"#fff"}}}},'
        f'font:{{color:"#fff",size:13,bold:true}},'
        f'shape:"dot",size:30,title:"{js(n["id"])}",borderWidth:2}}'
        for n in nodes
    )
    edge_js = ",\n    ".join(
        f'{{from:"{js(e["from"])}",to:"{js(e["to"])}",label:"{js(e["label"])}",'
        f'color:{{color:"{e["color"]}",highlight:"#fff"}},'
        f'font:{{size:10,color:"#bbb",strokeWidth:2,strokeColor:"#1a1a2e"}},'
        f'arrows:"to",width:2,smooth:{{enabled:false}}}}'
        for e in edges
    )

    return f"""<!DOCTYPE html><html><head>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  body{{margin:0;padding:0;background:#1a1a2e;}}
  #g{{width:100%;height:520px;}}
  .leg{{position:absolute;top:8px;right:10px;font:11px sans-serif;color:#bbb;line-height:1.8;}}
  .dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px;vertical-align:middle;}}
</style></head><body>
<div id="g"></div>
<div class="leg">
  <div><span class="dot" style="background:#3A86FF"></span>Match</div>
  <div><span class="dot" style="background:#E63946"></span>Team</div>
  <div><span class="dot" style="background:#FF6B35"></span>Selected player</div>
  <div><span class="dot" style="background:#F4A261"></span>Player</div>
  <div><span class="dot" style="background:#2DC653"></span>Event</div>
</div>
<script>
var net = new vis.Network(
  document.getElementById("g"),
  {{nodes: new vis.DataSet([{node_js}]),
    edges: new vis.DataSet([{edge_js}])}},
  {{physics:{{enabled:false}},
    interaction:{{hover:true,tooltipDelay:80}},
    nodes:{{borderWidth:2}}}}
);
</script></body></html>"""


with tab_node:
    # ── node selector ────────────────────────────────────────────────────────
    all_inst_uris = sorted(
        [str(s) for s in g.subjects(RDF.type, None)
         if isinstance(s, URIRef) and str(s).startswith(INST_NS)],
        key=lambda u: short(u)
    )

    default_idx = 0
    if st.session_state.node and st.session_state.node in all_inst_uris:
        default_idx = all_inst_uris.index(st.session_state.node)

    selected = st.selectbox(
        "Select or search a node",
        options=all_inst_uris,
        index=default_idx,
        format_func=lambda u: f"{short(u)}",
    )
    if selected:
        st.session_state.node = selected

    if not selected:
        st.caption("Select a node above or click **Inspect →** in the Instances tab.")
        st.stop()

    # ── node header ──────────────────────────────────────────────────────────
    uri = URIRef(selected)
    label = next((str(o) for o in g.objects(uri, RDFS.label)), None) or \
            next((str(o) for o in g.objects(uri, FOAF.name)), None) or short(selected)
    types = [short(t) for t in g.objects(uri, RDF.type) if isinstance(t, URIRef)]

    st.markdown(f"## {label}")
    st.caption(f"`{selected}`")
    st.markdown("  ".join(f"`{t}`" for t in types))
    st.divider()

    # ── path from Match ──────────────────────────────────────────────────────
    path = find_path_to_match(g, selected)
    if path:
        st.markdown("#### 📍 Path from Match")
        crumbs = []
        for (src, rel, dst) in path:
            src_lbl = next((str(o) for o in g.objects(URIRef(src), RDFS.label)), short(src))
            crumbs.append(f"**{src_lbl}** `{rel}`")
        dst_lbl = next((str(o) for o in g.objects(URIRef(path[-1][2]), RDFS.label)),
                       short(path[-1][2]))
        crumbs.append(f"**{dst_lbl}**")
        st.markdown("  ›  ".join(crumbs))
        st.divider()

    # ── properties (with clickable URI values) ───────────────────────────────
    st.markdown("#### 🔑 Properties")

    # group by predicate
    from collections import defaultdict as _dd
    pred_map = _dd(list)
    for pred, obj in g.predicate_objects(uri):
        if pred != RDF.type:
            pred_map[pred].append(obj)

    for pred in sorted(pred_map.keys(), key=lambda p: short(p)):
        pname = ns_prefix(pred)
        values = pred_map[pred]

        with st.expander(f"**`{pname}`**  ({len(values)} value{'s' if len(values)>1 else ''})",
                         expanded=True):
            for obj in values:
                if isinstance(obj, URIRef) and str(obj).startswith(INST_NS):
                    # clickable internal node
                    obj_label = next((str(o) for o in g.objects(obj, RDFS.label)), None) or \
                                next((str(o) for o in g.objects(obj, FOAF.name)), None) or short(str(obj))
                    obj_types  = [short(t) for t in g.objects(obj, RDF.type) if isinstance(t, URIRef)]
                    type_badge = f"  `{'  ·  '.join(obj_types[:2])}`" if obj_types else ""
                    col_val, col_nav = st.columns([6, 1])
                    col_val.markdown(f"🔗 **{obj_label}**{type_badge}  `{short(str(obj))}`")
                    if col_nav.button("→", key=f"nav_{pred}_{obj}", help=f"Inspect {short(str(obj))}"):
                        goto(str(obj))
                        st.rerun()
                elif isinstance(obj, URIRef):
                    # external URI (foaf, schema, etc.)
                    st.markdown(f"🌐 `{ns_prefix(obj)}`")
                else:
                    # literal value
                    st.markdown(f"📝 `{str(obj)}`")

    st.divider()

    # ── neighborhood graph ───────────────────────────────────────────────────
    st.markdown("#### 🕸️ Neighborhood Graph")
    n_html = build_neighborhood_graph(g, selected)
    components.html(n_html, height=530, scrolling=False)
