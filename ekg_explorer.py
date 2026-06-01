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
    return chain          # nearest first → [ActionEvent, Event, ...]


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
      Row 2 — ActionEvent (left)  CardEvent (right)
      Row 3 — ActionEvent subtypes (2 rows)  +  YellowCard / RedCard
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

    # ── ActionEvent anchor and children layout ────────────────────────────────
    AC = -2 * GX          # ActionEvent center x
    CC =  5 * GX          # CardEvent  center x

    # ActionEvent has 9 children — split into two rows of 5 and 4
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
        "ActionEvent":       (AC,    2 * GY),
        "CardEvent":         (CC,    2 * GY),
        # Row 3a — first 5 ActionEvent children
        "GoalEvent":         (R1[0], 3 * GY),
        "ShotEvent":         (R1[1], 3 * GY),
        "FoulEvent":         (R1[2], 3 * GY),
        "CornerEvent":       (R1[3], 3 * GY),
        "OffsideEvent":      (R1[4], 3 * GY),
        # Row 3b — next 4 ActionEvent children (offset row)
        "FreeKickEvent":     (R2[0], 4 * GY),
        "SubstitutionEvent": (R2[1], 4 * GY),
        "PenaltyEvent":      (R2[2], 4 * GY),
        "PassEvent":         (R2[3], 4 * GY),
        # Row 3a — CardEvent children (same row as first action subtypes)
        "YellowCardEvent":   (CC - GX,  3 * GY),
        "RedCardEvent":      (CC + GX,  3 * GY),
    }

    # ── color scheme ──────────────────────────────────────────────────────────
    CLASS_COLORS = {
        "Match":           ("#4A90D9", "#2c6fad", "#ffffff"),
        "Team":            ("#E74C3C", "#b03a2e", "#ffffff"),
        "Player":          ("#2ECC71", "#1a8a4a", "#ffffff"),
        "Event":           ("#8E44AD", "#6c3483", "#ffffff"),
        "ActionEvent":     ("#F0A500", "#c47d00", "#ffffff"),
        "CardEvent":       ("#E67E22", "#ca6f1e", "#ffffff"),
        "YellowCardEvent": ("#F9E400", "#c0a000", "#333333"),
        "RedCardEvent":    ("#C0392B", "#922b21", "#ffffff"),
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
        "PLAYS_FOR":       "#2ECC71",
        "PERFORMED":       "#E74C3C",
        "IS_PERFORMED_BY": "#E74C3C",
        "INVOLVED_IN":     "#E74C3C",
        "IN_MATCH":        "#9B59B6",
        "TRIGGERED":       "#E67E22",
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
    "GoalEvent"          : "#F1C40F",   # gold
    "ShotEvent"          : "#F39C12",   # orange
    "FoulEvent"          : "#E67E22",   # dark orange
    "CornerEvent"        : "#D4AC0D",   # olive
    "FreeKickEvent"      : "#CA6F1E",   # brown-orange
    "SubstitutionEvent"  : "#8E44AD",   # purple
    "OffsideEvent"       : "#1ABC9C",   # teal
    "YellowCardEvent"    : "#F9E400",   # yellow
    "RedCardEvent"       : "#C0392B",   # dark red
    "ActionEvent"        : "#F0A500",   # amber (fallback)
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

    def add_node(uri, label=None, color=None, size=14, title=None):
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
                     font={"size": 11})

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
        for cls in ekg_classes(g):
            add_node(cls,
                     label=short(cls),
                     color=NODE_COLORS.get(short(cls), "#7FB3D3"),
                     size=22)
        for cls in ekg_classes(g):
            for parent in g.objects(cls, RDFS.subClassOf):
                if isinstance(parent, URIRef) and str(parent).startswith(EKG_NS):
                    add_edge(cls, parent, "subClassOf", "#AAAAAA")
        for ptype, prop in all_properties(g):
            if ptype != "object":
                continue
            doms = list(g.objects(prop, RDFS.domain))
            rngs = list(g.objects(prop, RDFS.range))
            for d in doms:
                for r in rngs:
                    if isinstance(d, URIRef) and isinstance(r, URIRef):
                        add_edge(d, r, short(prop), "#4A90D9")

    # ── INSTANCE mode — A-Box data ───────────────────────────────────────────
    else:
        obj_props = {str(p) for _, p in all_properties(g) if _ == "object"}

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
        for uri in g.subjects(RDF.type, EKG.ActionEvent):
            if not isinstance(uri, URIRef):
                continue
            etype = next((short(t) for t in g.objects(uri, RDF.type)
                          if short(t).endswith("Event") and short(t) != "ActionEvent"), "ActionEvent")
            color = NODE_COLORS.get(etype, NODE_COLORS["ActionEvent"])
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
            edge_colors = {
                "IN_MATCH"      : "#4A90D9",
                "IS_PERFORMED_BY": "#2ECC71",
                "PERFORMED"     : "#2ECC71",
                "INVOLVED_IN"   : "#E74C3C",
                "PLAYS_FOR"     : "#8E44AD",
                "PRECEDED_BY"   : "#888888",
            }
            add_edge(s, o, prop_name, edge_colors.get(prop_name, "#BBBBBB"))

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
    Neo4j-style hierarchy: Match → Team → Player → Events.
    Uses raw vis.js — no pyvis dependency.
    Reification nodes and non-EKG predicates are filtered out.
    """
    EKG_PREDS = {
        "IN_MATCH", "IS_PERFORMED_BY", "PERFORMED", "INVOLVED_IN",
        "PLAYS_FOR", "PRECEDED_BY", "hasHomeTeam", "hasAwayTeam",
        "PARTICIPATED_IN", "TRIGGERED", "member",
    }
    SKIP_PREFIXES = ("plays_for_", "participated_in_", "card_for_",
                     "involved_in_", "in_match_")

    def glabel(uri_s):
        u = URIRef(uri_s)
        lbl = next((str(o) for o in g.objects(u, RDFS.label)), None) or \
              next((str(o) for o in g.objects(u, FOAF.name)), None)
        if lbl:
            return lbl
        s = short(uri_s)
        if s.startswith("event_"):
            tail = s.split("_")[-1].lstrip("0") or "0"
            typ  = next((short(str(t)) for t in g.objects(u, RDF.type)
                         if str(t).startswith(EKG_NS)), "Event")
            return f"{typ} #{tail}"
        return s

    def gtype(uri_s):
        for t in g.objects(URIRef(uri_s), RDF.type):
            s = short(str(t))
            if s in ("Match", "Team", "Player"):
                return s
            if s.endswith("Event"):
                return "Event"
        return "Other"

    def is_noise(uri_s):
        s = short(uri_s)
        return any(s.startswith(p) for p in SKIP_PREFIXES)

    # ── BFS 2-hop from center ─────────────────────────────────────────────
    nodes_info = {}   # uri_str → {label, type, is_center}
    raw_edges  = []   # (src, rel, dst) — strings

    def add_node(uri_s, is_center=False):
        if uri_s not in nodes_info and not is_noise(uri_s):
            nodes_info[uri_s] = {"label": glabel(uri_s),
                                  "type":  gtype(uri_s),
                                  "is_center": is_center}
            return True
        return False

    add_node(center_uri, is_center=True)
    frontier = [center_uri]
    visited  = {center_uri}

    for _ in range(2):
        nxt = []
        for cur in frontier:
            cur_uri = URIRef(cur)
            for pred, obj in g.predicate_objects(cur_uri):
                p = short(str(pred))
                if p not in EKG_PREDS or not isinstance(obj, URIRef):
                    continue
                obj_s = str(obj)
                if is_noise(obj_s):
                    continue
                add_node(obj_s)
                raw_edges.append((cur, p, obj_s))
                if obj_s not in visited:
                    visited.add(obj_s)
                    nxt.append(obj_s)
            for subj, pred in g.subject_predicates(cur_uri):
                p = short(str(pred))
                if p not in EKG_PREDS or not isinstance(subj, URIRef):
                    continue
                subj_s = str(subj)
                if is_noise(subj_s):
                    continue
                add_node(subj_s)
                raw_edges.append((subj_s, p, cur))
                if subj_s not in visited:
                    visited.add(subj_s)
                    nxt.append(subj_s)
        frontier = nxt

    # ── assign x,y by type ────────────────────────────────────────────────
    LEVEL_Y = {"Match": 60, "Team": 200, "Player": 340, "Event": 480, "Other": 340}
    TYPE_BG  = {"Match": "#3A86FF", "Team": "#E63946",
                 "Player": "#F4A261", "Event": "#2DC653", "Other": "#6C757D"}
    X_GAP = 200

    by_type: dict = {}
    for u, info in nodes_info.items():
        by_type.setdefault(info["type"], []).append(u)

    positions: dict = {}
    for t, uris in by_type.items():
        span = (len(uris) - 1) * X_GAP
        for i, u in enumerate(uris):
            positions[u] = (-span / 2 + i * X_GAP, LEVEL_Y.get(t, 340))

    # ── build vis.js HTML ─────────────────────────────────────────────────
    def js_str(s):
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")

    node_js_list = []
    for u, info in nodes_info.items():
        x, y = positions.get(u, (0, 340))
        bg   = "#FF6B35" if info["is_center"] else TYPE_BG.get(info["type"], "#6C757D")
        lbl  = js_str(info["label"])
        node_js_list.append(
            f'{{id:"{js_str(u)}",label:"{lbl}",x:{x},y:{y},'
            f'color:{{background:"{bg}",border:"rgba(255,255,255,0.3)",'
            f'highlight:{{background:"{bg}",border:"#fff"}}}},'
            f'font:{{color:"#fff",size:13,face:"Inter,sans-serif",bold:true}},'
            f'shape:"dot",size:28,title:"{js_str(u)}",borderWidth:2}}'
        )

    EDGE_C = {
        "IN_MATCH":"#3A86FF","IS_PERFORMED_BY":"#2DC653","PERFORMED":"#2DC653",
        "INVOLVED_IN":"#E63946","PLAYS_FOR":"#9B59B6","PRECEDED_BY":"#7F8C8D",
        "hasHomeTeam":"#E63946","hasAwayTeam":"#E63946",
        "PARTICIPATED_IN":"#3A86FF","TRIGGERED":"#FF6B35",
    }

    seen_edges: set = set()
    edge_js_list = []
    for src, rel, dst in raw_edges:
        key = (src, dst, rel)
        if key in seen_edges or src not in nodes_info or dst not in nodes_info:
            continue
        seen_edges.add(key)
        ec = EDGE_C.get(rel, "#AAAAAA")
        edge_js_list.append(
            f'{{from:"{js_str(src)}",to:"{js_str(dst)}",label:"{rel}",'
            f'color:{{color:"{ec}",highlight:"#fff"}},'
            f'font:{{size:10,color:"#ccc",strokeWidth:3,strokeColor:"#1a1a2e"}},'
            f'arrows:"to",width:2,smooth:{{enabled:false}}}}'
        )

    nodes_str = ",\n    ".join(node_js_list)
    edges_str = ",\n    ".join(edge_js_list)

    html = f"""<!DOCTYPE html><html><head>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  body{{margin:0;padding:0;background:#1a1a2e;}}
  #graph{{width:100%;height:520px;}}
  .legend{{position:absolute;top:10px;right:14px;font:11px Inter,sans-serif;color:#ccc;}}
  .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:middle;}}
</style></head>
<body>
<div id="graph"></div>
<div class="legend">
  <div><span class="dot" style="background:#3A86FF"></span>Match</div>
  <div><span class="dot" style="background:#E63946"></span>Team</div>
  <div><span class="dot" style="background:#FF6B35"></span>Selected</div>
  <div><span class="dot" style="background:#F4A261"></span>Player</div>
  <div><span class="dot" style="background:#2DC653"></span>Event</div>
</div>
<script>
var nodes = new vis.DataSet([
    {nodes_str}
]);
var edges = new vis.DataSet([
    {edges_str}
]);
var net = new vis.Network(document.getElementById("graph"),
    {{nodes:nodes, edges:edges}},
    {{physics:{{enabled:false}},
      interaction:{{hover:true,tooltipDelay:80,navigationButtons:false}},
      nodes:{{borderWidth:2}}}});
</script></body></html>"""
    return html


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
