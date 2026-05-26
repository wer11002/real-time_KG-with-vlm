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


# ── hierarchy tree (ASCII) ────────────────────────────────────────────────────

def render_tree(g, cls: URIRef) -> str:
    anc  = list(reversed(ancestors(g, cls)))   # root → ... → parent
    desc = descendants(g, cls)

    lines = []
    for i, a in enumerate(anc):
        lines.append("  " * i + ("└─ " if i else "") + ns_prefix(a))

    root_indent = len(anc)
    lines.append("  " * root_indent + ("└─ " if anc else "") +
                 f"► {ns_prefix(cls)}")

    for d, depth in desc:
        lines.append("  " * (root_indent + 1 + depth) + "└─ " + ns_prefix(d))

    return "\n".join(lines)


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

# tabs
tab_search, tab_schema, tab_graph, tab_instances = st.tabs([
    "🔍 Search",
    "📋 Full Schema",
    "🕸️ Graph",
    "🗂️ Instances",
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
        st.caption("Type a class or property name above — partial matches shown.")
    else:
        q = query.strip().lower()
        hit_classes = [c for c in classes  if q in short(c).lower()]
        hit_props   = [(t, p) for t, p in props if q in short(p).lower()]

        if not hit_classes and not hit_props:
            st.warning(f"No matches for **{query}**")
        else:
            # ── matched classes ───────────────────────────────────────────
            if hit_classes:
                st.subheader(f"Classes  ({len(hit_classes)})")
                for cls in hit_classes:
                    with st.expander(f"🏷️  {short(cls)}", expanded=True):
                        left, right = st.columns([1, 1])

                        with left:
                            st.markdown("##### Hierarchy")
                            st.code(render_tree(g, cls), language=None)

                            anc_list = ancestors(g, cls)
                            depth    = len(anc_list)
                            st.caption(
                                f"Depth: **{depth}** "
                                f"({'  ›  '.join(ns_prefix(a) for a in reversed(anc_list))} ›  {ns_prefix(cls)})"
                                if anc_list else f"Root class (depth 0)"
                            )

                        with right:
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

                            inst = instances_of(g, cls)
                            st.markdown(f"##### Instances: **{len(inst)}**")
                            if inst:
                                for i in inst[:8]:
                                    st.caption(f"• {short(i)}")
                                if len(inst) > 8:
                                    st.caption(f"…and {len(inst) - 8} more")

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
                # collect all triples for this instance
                triples = list(g.predicate_objects(i))
                label_val = next(
                    (str(o) for _, p, o in [(i, pr, obj) for pr, obj in triples]
                     if p in (RDFS.label, FOAF.name)), None
                )
                header = label_val or short(i)
                with st.container():
                    st.markdown(f"**{header}**  `{short(i)}`")
                    for pred, obj in sorted(triples, key=lambda x: short(x[0])):
                        if pred == RDF.type:
                            continue
                        obj_str = short(obj) if isinstance(obj, URIRef) else str(obj)
                        st.caption(f"  {ns_prefix(pred)} = {obj_str}")
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
