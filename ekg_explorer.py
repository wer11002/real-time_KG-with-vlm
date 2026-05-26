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
from pathlib import Path
from collections import defaultdict

import streamlit as st
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
tab_search, tab_schema, tab_instances = st.tabs([
    "🔍 Search",
    "📋 Full Schema",
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
