"""VOC preset registry for SAGE-VOC mode.

The registry deliberately labels all VOC quantities as proxy descriptors.
They are intended for UMA/OCP pre-screening of surface accessibility and
H*/OH* co-adsorption proximity, not for definitive electrochemical free
energies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class VOCAdsorbateSpec:
    key: str
    label: str
    template: str | None
    anchor_mode: str
    role: str
    warning: str = ""


VOC_TEMPLATE_FILES: Dict[str, str] = {
    # Reduction / target VOC
    "CH3CHO": "CH3CHO_box.cif",
    "CH3CH2O": "CH3CH2O_box.cif",
    "CH3CH2OH": "CH3CH2OH_box.cif",

    # Oxidation: acetate/deep-oxidation route
    "CH3CO": "CH3CO_box.cif",
    "CH3COO": "CH3COO_box.cif",
    "CO": "CO_box.cif",
    "COOH": "COOH_box.cif",

    # Optional/product-retention and water-reaction species
    "CH3COOH": "CH3COOH_box.cif",
    "O": "O_box.cif",
    "OH": "OH_box.cif",
}


VOC_ADSORBATES: Dict[str, VOCAdsorbateSpec] = {
    "CH3CHO": VOCAdsorbateSpec(
        key="CH3CHO",
        label="Acetaldehyde",
        template="CH3CHO_box.cif",
        anchor_mode="carbonyl_c",
        role="target_voc",
    ),
    "CH3CH2O": VOCAdsorbateSpec(
        key="CH3CH2O",
        label="Ethoxy-like intermediate",
        template="CH3CH2O_box.cif",
        anchor_mode="o_atom",
        role="reduction_intermediate_proxy",
        warning="Reduction-route proxy; do not interpret as a full ethanol-formation free energy.",
    ),
    "CH3CH2OH": VOCAdsorbateSpec(
        key="CH3CH2OH",
        label="Ethanol product proxy",
        template="CH3CH2OH_box.cif",
        anchor_mode="o_atom",
        role="reduction_product_desorption_proxy",
        warning="Product-binding proxy; use as desorption/retention indicator only.",
    ),
    "CH3CO": VOCAdsorbateSpec(
        key="CH3CO",
        label="Acyl-like oxidation intermediate",
        template="CH3CO_box.cif",
        anchor_mode="carbonyl_c",
        role="oxidation_aldehyde_activation_proxy",
        warning="Oxidation-route acyl-like proxy; hydride/H-transfer and potential effects are not explicit.",
    ),
    "CH3COO": VOCAdsorbateSpec(
        key="CH3COO",
        label="Acetate-like intermediate",
        template="CH3COO_box.cif",
        anchor_mode="o_o_midpoint",
        role="oxidation_acetate_or_poisoning_proxy",
        warning="Carboxylate-like binding proxy; solvation/charge/speciation effects are not explicit.",
    ),
    "CO": VOCAdsorbateSpec(
        key="CO",
        label="CO-like C1 fragment proxy",
        template="CO_box.cif",
        anchor_mode="c_atom",
        role="deep_oxidation_c1_fragment_proxy",
        warning="Deep-oxidation proxy; can also indicate CO-like poisoning.",
    ),
    "COOH": VOCAdsorbateSpec(
        key="COOH",
        label="COOH-like CO-to-CO2 proxy",
        template="COOH_box.cif",
        anchor_mode="carboxyl_c",
        role="deep_oxidation_co2_pathway_proxy",
        warning="CO oxidation/CO2-formation proxy; not a full mineralization free energy.",
    ),
    "O": VOCAdsorbateSpec(
        key="O",
        label="O* oxidized-surface proxy",
        template="O_box.cif",
        anchor_mode="atom0",
        role="oxidized_surface_deep_oxidation_proxy",
    ),
    "OH": VOCAdsorbateSpec(
        key="OH",
        label="OH* hydroxyl/oxyhydroxide proxy",
        template="OH_box.cif",
        anchor_mode="o_atom",
        role="hydroxyl_oxidation_proxy",
    ),
    "CH3COOH": VOCAdsorbateSpec(
        key="CH3COOH",
        label="Acetic-acid product-retention proxy",
        template="CH3COOH_box.cif",
        anchor_mode="carboxyl_c",
        role="optional_product_retention_proxy",
        warning="Optional protonated-acetate proxy; pH/speciation is not explicit.",
    ),
}


ACETALDEHYDE_REDUCTION_STATES: List[str] = [
    "H*",
    "CH3CHO*",
    "CH3CH2O*",
    "CH3CH2OH*",
]

# ECH / co-adsorption states are intentionally disabled in the stable VOC branch.
# The current VOC workflow keeps only direct reduction and oxidation descriptors;
# H*+CH3CHO* and H*+H* should be reintroduced only after a site-aware,
# benchmarked co-adsorption placement policy is implemented.
ACETALDEHYDE_ECH_REDUCTION_STATES: List[str] = []

# Internal seed policies for disabled ECH states. Kept as an empty mapping for
# backward-compatible imports.
ECH_SEED_POLICIES: Dict[str, List[str]] = {}


ACETALDEHYDE_OXIDATION_STATES: List[str] = [
    "OH*",
    "CH3CHO*",
    "CH3CO*",
    "CH3COO*",
    "CO*",
    "COOH*",
]

ACETALDEHYDE_OPTIONAL_STATES: List[str] = [
    "CH3COOH*",
]


# -----------------------------------------------------------------------------
# Generic oxide-site routing for VOC descriptors
# -----------------------------------------------------------------------------
# Use generic classes internally; the backend expands these to element-resolved
# labels such as Ni_top, Fe_top, Ni-Ni_bridge, Ni-Fe_bridge, Ni-O_bridge, etc.
OXIDE_VOC_SITE_ROUTING: Dict[str, List[str]] = {
    # Reduction route
    "H": ["anion_top", "cation_top"],
    "CH3CHO": ["cation_top"],
    "CH3CH2O": ["cation_top"],
    "CH3CH2OH": ["cation_top"],

    # Oxidation route: acetate + deep oxidation extension
    "OH": ["cation_top"],
    "O": ["cation_top"],
    "CH3CO": ["cation_top"],
    "CH3COO": ["cation_cation_bridge"],
    "CO": ["cation_top"],
    "COOH": ["cation_top"],

    # Optional / diagnostic states
    "CH3COOH": ["cation_top", "cation_cation_bridge"],
}

OXIDE_VOC_SITE_POLICIES: Dict[str, dict] = {
    "fast_routed": {
        "label": "Fast routed oxide sites",
        "cation_top_per_element": 1,
        "anion_top_per_element": 1,
        "cation_pair_bridge_per_pair": 1,
        "cation_oxygen_bridge_per_pair": 0,
        "surface_z_window_A": 4.0,
    },
    "standard_routed": {
        "label": "Standard routed oxide sites",
        "cation_top_per_element": 1,
        "anion_top_per_element": 1,
        "cation_pair_bridge_per_pair": 1,
        "cation_oxygen_bridge_per_pair": 0,
        "surface_z_window_A": 4.0,
    },
    "extended_scan": {
        "label": "Extended oxide site scan",
        "cation_top_per_element": 2,
        "anion_top_per_element": 1,
        "cation_pair_bridge_per_pair": 2,
        "cation_oxygen_bridge_per_pair": 1,
        "surface_z_window_A": 5.0,
    },
}


def oxide_site_policy_options() -> List[str]:
    return list(OXIDE_VOC_SITE_POLICIES.keys())


def get_oxide_voc_site_policy(policy: str | None) -> dict:
    key = str(policy or "standard_routed").strip().lower()
    if key not in OXIDE_VOC_SITE_POLICIES:
        key = "standard_routed"
    out = dict(OXIDE_VOC_SITE_POLICIES[key])
    out["key"] = key
    return out


def allowed_oxide_site_classes_for_state(state: str) -> List[str]:
    comps = [p.replace("*", "").strip().upper() for p in str(state or "").replace(" ", "").split("+") if p.strip()]
    if not comps:
        return ["cation_top"]
    key = "+".join(comps)
    if key in OXIDE_VOC_SITE_ROUTING:
        return list(OXIDE_VOC_SITE_ROUTING[key])
    if len(comps) == 1:
        return list(OXIDE_VOC_SITE_ROUTING.get(comps[0], ["cation_top"]))
    return ["cation_top"]


def _dedupe_states(states: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for st in states:
        norm = str(st)
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


ACETALDEHYDE_BOTH_STATES: List[str] = _dedupe_states(
    ACETALDEHYDE_REDUCTION_STATES + ACETALDEHYDE_OXIDATION_STATES
)


# -----------------------------------------------------------------------------
# Deterministic VOC pathway display registry
# -----------------------------------------------------------------------------
# These definitions provide only a layout for registered descriptor states.
# Node colors are assigned later from selected QA-valid ΔE_proxy values. The
# registry does not infer products, selectivity, reaction barriers, or step energies.
ACETALDEHYDE_REDUCTION_PATHWAY: dict = {
    "lane_labels": {
        0: "Registered reduction sequence",
        1: "Auxiliary reduction descriptor",
    },
    "nodes": [
        {"id": "CH3CHO*", "state": "CH3CHO*", "label": "CH₃CHO*", "role": "reactant", "group": "main", "level": 0, "lane": 0},
        {"id": "CH3CH2O*", "state": "CH3CH2O*", "label": "CH₃CH₂O*", "role": "intermediate", "group": "main", "level": 1, "lane": 0},
        {"id": "CH3CH2OH*", "state": "CH3CH2OH*", "label": "CH₃CH₂OH*", "role": "registered_state", "group": "main", "level": 2, "lane": 0},
        {"id": "H*", "state": "H*", "label": "H*", "role": "auxiliary_descriptor", "group": "auxiliary", "level": 1, "lane": 1},
    ],
    "edges": [
        {"from": "CH3CHO*", "to": "CH3CH2O*", "edge_type": "registered_sequence"},
        {"from": "CH3CH2O*", "to": "CH3CH2OH*", "edge_type": "registered_sequence"},
        {"from": "H*", "to": "CH3CH2O*", "edge_type": "co_reactant_support"},
    ],
    "products": [],
}


ACETALDEHYDE_OXIDATION_PATHWAY: dict = {
    "lane_labels": {
        0: "Registered C₂ oxidation sequence",
        1: "Auxiliary oxidation descriptor",
        2: "Independent C₁ formation descriptors",
    },
    "nodes": [
        {"id": "CH3CHO*", "state": "CH3CHO*", "label": "CH₃CHO*", "role": "reactant", "group": "main", "level": 0, "lane": 0},
        {"id": "CH3CO*", "state": "CH3CO*", "label": "CH₃CO*", "role": "intermediate", "group": "main", "level": 1, "lane": 0},
        {"id": "CH3COO*", "state": "CH3COO*", "label": "CH₃COO*", "role": "registered_state", "group": "main", "level": 2, "lane": 0},
        {"id": "OH*", "state": "OH*", "label": "OH*", "role": "auxiliary_descriptor", "group": "auxiliary", "level": 1, "lane": 1},
        {"id": "CO*", "state": "CO*", "label": "CO*", "role": "independent_descriptor", "group": "independent_c1", "level": 0, "lane": 2},
        {"id": "COOH*", "state": "COOH*", "label": "COOH*", "role": "independent_descriptor", "group": "independent_c1", "level": 1, "lane": 2},
    ],
    "edges": [
        {"from": "CH3CHO*", "to": "CH3CO*", "edge_type": "registered_sequence"},
        {"from": "CH3CO*", "to": "CH3COO*", "edge_type": "registered_sequence"},
        {"from": "OH*", "to": "CH3CO*", "edge_type": "co_reactant_support"},
    ],
    "products": [],
}


def _merge_pathway_definitions(*defs: dict) -> dict:
    """Merge small registry graphs while preserving first-seen order."""
    nodes, edges, products = [], [], []
    seen_nodes, seen_edges, seen_products = set(), set(), set()
    for definition in defs:
        for node in definition.get("nodes", []):
            key = str(node.get("id", ""))
            if key and key not in seen_nodes:
                seen_nodes.add(key)
                nodes.append(dict(node))
        for edge in definition.get("edges", []):
            key = (str(edge.get("from", "")), str(edge.get("to", "")), str(edge.get("edge_type", "")))
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append(dict(edge))
        for product in definition.get("products", []):
            key = str(product.get("id", ""))
            if key and key not in seen_products:
                seen_products.add(key)
                products.append(dict(product))
    return {"nodes": nodes, "edges": edges, "products": products}


ACETALDEHYDE_BOTH_PATHWAY: dict = _merge_pathway_definitions(
    ACETALDEHYDE_REDUCTION_PATHWAY,
    ACETALDEHYDE_OXIDATION_PATHWAY,
)


VOC_PRESETS: Dict[str, dict] = {
    "acetaldehyde": {
        "label": "Acetaldehyde (CH₃CHO)",
        "formula": "CH3CHO",
        "target_adsorbate": "CH3CHO",
        "routes": {
            "both": {
                "label": "Both routes: reduction + oxidation",
                "description": "Union of the registered ethanol-side reduction states and C2 oxidation/C1 descriptor states.",
                "states": ACETALDEHYDE_BOTH_STATES,
                **ACETALDEHYDE_BOTH_PATHWAY,
            },
            "reduction": {
                "label": "Acetaldehyde reduction state-energy map",
                "description": "Selected QA-valid ΔE_proxy values for the registered direct-reduction states; H* is shown separately as an auxiliary descriptor.",
                "states": ACETALDEHYDE_REDUCTION_STATES,
                **ACETALDEHYDE_REDUCTION_PATHWAY,
            },
            "oxidation": {
                "label": "Acetaldehyde oxidation state-energy map",
                "description": "Selected QA-valid ΔE_proxy values for the registered C2 states. CO* and COOH* are displayed separately as independent C1 descriptors; no product or CO2 prediction is made.",
                "states": ACETALDEHYDE_OXIDATION_STATES,
                **ACETALDEHYDE_OXIDATION_PATHWAY,
            },
        },
        "reduction_states": ACETALDEHYDE_REDUCTION_STATES,
        "oxidation_states": ACETALDEHYDE_OXIDATION_STATES,
        "optional_states": ACETALDEHYDE_OPTIONAL_STATES,
        "core_adsorbates": ["H*", "OH*", "CH3CHO*"],
        "proximity_states": [],
        "optional_intermediates": ACETALDEHYDE_OPTIONAL_STATES,
        "default_route": "reduction",
        "default_states": ACETALDEHYDE_REDUCTION_STATES,
        "all_states": _dedupe_states(ACETALDEHYDE_BOTH_STATES + ACETALDEHYDE_OPTIONAL_STATES),
        "interpretation": (
            "UMA/OCP-based acetaldehyde route proxy descriptors. Reduction uses direct ethanol-route proxies; "
            "oxidation uses a C2 sequence to acetate-like binding plus independent CO*/COOH* C1 formation descriptors."
        ),
        "warning": (
            "SAGE-VOC reports ΔE_proxy and route/proximity descriptors. These are not definitive electrochemical ΔG values and do not explicitly resolve "
            "electrode potential, GPE/solvation, CTAC, or mass-transfer effects."
        ),
    }
}


def normalize_voc_state(state: str) -> str:
    """Normalize user-facing state labels while preserving co-adsorption syntax."""
    s = str(state or "").strip().replace(" ", "")
    if not s:
        return s
    parts = [p.replace("*", "").upper() + "*" for p in s.split("+")]
    return "+".join(parts)


def clean_adsorbate_label(label: str) -> str:
    return str(label or "").replace("*", "").strip().upper()


def get_voc_preset(key: str) -> dict:
    k = str(key or "acetaldehyde").strip().lower()
    if k not in VOC_PRESETS:
        raise KeyError(f"Unsupported VOC preset: {key!r}")
    return VOC_PRESETS[k]


def get_voc_route(key: str, route: str | None = None) -> dict:
    """Return a route definition with deterministic fallback to the preset default."""
    preset = get_voc_preset(key)
    routes = dict(preset.get("routes", {}))
    route_key = str(route or preset.get("default_route", "reduction")).strip().lower()
    if route_key not in routes:
        route_key = str(preset.get("default_route", "reduction")).strip().lower()
    if route_key not in routes:
        raise KeyError(f"Unsupported VOC route {route!r} for preset {key!r}")
    out = dict(routes[route_key])
    out["key"] = route_key
    return out


def state_components(state: str) -> Tuple[str, ...]:
    norm = normalize_voc_state(state)
    if not norm:
        return tuple()
    return tuple(clean_adsorbate_label(p) for p in norm.split("+"))


def is_voc_proximity_state(state: str) -> bool:
    return "+" in normalize_voc_state(state)
