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


VOC_PRESETS: Dict[str, dict] = {
    "acetaldehyde": {
        "label": "Acetaldehyde (CH₃CHO)",
        "formula": "CH3CHO",
        "target_adsorbate": "CH3CHO",
        "routes": {
            "both": {
                "label": "Both routes: reduction + oxidation",
                "description": "Union of ethanol-route and acetate/CO2-route proxy descriptors.",
                "states": ACETALDEHYDE_BOTH_STATES,
            },
            "reduction": {
                "label": "Direct reduction route: CH3CHO → ethanol proxy",
                "description": "Direct acetaldehyde electroreduction proxy using H*, CH3CHO*, ethoxy-like, and ethanol-like states.",
                "states": ACETALDEHYDE_REDUCTION_STATES,
            },
            "oxidation": {
                "label": "Oxidation route: CH3CHO → acetate → CO2 proxy",
                "description": "Aldehyde activation, acetate-like intermediate formation, and deep oxidation using CO*/COOH* C1 proxies. O* is handled in OER mode.",
                "states": ACETALDEHYDE_OXIDATION_STATES,
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
            "oxidation uses aldehyde activation, acetate-like binding, and CO*/COOH* deep-oxidation proxies."
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


def state_components(state: str) -> Tuple[str, ...]:
    norm = normalize_voc_state(state)
    if not norm:
        return tuple()
    return tuple(clean_adsorbate_label(p) for p in norm.split("+"))


def is_voc_proximity_state(state: str) -> bool:
    return "+" in normalize_voc_state(state)
