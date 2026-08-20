"""Central registry for SAGE CO2RR C1 intermediate screening.

The registry defines product-targeted intermediate *sets*.  These sets are
adsorption/CHE descriptor presets, not a claim that one unique elementary
mechanism applies to every catalyst surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple


@dataclass(frozen=True)
class CO2RRAdsorbateSpec:
    key: str
    label: str
    template: str
    anchor_mode: str
    role: str
    h2_coeff: float
    h2o_coeff: float
    composition: Tuple[Tuple[str, int], ...]
    bond_requirements: Tuple[Tuple[str, str, int, float], ...]
    binding_mode: str = "C_bound"
    surface_anchor_family: str = "single_cation_basin"
    target_bond_length_A: float = 2.00
    allowed_site_kinds: Tuple[str, ...] = ("ontop", "bridge", "hollow")
    preferred_site_kinds: Tuple[str, ...] = ("ontop",)
    initial_clearance_A: float = 1.80
    migration_threshold_A: float = 2.50
    warning: str = ""


@dataclass(frozen=True)
class CO2RRBindingVariant:
    """One alternative initial binding geometry for a chemical state.

    Variants share the same CHE stoichiometry and adsorbate key.  They differ
    only in the template/anchor used to seed the surface calculation.
    """

    key: str
    label: str
    template: str
    anchor_mode: str
    binding_mode: str
    surface_anchor_family: str
    target_bond_length_A: float
    allowed_site_kinds: Tuple[str, ...]
    preferred_site_kinds: Tuple[str, ...]
    initial_clearance_A: float


@dataclass(frozen=True)
class CO2RREdgeSpec:
    """One directed edge in the CO2RR thermodynamic reaction network.

    ``n_pe`` is the number of proton-electron pairs consumed in the forward
    (reduction) direction.  Only PCET edges respond directly to electrode
    potential.  Desorption and other chemical edges are retained separately
    so that an uphill chemical step is never hidden inside a limiting
    potential.
    """

    source: str
    target: str
    label: str
    n_pe: int
    edge_type: str = "pcet"
    potential_sensitive: bool = True
    required_site_count: int = 1


@dataclass(frozen=True)
class CO2RRProductSpec:
    """Gas/solution endpoint referenced to CO2, H2, and H2O."""

    key: str
    state_id: str
    label: str
    gas_key: str
    h2_coeff: float
    h2o_coeff: float
    terminal_states: Tuple[str, ...]
    branch_start_state: str = "CO2"
    warning: str = ""


def _spec(
    key: str,
    label: str,
    anchor_mode: str,
    role: str,
    h2_coeff: float,
    h2o_coeff: float,
    composition,
    bond_requirements,
    *,
    binding_mode: str = "C_bound",
    surface_anchor_family: str = "single_cation_basin",
    target_bond_length_A: float = 2.00,
    allowed_site_kinds=("ontop", "bridge", "hollow"),
    preferred_site_kinds=("ontop",),
    initial_clearance_A: float = 1.80,
    migration_threshold_A: float = 2.50,
    warning: str = "",
) -> CO2RRAdsorbateSpec:
    return CO2RRAdsorbateSpec(
        key=key,
        label=label,
        template=f"{key}_box.cif",
        anchor_mode=anchor_mode,
        role=role,
        h2_coeff=float(h2_coeff),
        h2o_coeff=float(h2o_coeff),
        composition=tuple((str(s), int(n)) for s, n in composition),
        bond_requirements=tuple((str(a), str(b), int(n), float(r)) for a, b, n, r in bond_requirements),
        binding_mode=str(binding_mode),
        surface_anchor_family=str(surface_anchor_family),
        target_bond_length_A=float(target_bond_length_A),
        allowed_site_kinds=tuple(str(x) for x in allowed_site_kinds),
        preferred_site_kinds=tuple(str(x) for x in preferred_site_kinds),
        initial_clearance_A=float(initial_clearance_A),
        migration_threshold_A=float(migration_threshold_A),
        warning=str(warning),
    )


CO2RR_ADSORBATES: Dict[str, CO2RRAdsorbateSpec] = {
    "COOH": _spec(
        "COOH", "COOH* (C-bound)", "c_atom", "co_pathway_entry",
        0.5, 0.0, (("C", 1), ("O", 2), ("H", 1)),
        (("C", "O", 2, 1.90), ("O", "H", 1, 1.35)),
        binding_mode="C_bound_carboxyl", target_bond_length_A=2.00,
    ),
    "CO": _spec(
        "CO", "CO* (C-bound)", "c_atom", "co_intermediate_or_product",
        1.0, 1.0, (("C", 1), ("O", 1)), (("C", "O", 1, 1.60),),
        binding_mode="C_bound_CO", target_bond_length_A=1.90,
        initial_clearance_A=1.65, migration_threshold_A=0.80,
    ),
    "HCOO": _spec(
        "HCOO", "Formate* (bidentate O,O)", "o_o_midpoint", "formate_bidentate",
        0.5, 0.0, (("C", 1), ("O", 2), ("H", 1)),
        (("C", "O", 2, 1.90), ("C", "H", 1, 1.35)),
        binding_mode="bidentate_OO", surface_anchor_family="cation_pair",
        target_bond_length_A=2.10, allowed_site_kinds=("bridge", "cation_pair"),
        preferred_site_kinds=("bridge",), initial_clearance_A=1.40,
    ),
    "OCHO": _spec(
        "OCHO", "Formate* (monodentate O)", "o_min_z", "formate_monodentate",
        0.5, 0.0, (("C", 1), ("O", 2), ("H", 1)),
        (("C", "O", 2, 1.90), ("C", "H", 1, 1.35)),
        binding_mode="monodentate_O", target_bond_length_A=2.05,
        preferred_site_kinds=("ontop",), initial_clearance_A=1.40,
    ),
    "CHO": _spec(
        "CHO", "CHO* (C-bound)", "c_atom", "methanol_carbonyl_hydrogenation",
        1.5, 1.0, (("C", 1), ("O", 1), ("H", 1)),
        (("C", "O", 1, 1.90), ("C", "H", 1, 1.35)),
        binding_mode="C_bound_formyl", target_bond_length_A=1.95,
        initial_clearance_A=1.55,
    ),
    "COH": _spec(
        "COH", "COH* (C-bound)", "c_atom", "methane_deoxygenation_branch",
        1.5, 1.0, (("C", 1), ("O", 1), ("H", 1)),
        (("C", "O", 1, 1.90), ("O", "H", 1, 1.35)),
        binding_mode="C_bound_hydroxymethylidene", target_bond_length_A=1.95,
        initial_clearance_A=1.55,
    ),
    "CHOH": _spec(
        "CHOH", "CHOH* (C-bound hydroxycarbene)", "c_atom", "methane_cho_deoxygenation_branch",
        2.0, 1.0, (("C", 1), ("O", 1), ("H", 2)),
        (("C", "O", 1, 1.90), ("C", "H", 1, 1.35), ("O", "H", 1, 1.35)),
        binding_mode="C_bound_hydroxycarbene", target_bond_length_A=1.95,
        initial_clearance_A=1.50,
    ),
    "CH2O": _spec(
        "CH2O", "CH2O* (formaldehyde; C/O binding screened)", "c_atom", "methanol_oxygenate_intermediate",
        2.0, 1.0, (("C", 1), ("O", 1), ("H", 2)),
        (("C", "O", 1, 1.90), ("C", "H", 2, 1.35)),
        binding_mode="C_bound_formaldehyde", target_bond_length_A=2.00,
        initial_clearance_A=1.55,
    ),
    "CH3O": _spec(
        "CH3O", "CH3O* (O-bound methoxy)", "o_atom", "methanol_methoxy_branch",
        2.5, 1.0, (("C", 1), ("O", 1), ("H", 3)),
        (("C", "O", 1, 1.90), ("C", "H", 3, 1.35)),
        binding_mode="O_bound_methoxy", target_bond_length_A=2.05,
        preferred_site_kinds=("ontop",), initial_clearance_A=1.45,
    ),
    "CH2OH": _spec(
        "CH2OH", "CH2OH* (C-bound hydroxymethyl)", "c_atom", "methanol_hydroxymethyl_branch",
        2.5, 1.0, (("C", 1), ("O", 1), ("H", 3)),
        (("C", "O", 1, 1.90), ("C", "H", 2, 1.35), ("O", "H", 1, 1.35)),
        binding_mode="C_bound_hydroxymethyl", target_bond_length_A=2.00,
        initial_clearance_A=1.55,
    ),
    "CH": _spec(
        "CH", "CH* (C-bound)", "c_atom", "methane_fragment",
        2.5, 2.0, (("C", 1), ("H", 1)), (("C", "H", 1, 1.35),),
        binding_mode="C_bound_CH", target_bond_length_A=1.90,
        initial_clearance_A=1.45, migration_threshold_A=0.80,
    ),
    "CH2": _spec(
        "CH2", "CH2* (C-bound)", "c_atom", "methane_fragment",
        3.0, 2.0, (("C", 1), ("H", 2)), (("C", "H", 2, 1.35),),
        binding_mode="C_bound_CH2", target_bond_length_A=1.95,
        initial_clearance_A=1.45, migration_threshold_A=0.80,
    ),
    "CH3": _spec(
        "CH3", "CH3* (C-bound)", "c_atom", "methane_fragment",
        3.5, 2.0, (("C", 1), ("H", 3)), (("C", "H", 3, 1.35),),
        binding_mode="C_bound_CH3", target_bond_length_A=2.00,
        initial_clearance_A=1.45, migration_threshold_A=0.80,
    ),
}

CO2RR_TEMPLATE_FILES: Dict[str, str] = {
    key: spec.template for key, spec in CO2RR_ADSORBATES.items()
}


def _default_binding_variant(spec: CO2RRAdsorbateSpec) -> CO2RRBindingVariant:
    return CO2RRBindingVariant(
        key="default",
        label=str(spec.label),
        template=str(spec.template),
        anchor_mode=str(spec.anchor_mode),
        binding_mode=str(spec.binding_mode),
        surface_anchor_family=str(spec.surface_anchor_family),
        target_bond_length_A=float(spec.target_bond_length_A),
        allowed_site_kinds=tuple(spec.allowed_site_kinds),
        preferred_site_kinds=tuple(spec.preferred_site_kinds),
        initial_clearance_A=float(spec.initial_clearance_A),
    )


# CH2O is retained only as a legacy/custom surface state so that historical
# result files remain readable.  It is not part of the default C1 network:
# neutral CH2O is formaldehyde and is represented by HCHO(g) when used as a
# product endpoint.  CH2OH*, on the other hand, is a genuine adsorbed radical
# state and is seeded through both C and O to avoid a single-orientation bias.
CO2RR_BINDING_VARIANTS: Dict[str, Tuple[CO2RRBindingVariant, ...]] = {
    "CH2O": (
        CO2RRBindingVariant(
            key="C_bound",
            label="CH2O* C-bound seed",
            template="CH2O_box.cif",
            anchor_mode="c_atom",
            binding_mode="C_bound_formaldehyde",
            surface_anchor_family="single_cation_basin",
            target_bond_length_A=2.00,
            allowed_site_kinds=("ontop", "bridge", "hollow"),
            preferred_site_kinds=("ontop",),
            initial_clearance_A=1.55,
        ),
        CO2RRBindingVariant(
            key="O_bound",
            label="CH2O* O-bound seed",
            template="CH2O_O_box.cif",
            anchor_mode="o_atom",
            binding_mode="O_bound_formaldehyde",
            surface_anchor_family="single_cation_basin",
            target_bond_length_A=2.05,
            allowed_site_kinds=("ontop", "bridge", "hollow"),
            preferred_site_kinds=("ontop",),
            initial_clearance_A=1.45,
        ),
    ),
    "CH2OH": (
        CO2RRBindingVariant(
            key="C_bound",
            label="CH2OH* C-bound seed",
            template="CH2OH_box.cif",
            anchor_mode="c_atom",
            binding_mode="C_bound_hydroxymethyl",
            surface_anchor_family="single_cation_basin",
            target_bond_length_A=2.00,
            allowed_site_kinds=("ontop", "bridge", "hollow"),
            preferred_site_kinds=("ontop",),
            initial_clearance_A=1.55,
        ),
        CO2RRBindingVariant(
            key="O_bound",
            label="CH2OH* O-bound seed",
            template="CH2OH_box.cif",
            anchor_mode="o_atom",
            binding_mode="O_bound_hydroxymethyl",
            surface_anchor_family="single_cation_basin",
            target_bond_length_A=2.05,
            allowed_site_kinds=("ontop", "bridge", "hollow"),
            preferred_site_kinds=("ontop",),
            initial_clearance_A=1.45,
        ),
    ),
}

CO2RR_PRODUCTS: Dict[str, CO2RRProductSpec] = {
    "co": CO2RRProductSpec(
        key="co",
        state_id="CO(g)",
        label="CO(g)",
        gas_key="CO",
        h2_coeff=1.0,
        h2o_coeff=1.0,
        terminal_states=("CO*",),
        branch_start_state="CO*",
    ),
    "formate": CO2RRProductSpec(
        key="formate",
        state_id="HCOOH(g)",
        label="HCOOH(g) / formate thermodynamic proxy",
        gas_key="HCOOH",
        h2_coeff=1.0,
        h2o_coeff=0.0,
        terminal_states=("HCOO*", "OCHO*"),
        branch_start_state="CO2",
        warning=(
            "Neutral HCOOH is used as the molecular endpoint. Explicit aqueous "
            "formate activity and acid-base speciation are not included."
        ),
    ),
    "formaldehyde": CO2RRProductSpec(
        key="formaldehyde",
        state_id="HCHO(g)",
        label="HCHO(g)",
        gas_key="HCHO",
        h2_coeff=2.0,
        h2o_coeff=1.0,
        terminal_states=("CHO*",),
        branch_start_state="CO*",
        warning=(
            "CH2O_box.cif is treated as neutral gas-phase formaldehyde "
            "(HCHO), not as a mandatory adsorbed CH2O* intermediate."
        ),
    ),
    "methanol": CO2RRProductSpec(
        key="methanol",
        state_id="CH3OH(g)",
        label="CH3OH(g)",
        gas_key="CH3OH",
        h2_coeff=3.0,
        h2o_coeff=1.0,
        terminal_states=("CH2OH*",),
        branch_start_state="CHOH*",
        warning=(
            "CH3OH(g) endpoint is complete only when CH3OH is supplied in "
            "thermo_CO2RR.json or CH3OH_box.cif exists."
        ),
    ),
    "methane": CO2RRProductSpec(
        key="methane",
        state_id="CH4(g)",
        label="CH4(g)",
        gas_key="CH4",
        h2_coeff=4.0,
        h2o_coeff=2.0,
        terminal_states=("CH3*",),
        branch_start_state="CHOH*",
    ),
}


# Full C1 graph.  Product presets below expose induced subgraphs of this
# registry; the competitive preset evaluates the union once so shared states
# such as COOH* and CO* are never recalculated for each product.
CO2RR_NETWORK_EDGES: Tuple[CO2RREdgeSpec, ...] = (
    CO2RREdgeSpec("CO2", "COOH*", "CO2 activation via COOH*", 1),
    CO2RREdgeSpec("COOH*", "CO*", "CO formation and H2O release", 1),
    CO2RREdgeSpec("CO*", "CO(g)", "CO desorption", 0, "desorption", False),
    CO2RREdgeSpec("CO2", "HCOO*", "Bidentate formate formation", 1),
    CO2RREdgeSpec("CO2", "OCHO*", "Monodentate formate formation", 1),
    CO2RREdgeSpec("HCOO*", "HCOOH(g)", "Formic-acid/formate endpoint", 1),
    CO2RREdgeSpec("OCHO*", "HCOOH(g)", "Formic-acid/formate endpoint", 1),
    CO2RREdgeSpec("CO*", "CHO*", "CO hydrogenation via CHO*", 1),
    CO2RREdgeSpec("CHO*", "HCHO(g)", "Formaldehyde release", 1),
    # Neutral formaldehyde is not forced onto the surface.  The retained
    # oxygenate route proceeds through hydroxycarbene/hydroxymethyl states.
    CO2RREdgeSpec("CHO*", "CHOH*", "CHO hydrogenation toward C-O cleavage", 1),
    CO2RREdgeSpec("CHOH*", "CH2OH*", "Hydroxycarbene hydrogenation", 1),
    CO2RREdgeSpec("CH2OH*", "CH3OH(g)", "Methanol release", 1),
    CO2RREdgeSpec("CHOH*", "CH*", "CHOH protonation and H2O elimination", 1),
    CO2RREdgeSpec("CH2OH*", "CH2*", "C-O removal and H2O release", 1),
    CO2RREdgeSpec("CO*", "COH*", "CO protonation via COH*", 1),
    CO2RREdgeSpec("COH*", "CHOH*", "COH hydrogenation", 1),
    CO2RREdgeSpec("CH*", "CH2*", "CH hydrogenation", 1),
    CO2RREdgeSpec("CH2*", "CH3*", "CH2 hydrogenation", 1),
    CO2RREdgeSpec("CH3*", "CH4(g)", "Methane release", 1),
)


def _edge_dict(edge: CO2RREdgeSpec) -> dict:
    return {
        "from": edge.source,
        "to": edge.target,
        "label": edge.label,
        "n_pe": int(edge.n_pe),
        "edge_type": edge.edge_type,
        "potential_sensitive": bool(edge.potential_sensitive),
        "required_site_count": int(edge.required_site_count),
    }


def _pathway_edges(*state_ids: str) -> List[dict]:
    allowed = {str(x) for x in state_ids}
    return [
        _edge_dict(edge)
        for edge in CO2RR_NETWORK_EDGES
        if edge.source in allowed and edge.target in allowed
    ]


_COMPETITIVE_C1_STATES = [
    "COOH*", "CO*", "HCOO*", "OCHO*", "CHO*", "COH*", "CHOH*",
    "CH2OH*", "CH*", "CH2*", "CH3*",
]

CO2RR_PATHWAYS: Dict[str, dict] = {
    "competitive_c1": {
        "label": "Competitive C1 network",
        "description": (
            "Evaluate shared C1 intermediates once and compare CO, formate, "
            "formaldehyde, methanol, and methane endpoint pathways. Neutral "
            "formaldehyde is not forced to remain as CH2O*."
        ),
        "states": list(_COMPETITIVE_C1_STATES),
        "products": ["co", "formate", "formaldehyde", "methanol", "methane"],
        "edges": [_edge_dict(edge) for edge in CO2RR_NETWORK_EDGES],
    },
    "co": {
        "label": "CO pathway",
        "description": "COOH*/CO* entry followed by explicit CO desorption.",
        "states": ["COOH*", "CO*"],
        "products": ["co"],
        "edges": _pathway_edges("CO2", "COOH*", "CO*", "CO(g)"),
    },
    "formate": {
        "label": "Formate pathway",
        "description": (
            "Alternative bidentate HCOO* and monodentate OCHO* routes to a "
            "neutral HCOOH molecular endpoint proxy."
        ),
        "states": ["HCOO*", "OCHO*"],
        "products": ["formate"],
        "state_families": {"FORMATE*": ["HCOO*", "OCHO*"]},
        "edges": _pathway_edges("CO2", "HCOO*", "OCHO*", "HCOOH(g)"),
    },
    "methanol": {
        "label": "Methanol (C1 oxygenate) pathway",
        "description": (
            "CH2O*-free oxygenate route through CHO*/COH*, CHOH*, and "
            "CH2OH*. A missing CH3OH endpoint is reported explicitly."
        ),
        "states": [
            "COOH*", "CO*", "CHO*", "COH*", "CHOH*", "CH2OH*",
        ],
        "products": ["methanol"],
        "edges": _pathway_edges(
            "CO2", "COOH*", "CO*", "CHO*", "COH*", "CHOH*",
            "CH2OH*", "CH3OH(g)",
        ),
    },
    "methane": {
        "label": "Methane (C1 hydrocarbon) pathway",
        "description": (
            "Competing COH* and CHO*-derived deoxygenation routes followed "
            "by C1 hydrogenation."
        ),
        "states": [
            "COOH*", "CO*", "CHO*", "COH*", "CHOH*",
            "CH2OH*", "CH*", "CH2*", "CH3*",
        ],
        "products": ["methane"],
        "edges": _pathway_edges(
            "CO2", "COOH*", "CO*", "CHO*", "COH*", "CHOH*",
            "CH2OH*", "CH*", "CH2*", "CH3*", "CH4(g)",
        ),
    },
}

CO2RR_PATHWAY_ORDER: Tuple[str, ...] = (
    "competitive_c1", "co", "formate", "methanol", "methane", "custom"
)
CO2RR_DEFAULT_PATHWAY = "competitive_c1"

CO2RR_WARNING = (
    "Thermodynamic CO2RR reaction-network screening. PDS and limiting potential "
    "are provisional when only MLIP electronic energies are available; kinetic "
    "barriers, explicit solvent, coverage effects, and microkinetic selectivity "
    "are not included."
)


def clean_co2rr_adsorbate(label: str) -> str:
    return str(label or "").replace("*", "").replace(" ", "").upper()


def normalize_co2rr_state(label: str) -> str:
    key = clean_co2rr_adsorbate(label)
    return f"{key}*" if key else ""



def normalize_co2rr_site_kind(kind: object) -> str:
    """Normalize generic and oxide-specific site labels for adsorbate routing."""
    k = str(kind or "").strip().lower().replace("-", "_")
    aliases = {
        "top": "ontop",
        "atop": "ontop",
        "on_top": "ontop",
        "metal_top": "ontop",
        "cation": "ontop",
        "cation_top": "ontop",
        "fcc": "hollow",
        "hcp": "hollow",
        "threefold": "hollow",
        "metal_bridge": "bridge",
        "pair_bridge": "bridge",
        "cation_bridge": "cation_pair",
        "cation_cation_bridge": "cation_pair",
        "cation_pair_bridge": "cation_pair",
        "metal_pair": "cation_pair",
    }
    return aliases.get(k, k or "ontop")


def get_co2rr_binding_variants(adsorbate: str) -> Tuple[CO2RRBindingVariant, ...]:
    """Return all initial binding variants for one chemical state."""
    spec = get_co2rr_adsorbate_spec(adsorbate)
    return CO2RR_BINDING_VARIANTS.get(spec.key, (_default_binding_variant(spec),))


def get_co2rr_binding_variant(
    adsorbate: str,
    variant_key: str | None = None,
) -> CO2RRBindingVariant:
    """Resolve one binding variant, defaulting to the first registered seed."""
    variants = get_co2rr_binding_variants(adsorbate)
    if variant_key is None:
        return variants[0]
    wanted = str(variant_key).strip().lower()
    for variant in variants:
        if str(variant.key).strip().lower() == wanted:
            return variant
    raise KeyError(
        f"Unsupported binding variant {variant_key!r} for {adsorbate!r}; "
        f"available variants are {[v.key for v in variants]!r}."
    )


def co2rr_site_allowed(
    adsorbate: str,
    site_kind: object,
    binding_variant: CO2RRBindingVariant | None = None,
) -> bool:
    """Return whether a registry adsorbate may be seeded on the site family."""
    spec = get_co2rr_adsorbate_spec(adsorbate)
    kind = normalize_co2rr_site_kind(site_kind)
    allowed_source = (
        binding_variant.allowed_site_kinds
        if binding_variant is not None
        else spec.allowed_site_kinds
    )
    allowed = {normalize_co2rr_site_kind(x) for x in allowed_source}
    return kind in allowed


def co2rr_state_family(label: str) -> str:
    """Collapse alternative starting geometries into one chemical state family."""
    key = clean_co2rr_adsorbate(label)
    if key in {"HCOO", "OCHO"}:
        return "FORMATE"
    return key

def get_co2rr_adsorbate_spec(label: str) -> CO2RRAdsorbateSpec:
    key = clean_co2rr_adsorbate(label)
    if key not in CO2RR_ADSORBATES:
        raise KeyError(f"Unsupported CO2RR adsorbate: {label!r}")
    return CO2RR_ADSORBATES[key]


def is_supported_co2rr_adsorbate(label: str) -> bool:
    return clean_co2rr_adsorbate(label) in CO2RR_ADSORBATES


def get_co2rr_preset(key: str) -> dict:
    k = str(key or CO2RR_DEFAULT_PATHWAY).strip().lower()
    if k == "custom":
        return {
            "label": "Custom intermediate set",
            "description": "User-selected CO2RR C1 intermediates.",
            "states": ["COOH*", "CO*"],
            "products": [],
            "edges": [],
        }
    if k not in CO2RR_PATHWAYS:
        raise KeyError(f"Unsupported CO2RR pathway preset: {key!r}")
    return dict(CO2RR_PATHWAYS[k])


def all_co2rr_states(*, include_legacy: bool = False) -> List[str]:
    """Return screenable states; hide neutral-formaldehyde legacy seeds by default."""
    legacy = {"CH2O", "CH3O"}
    return [
        f"{key}*" for key in CO2RR_ADSORBATES
        if include_legacy or key not in legacy
    ]


def co2rr_state_label(state: str) -> str:
    try:
        return get_co2rr_adsorbate_spec(state).label
    except Exception:
        return str(state)


def co2rr_reference_energy(
    adsorbate: str,
    E_CO2: float,
    E_H2: float,
    E_H2O: float,
) -> tuple[float, str]:
    spec = get_co2rr_adsorbate_spec(adsorbate)
    value = float(E_CO2 + spec.h2_coeff * E_H2 - spec.h2o_coeff * E_H2O)
    parts = ["CO2"]
    if abs(spec.h2_coeff) > 1e-12:
        parts.append(f"+ {spec.h2_coeff:g} H2")
    if abs(spec.h2o_coeff) > 1e-12:
        parts.append(f"- {spec.h2o_coeff:g} H2O")
    return value, " ".join(parts)


def get_co2rr_product_spec(key: str) -> CO2RRProductSpec:
    product_key = str(key or "").strip().lower()
    if product_key not in CO2RR_PRODUCTS:
        raise KeyError(f"Unsupported CO2RR product: {key!r}")
    return CO2RR_PRODUCTS[product_key]


def co2rr_product_reference_energy(
    product: str,
    E_CO2: float,
    E_H2: float,
    E_H2O: float,
) -> tuple[float, str]:
    spec = get_co2rr_product_spec(product)
    value = float(E_CO2 + spec.h2_coeff * E_H2 - spec.h2o_coeff * E_H2O)
    parts = ["CO2"]
    if abs(spec.h2_coeff) > 1e-12:
        parts.append(f"+ {spec.h2_coeff:g} H2")
    if abs(spec.h2o_coeff) > 1e-12:
        parts.append(f"- {spec.h2o_coeff:g} H2O")
    return value, " ".join(parts)


def required_co2rr_products(
    adsorbates: Tuple[str, ...] | List[str],
) -> Tuple[str, ...]:
    """Return product endpoints reachable from the requested intermediate set."""
    requested = {normalize_co2rr_state(x) for x in adsorbates}
    products = []
    for key, spec in CO2RR_PRODUCTS.items():
        if any(normalize_co2rr_state(s) in requested for s in spec.terminal_states):
            products.append(key)
    return tuple(products)


def co2rr_migration_threshold(adsorbate: str, fallback: float = 2.50) -> float:
    try:
        return float(get_co2rr_adsorbate_spec(adsorbate).migration_threshold_A)
    except Exception:
        return float(fallback)
