# Surface Adsorption and Geometry Evaluator (SAGE)

SAGE is a Streamlit-based application for **surface-based electrocatalyst screening**.
It integrates slab preparation, geometry quality control, adsorption-site generation,
ML-assisted candidate screening, thermodynamic post-processing, and result export within a
**single user-facing workflow**.

The main contribution of SAGE is **workflow integration**, not the introduction of a new atomistic engine.
Many surface-screening failures arise not from the calculator alone, but from inconsistent slab setup,
site generation, post-relaxation interpretation, and result handling.
SAGE was designed to reduce that fragmentation and make the full screening route more reproducible,
inspectable, and easier to operate.

---

## Version focus: SAGE v1.1.0

SAGE v1.1.0 extends the original HER-oriented workflow with a **confidence-aware OER screening branch**.
The update separates metal and oxide OER workflows because clean metallic surfaces and metal-oxide surfaces
require different site taxonomies and different interpretation rules.

The v1.1.0 update includes:

- **metal-OER oxygen-intermediate screening** using OH\*, O\*, and OOH\* descriptors;
- **oxide-OER cation-bound AEM screening** using `oer_cation` sites;
- **OER-specific oxide slab handling** that separates HER-oriented O-exposed slabs from OER-oriented cation-exposed slabs;
- **same-site OH/O/OOH triplet grouping** for OER summary generation;
- **confidence-aware OER outputs**, including explicit, scaling-proxy, and recommended OER descriptors;
- **benchmark tables** for metal and oxide OER screening.

SAGE v1.1.0 should be interpreted as a **rapid, confidence-aware screening workflow**, not as a replacement for
constant-potential DFT, explicit-solvent calculations, or final experimental validation.

---

## Why SAGE?

Surface-adsorption screening is often performed through loosely connected scripts for:

- structure input and slab generation;
- supercell and vacuum handling;
- adsorption-site generation;
- pre-screening and relaxation;
- thermodynamic correction;
- result ranking and export.

In practice, this fragmentation can lead to inconsistency in:

- slab construction;
- adsorption-site definition;
- geometry QC;
- relaxed-site interpretation;
- metadata tracking;
- export formatting.

SAGE addresses this by providing a single application that organizes the main screening steps into a consistent workflow.

### Workflow-level contributions

SAGE emphasizes **workflow-level novelty** rather than a new calculator:

- **single-interface execution** of the main surface-screening steps;
- **state-guided slab preparation** through a stepwise surface-setup wizard;
- **built-in geometry QC** before interpretation;
- **migration-aware output interpretation**;
- **oxide-aware HER handling**;
- **OER-specific metal/oxide site handling**;
- **metadata-aware export**;
- **deterministic seed control** for reproducibility.

This positioning is important because many practical failures in catalyst screening arise from inconsistent
preprocessing and interpretation rather than from the energy model alone.

---

## Current Scope

### Supported material classes

- Metals
- Selected oxide surfaces

### Supported reaction modes

| Reaction | Adsorbates / descriptors | Current interpretation |
|----------|---------------------------|------------------------|
| HER (metal) | H\* | Conventional ΔG_H with CHE-style correction |
| HER (oxide) | O-top H / protonated surface O / reactive-H descriptor states | Oxide-oriented protonation / relaxed-state descriptor workflow |
| CO2RR | COOH\*, CO\*, HCOO\*, OCHO\* | ΔG_ads vs reference states; experimental |
| OER (metal) | OH\*, O\*, OOH\* | Nørskov-type oxygen-intermediate screening on metal surfaces |
| OER (oxide) | cation-bound OH\*, O\*, OOH\* | Confidence-aware cation-bound AEM screening; termination-sensitive |

### Current practical emphasis

- **Metal HER** remains the most benchmark-friendly HER mode.
- **Metal OER** can be compared directly against Nørskov-type OH\*/O\* adsorption benchmarks.
- **Oxide HER** is supported as a structured protonation/relaxed-state descriptor workflow.
- **Oxide OER** is supported as a confidence-aware cation-bound screening workflow, not as a universal oxide OER predictor.

---

## SAGE App Interface

<img width="1191" height="720" alt="Image" src="https://github.com/user-attachments/assets/09a38cb8-541e-4dd3-be29-2a29a4a32b29" />

---

## Installation

```bash
git clone https://github.com/Junhepark/Surface-Adsorption-and-Geometry-Evaluator-SAGE-.git
cd Surface-Adsorption-and-Geometry-Evaluator-SAGE-
pip install -r requirements.txt
```

### Optional model setup for CHE evaluation

The CHE evaluation stage uses Meta's UMA model as the default energy calculator.

To enable UMA-based evaluation:

1. Create a Hugging Face account.
2. Request access to the UMA model.
3. Log in locally.

```bash
pip install huggingface_hub
huggingface-cli login
```

The model weights will be downloaded automatically on first use.

---

## Running the App

```bash
streamlit run app/Home.py
```

---

## Workflow Overview

The current interface is organized as a stepwise workflow:

### 1. Load structure

Load a structure from:

- a Materials Project ID;
- an uploaded CIF file.

### 2. Surface setup

Prepare the working slab through a staged wizard:

- slab selection;
- vacuum setup;
- XY supercell expansion;
- slab-thickness reduction;
- prepared-slab review.

This design reduces the chance of skipping critical slab-preparation steps.

### 3. Site selection

Generate adsorption candidates from the prepared slab.

Depending on the reaction mode, SAGE supports:

- geometry-based site generation;
- optional ML-assisted pre-screening;
- oxide-specific HER site logic;
- oxide-specific OER `oer_cation` logic.

### 4. Run calculation

Run the selected thermodynamic workflow:

- HER;
- CO2RR;
- OER.

### 5. Review and export

Inspect:

- ranked outputs;
- relaxed structures;
- site migration behavior;
- OER summaries;
- metadata-rich CSV exports;
- CIF exports for downstream checking.

---

## Quick Start

### 1. Load a structure

Enter a Materials Project ID, for example `mp-23` for Ni, or upload a `.cif` file.

<img width="1852" height="1102" alt="Image" src="https://github.com/user-attachments/assets/4de16c09-e084-4c75-9049-283be93e4533" />

### 2. Choose material type and reaction mode

Select:

- **Metal / Oxide**
- **HER / CO2RR / OER**

### 3. Prepare the slab

Use the surface-setup wizard to define:

- slab candidate;
- vacuum thickness;
- XY supercell;
- slab thickness reduction.

### 4. Generate adsorption candidates and evaluate

Run site generation and the selected reaction workflow.

<img width="1280" height="634" alt="Image" src="https://github.com/user-attachments/assets/538d0ffa-b14f-47d4-a237-5b21897a14c5" />

### 5. Review outputs and export

Inspect ranked candidates, relaxed structures, and exported metadata tables.

<img width="1282" height="502" alt="Image" src="https://github.com/user-attachments/assets/af16bec9-b26e-4b09-9001-6e8c4c0c3c4e" />

---

## Supported Reactions

| Reaction | Adsorbates / descriptors | Current interpretation |
|----------|---------------------------|------------------------|
| HER (metal) | H\* | Conventional ΔG_H with CHE-style correction |
| HER (oxide) | O-top H / protonated surface O / reactive-H descriptor states | Oxide-oriented protonation / relaxed-state descriptor workflow |
| CO2RR | COOH\*, CO\*, HCOO\*, OCHO\* | ΔG_ads vs reference states; experimental |
| Metal OER | metal OH\* / O\* / OOH\* sites | Nørskov-type metal benchmark |
| Oxide OER | `oer_cation` OH\* / O\* / OOH\* triplets | confidence-aware oxide screening |

---

## How SAGE Treats Metals vs Oxides

### Metallic surfaces

Metal surfaces are the most direct benchmark target.
For metallic HER, SAGE is intended to recover physically reasonable ΔG_H trends on representative close-packed surfaces.
For metallic OER, SAGE evaluates OH\*, O\*, and OOH\* oxygen-intermediate descriptors and can be compared against Nørskov-type ΔG_OH / ΔG_O references.

### Oxide surfaces

Oxide surfaces are more complex because their behavior depends strongly on:

- surface termination;
- cation/anion exposure;
- local stoichiometry;
- slab polarity;
- post-relaxation migration;
- protonated / OH-like final states.

For this reason, SAGE does **not** treat oxide HER or oxide OER as generic metal-like adsorption problems.

### Oxide HER

The oxide HER workflow uses O-top and protonation-oriented logic.
It is designed as an oxide-oriented screening route in which hydrogen is preferentially seeded using an O-top rule and relaxed structures are interpreted with attention to surface protonation and OH-like states.

### Oxide OER

The oxide OER branch uses a separate cation-bound AEM logic.
The OER strict mode does not use the metallic `fcc`, `hcp`, or generic `hollow` site taxonomy for oxide benchmarks.
Instead, it uses `oer_cation` sites to generate same-site OH/O/OOH triplets.

Practical summary:

| Surface class | HER interpretation | OER interpretation |
|---------------|-------------------|-------------------|
| Metal | H\* adsorption on metal sites | metal OH\* / O\* / OOH\* descriptors |
| Oxide | O-top / protonation-oriented descriptors | cation-bound `oer_cation` OH/O/OOH triplets |

---

## OER Descriptor Interpretation

SAGE reports OER descriptors using a stepwise CHE-style framework.
For a valid same-site OH/O/OOH triplet, the OER steps are interpreted as:

```text
ΔG1 = ΔG_*OH
ΔG2 = ΔG_*O - ΔG_*OH
ΔG3 = ΔG_*OOH - ΔG_*O
ΔG4 = 4.92 - ΔG_*OOH
η_OER = max(ΔG1, ΔG2, ΔG3, ΔG4) - 1.23
```

For oxide OER, SAGE additionally reports confidence-aware fields:

- `η_OER_explicit`;
- `η_OER_scaling_proxy`;
- `η_OER_recommended`;
- representative OER site ranking.

These values should be interpreted as screening descriptors, not as direct replacements for full DFT or experimental OER overpotentials.

---

## Key Features

- **Structure input**
  - Materials Project retrieval
  - CIF upload

- **Surface setup wizard**
  - slab selection
  - vacuum control
  - XY expansion
  - slab-thickness reduction
  - prepared-slab review

- **Geometry QC**
  - pairwise distance checks
  - isolated-atom detection
  - vacuum validation

- **Adsorption-site generation**
  - metallic ontop / bridge / hollow logic
  - oxide-oriented HER site treatment
  - oxide-oriented OER `oer_cation` treatment

- **ML pre-screening**
  - optional CHGNet-assisted filtering

- **Thermodynamic workflows**
  - HER
  - CO2RR
  - OER

- **OER-specific summary handling**
  - same-site OH/O/OOH triplet grouping
  - explicit/scaling/recommended OER descriptors
  - representative site ranking

- **Migration-aware interpretation**
  - initial vs relaxed site behavior
  - reliable vs unreliable output separation

- **Relaxed-structure inspection**
  - post-run structure viewing
  - CIF download

- **Metadata-rich export**
  - ranked CSV outputs
  - reproducible result tracking

- **LLM-ready reporting support**
  - optional structured summarization of ranked outputs

---

## What Is Novel in SAGE?

The novelty of SAGE is not that it introduces a completely new atomistic calculator.
Its main novelty is that it **turns fragmented script-level tools into an executable research application**.

In practical terms, SAGE integrates:

1. structure retrieval or upload;
2. slab preparation;
3. geometry QC;
4. adsorption-site generation;
5. ML-assisted pre-screening;
6. CHE-based post-processing;
7. relaxed-site interpretation;
8. ranked export with metadata;
9. oxide-oriented HER treatment;
10. confidence-aware OER triplet screening;
11. user-facing review within one interface.

This matters because adsorption screening is often limited not only by the underlying calculator,
but by inconsistency between intermediate workflow steps.
SAGE is intended to reduce that inconsistency through a more structured interface.

---

## Benchmarking: Metallic HER

To test whether the workflow reproduces physically reasonable hydrogen-adsorption trends on representative metal surfaces,
SAGE-calculated ΔG_H values were compared against literature reference values in the Nørskov framework.

| Metal surface | Calculated ΔG_H (eV) | Nørskov θ = 0.25 (eV) | Difference (eV) | MP ID |
|---------------|----------------------:|------------------------:|----------------:|-------|
| Ni(111) | -0.255 | -0.27 | +0.015 | `mp-23` |
| Co(111) | -0.271 | -0.27 | -0.001 | `mp-102` |
| Pt(111) | -0.100 | -0.09 | -0.010 | `mp-126` |
| Pd(111) | -0.168 | -0.14 | -0.028 | `mp-2` |
| Rh(111) | -0.117 | -0.10 | -0.017 | `mp-74` |
| Cu(111) | +0.178 | +0.19 | -0.012 | `mp-30` |
| Ag(111) | +0.555 | +0.51 | +0.045 | `mp-124` |

Across these surfaces, the mean absolute deviation is approximately **0.018 eV**.

This benchmark is not intended to claim a new state-of-the-art calculator.
Instead, it supports the practical claim that SAGE can recover established metallic screening trends within an integrated workflow.

---

## Benchmarking: Metallic OER Oxygen Intermediates

The metal-OER branch was benchmarked against Nørskov-type OH\*/O\* adsorption free energies on close-packed metal surfaces.
This benchmark is a descriptor-level comparison and should not be interpreted as a complete experimental OER activity ranking for metallic electrodes.

| Metal surface | MP ID | ΔG_OH, lit. (eV) | ΔG_OH, SAGE (eV) | ΔG_O, lit. (eV) | ΔG_O, SAGE (eV) | η_OER, SAGE (V) | Status |
|---------------|-------|------------------:|------------------:|----------------:|----------------:|-----------------:|--------|
| Au(111) | `mp-81` | 1.84 | 1.885 | 2.80 | 2.885 | 0.897 | pass |
| Ag(111) | `mp-124` | 1.07 | 1.186 | 2.17 | 2.285 | 1.014 | pass |
| Pd(111) | `mp-2` | 1.27 | 1.476 | 1.58 | 1.485 | 1.766 | pass |
| Pt(111) | `mp-126` | 1.40 | 1.983 | 1.62 | 1.811 | 1.454 | caution; OH\* overestimated |
| Rh(111) | `mp-74` | 0.69 | 0.909 | 0.49 | 0.733 | 2.149 | pass |
| Ir(111) | `mp-101` | 0.98 | 1.398 | 1.05 | 1.063 | 2.035 | pass / caution |
| Ni(111) | `mp-23` | 0.48 | 0.670 | 0.39 | 0.329 | 1.471 | site-sensitive |
| Cu(111) | `mp-30` | 0.72 | 0.846 | 1.25 | 1.290 | — | pass |

The primary purpose of this table is to show that SAGE can rapidly reproduce Nørskov-type metal oxygen-intermediate trends within an integrated workflow.
Pt and Ni are retained in the table because they are informative site-sensitivity cases, not because they should be hidden from the benchmark.

---

## Benchmarking: Oxide OER Screening

The oxide-OER branch was evaluated using cation-bound OH\*/O\*/OOH\* triplets.
Unlike the metal benchmark, the oxide table is not a direct Nørskov metal-surface comparison.
It is a confidence-aware oxide screening benchmark.

| Oxide | MP ID | Phase / facet | Representative site | ΔG_OH, SAGE (eV) | ΔG_O, SAGE (eV) | η_OER, SAGE (V) | PDS | Benchmark role |
|-------|-------|---------------|---------------------|-----------------:|---------------:|----------------:|-----|----------------|
| RuO₂ | `mp-825` | rutile (110) | `oer_cation_0_Ru58` | 0.458 | 2.038 | 0.577 | *O → *OOH | strong positive |
| Co₃O₄ | `mp-18748` | spinel (111) | `oer_cation_0_Co0` | 0.561 | 1.742 | 0.714 | *O → *OOH | non-PGM active oxide |
| NiO | `mp-19009` | rock-salt, likely (100) | `oer_cation_0_Ni65` | 1.372 | 3.419 | 0.817 | *OH → *O | precursor-like moderate oxide |
| IrO₂ | `mp-2723` | rutile (110) | `oer_cation_2_Ir2` | 0.091 | 2.174 | 0.854 | *OH → *O | noble oxide positive, conservative |
| TiO₂ | `mp-2657` | rutile (110) | `oer_cation_1_Ti10` | 1.228 | 3.409 | 0.951 | *OH → *O | weak/moderate rutile control |
| Fe₂O₃ | `mp-24972` | hematite (001) | `oer_cation_1_Fe7` | 1.397 | 3.870 | 1.243 | *OH → *O | weak oxide control |

The oxide benchmark supports the following qualitative trend:

```text
RuO₂ < Co₃O₄ < NiO ≈ IrO₂ < TiO₂ < Fe₂O₃
```

NiO is included as a precursor-like Ni-based oxide control.
Its clean-surface value should not be interpreted as a direct representation of reconstructed NiOOH or NiFeOOH active phases.

---

## How SAGE Approaches Metal-Oxide Surfaces

Metal-oxide surfaces are intrinsically harder than close-packed metals because adsorption energetics are more sensitive to:

- termination;
- local stoichiometry;
- cation/anion exposure;
- polarity;
- relaxed-state migration;
- protonated oxygen / OH-like final states.

For this reason, SAGE does **not** position oxide outputs as universal benchmark values.

Instead, oxide calculations in the current release should be viewed as:

- **structured screening workflows**;
- **inspectable case-study outputs**;
- **termination-sensitive results requiring user judgment**.

This distinction is intentional.

- **Metals** are used for direct descriptor-level benchmark comparisons.
- **Oxides** are used for confidence-aware screening and workflow demonstration.

---

## Example Interpretation Strategy

### For metals

Use ΔG_H, ΔG_OH, and ΔG_O trends to judge whether the workflow reproduces physically reasonable screening behavior.

### For oxides

Use SAGE to:

- generate candidate surfaces;
- perform geometry QC;
- inspect relaxed-site behavior;
- compare relative site or facet trends;
- export structures and ranked outputs for further validation.

In oxide mode, final interpretation should always retain explicit user judgment about:

- termination validity;
- protonation or cation-bound adsorption realism;
- whether the relaxed state is chemically meaningful.

---

## Dependencies

- Python >= 3.10
- ASE
- pymatgen
- CHGNet
- PyTorch
- Streamlit
- py3Dmol
- NumPy
- pandas

See `requirements.txt` for the full dependency list.

---

## Limitations

- CO2RR support is still experimental.
- Oxide outputs remain more termination-sensitive than metallic benchmarks.
- Oxide HER values should not be interpreted as one-to-one replacements for close-packed metallic ΔG_H benchmarks.
- Oxide OER values should be interpreted as confidence-aware screening descriptors, not universal OER overpotential predictions.
- OER scaling-proxy values can produce false-positive behavior in some weak oxides.
- Some oxide surfaces may fail to form valid cation-bound OH/O/OOH triplets under strict AEM mode.
- D2-only oxide results are user-facing representative values, not universal oxide benchmarks.
- Optional local vibrational refinement is a local correction mode, not a full phonon treatment.
- Some oxide systems still require additional user validation after slab construction and relaxation.
- Runtime can increase significantly for large supercells.

---

## Citation

If you use SAGE in your research, please cite the software repository.

```text
Junhe Park. Surface Adsorption and Geometry Evaluator (SAGE), v1.1.0. GitHub repository.
```

A manuscript citation will be added here after publication.

---

## Acknowledgements

This research was supported by the National Research Foundation of Korea (NRF) funded by the Ministry of Education  
(Grant No. NRF-2023R1A2C2003796; NRF-2020R1A6A1A03042742).

---

## License

This project is distributed under the **AGPL-3.0** license.
