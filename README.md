# Surface Adsorption and Geometry Evaluator (SAGE)

SAGE is a Streamlit-based application for **surface-based electrocatalyst screening**.
It integrates slab preparation, geometry quality control, adsorption-site generation,
ML-assisted candidate screening, thermodynamic post-processing, and result export
within a **single user-facing workflow**.

The main contribution of SAGE is **workflow integration**, not the introduction of a new atomistic engine.
Many surface-screening failures arise not from the calculator alone, but from inconsistent slab setup,
site generation, post-relaxation interpretation, and result handling.
SAGE was designed to reduce that fragmentation and make the full screening route more reproducible,
inspectable, and easier to operate.

---

## Why SAGE?

Surface-adsorption screening is often performed through loosely connected scripts for:

- structure input and slab generation
- supercell and vacuum handling
- adsorption-site generation
- pre-screening and relaxation
- thermodynamic correction
- result ranking and export

In practice, this fragmentation can lead to inconsistency in:

- slab construction
- adsorption-site definition
- geometry QC
- relaxed-site interpretation
- metadata tracking
- export formatting

SAGE addresses this by providing a single application that organizes the main screening steps into a
consistent workflow.

### Workflow-level contributions

SAGE emphasizes **workflow-level novelty** rather than a new calculator:

- **single-interface execution** of the main surface-screening steps
- **state-guided slab preparation** through a stepwise surface-setup wizard
- **built-in geometry QC** before interpretation
- **migration-aware output interpretation**
- **oxide-aware HER handling**
- **metadata-aware export**
- **deterministic seed control** for reproducibility

This positioning is important because many practical failures in catalyst screening arise from
inconsistent preprocessing and interpretation rather than from the energy model alone.

---

## Current Scope

### Supported material classes
- Metals
- Selected oxide surfaces

### Supported reaction modes
- **HER**
- **CO2RR**
- **ORR** *(experimental / limited validation)*

### Current practical emphasis
The current release is focused primarily on **surface-based screening workflows** for HER and related
adsorption-energy analysis.

- **Metal HER** is the most benchmark-friendly mode in the current release.
- **Oxide HER** is supported as a **structured screening and interpretation workflow**, not as a universal
  black-box benchmark for all oxide terminations.

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

1. Create a Hugging Face account
2. Request access to the UMA model
3. Log in locally

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
- a Materials Project ID
- an uploaded CIF file

### 2. Surface setup
Prepare the working slab through a staged wizard:

- slab selection
- vacuum setup
- XY supercell expansion
- slab-thickness reduction
- prepared-slab review

This design reduces the chance of skipping critical slab-preparation steps.

### 3. Site selection
Generate adsorption candidates from the prepared slab.

Depending on the reaction mode, SAGE supports:
- geometry-based site generation
- optional ML-assisted pre-screening
- oxide-specific HER site logic

### 4. Run calculation
Run the selected thermodynamic workflow:
- HER
- CO2RR
- ORR (experimental)

### 5. Review and export
Inspect:
- ranked outputs
- relaxed structures
- site migration behavior
- metadata-rich CSV exports
- CIF exports for downstream checking

---

## Quick Start

### 1. Load a structure
Enter a Materials Project ID (for example, `mp-23` for Ni) or upload a `.cif` file.

<img width="1852" height="1102" alt="Image" src="https://github.com/user-attachments/assets/4de16c09-e084-4c75-9049-283be93e4533" />

### 2. Choose material type and reaction mode
Select:
- **Metal / Oxide**
- **HER / CO2RR / ORR**

### 3. Prepare the slab
Use the surface-setup wizard to define:
- slab candidate
- vacuum thickness
- XY supercell
- slab thickness reduction

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
| HER (metal) | H* | Conventional ΔG_H with CHE-style correction |
| HER (oxide) | O-top H / protonated surface O / reactive-H descriptor states | Oxide-oriented protonation / relaxed-state descriptor workflow |
| CO2RR | COOH*, CO*, HCOO*, OCHO* | ΔG_ads vs reference states |
| ORR* | OOH*, O*, OH* | 4e- Nørskov CHE framework |

\* ORR support is currently experimental and should not yet be treated as the primary validated mode of the software.

---

## How SAGE Treats Metals vs Oxides

### Metallic surfaces
Metal surfaces are the most direct benchmark target in the current release.
For metallic HER, SAGE is intended to recover physically reasonable ΔG_H trends on representative
close-packed surfaces.

### Oxide surfaces
Oxide surfaces are more complex because their behavior depends strongly on:

- surface termination
- cation/anion exposure
- local stoichiometry
- slab polarity
- post-relaxation migration
- protonated / OH-like final states

For this reason, SAGE does **not** treat oxide HER as a generic metal-like adsorption problem.

Instead, the current oxide HER workflow is designed as an **oxide-oriented screening route** in which:
- hydrogen is preferentially seeded using an **O-top protonation-oriented rule**
- relaxed structures are interpreted with attention to **surface protonation and OH-like states**
- outputs are meant to support **case-study-style oxide analysis**, not universal oxide benchmarking

### Practical implication
- **Metals** → benchmark-friendly
- **Oxides** → termination-sensitive, case-study-oriented, interpretation-heavy

---

## Oxide HER Output Interpretation

For oxide HER mode, the most useful interpretation is usually not a direct metallic-style volcano comparison.

Instead, the current implementation is better viewed as a descriptor of:

- how readily surface oxygen accepts hydrogen
- whether the relaxed structure tends toward an OH-like protonated state
- how strongly that protonated / reactive-H state is stabilized after relaxation

### Representative display logic in oxide HER
SAGE distinguishes between different user-facing values for oxide HER outputs.

- **Representative occupied site**  
  The most stabilized reliable H* site among reliable candidates  
  (`minimum ΔG_H`)

- **HER-optimal reference**  
  The reliable site closest to `0 eV`  
  (`minimum |ΔG_H|`)

- **Selected descriptor profile**  
  The profile-linked descriptor view used for multi-stage interpretation

### D2-only mode
When **`D2_Hreact only (reactive H state)`** is selected, the displayed `D2_Hreact` result is aligned to the
**representative occupied site** among reliable candidates.
This avoids confusion between:
- the best reliable occupied site
- a profile-linked internal descriptor value

This distinction is especially important for oxide surfaces where descriptor profiles and relaxed final states
do not always collapse to a single obvious representative number.

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

- **ML pre-screening**
  - optional CHGNet-assisted filtering

- **Thermodynamic workflows**
  - HER
  - CO2RR
  - ORR (experimental)

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

1. structure retrieval or upload
2. slab preparation
3. geometry QC
4. adsorption-site generation
5. ML-assisted pre-screening
6. CHE-based post-processing
7. relaxed-site interpretation
8. ranked export with metadata
9. oxide-oriented HER treatment
10. user-facing review within one interface

This matters because adsorption screening is often limited not only by the underlying calculator,
but by inconsistency between intermediate workflow steps.
SAGE is intended to reduce that inconsistency through a more structured interface.

---

## Benchmarking on Metallic Surfaces

To test whether the workflow reproduces physically reasonable adsorption trends on representative metal surfaces,
SAGE-calculated ΔG_H values were compared against literature reference values in the Nørskov framework.

| Metal surface | Calculated ΔG_H (eV) | Norskov θ = 0.25 (eV) | Difference (eV) | mp-number |
|---------------|----------------------|------------------------|-----------------|----------|
| Ni(111) | -0.255 | -0.27 | +0.015 | mp-23 |
| Co(111) | -0.271 | -0.27 | -0.001 | mp-102 |
| Pt(111) | -0.100 | -0.09 | -0.010 | mp-126 |
| Pd(111) | -0.168 | -0.14 | -0.028 | mp-2 |
| Rh(111) | -0.117 | -0.10 | -0.017 | mp-74 |
| Cu(111) | +0.178 | +0.19 | -0.012 | mp-30 |
| Ag(111) | +0.555 | +0.51 | +0.045 | mp-124 |

Across these surfaces, the mean absolute deviation is approximately **0.018 eV**.

This benchmark is not intended to claim a new state-of-the-art calculator.
Instead, it supports the practical claim that **SAGE can recover established metallic screening trends within an integrated workflow**.

---

## How SAGE Approaches Metal-Oxide Surfaces

Metal-oxide surfaces are intrinsically harder than close-packed metals because adsorption energetics are more
sensitive to:

- termination
- local stoichiometry
- cation/anion exposure
- polarity
- relaxed-state migration
- protonated oxygen / OH-like final states

For this reason, SAGE does **not** position oxide outputs as universal benchmark values.

Instead, oxide calculations in the current release should be viewed as:

- **structured screening workflows**
- **inspectable case-study outputs**
- **termination-sensitive results requiring user judgment**

This distinction is intentional.

- **Metals** are used as the main benchmark set.
- **Oxides** are better treated as case studies or workflow demonstrations.

---

## Example Interpretation Strategy

A practical interpretation strategy is:

### For metals
Use ΔG_H trends to judge whether the workflow reproduces physically reasonable screening behavior.

### For oxides
Use SAGE to:
- generate candidate surfaces
- perform geometry QC
- inspect relaxed-site behavior
- compare relative site or facet trends
- export structures and ranked outputs for further validation

In oxide mode, final interpretation should always retain explicit user judgment about:
- termination validity
- protonation realism
- whether the relaxed state is chemically meaningful

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

- ORR support is still experimental
- CO2RR support is still experimental
- oxide outputs remain more termination-sensitive than metallic benchmarks
- oxide HER values should not be interpreted as one-to-one replacements for close-packed metallic ΔG_H benchmarks
- D2-only oxide results are user-facing representative values, not universal oxide benchmarks
- optional local vibrational refinement is a local correction mode, not a full phonon treatment
- some oxide systems still require additional user validation after slab construction and relaxation
- runtime can increase significantly for large supercells

---

## Citation

If you use SAGE in your research, please cite the software repository.

```text
Junhe Park. Surface Adsorption and Geometry Evaluator (SAGE), v1.0.0. GitHub repository.
```

A manuscript citation will be added here after publication.

---

## Acknowledgements

This research was supported by the National Research Foundation of Korea (NRF) funded by the Ministry of Education  
(Grant No. NRF-2023R1A2C2003796; NRF-2020R1A6A1A03042742).

---

## License

This project is distributed under the **AGPL-3.0** license.
