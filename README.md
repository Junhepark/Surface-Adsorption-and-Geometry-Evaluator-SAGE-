# Surface Adsorption and Geometry Evaluator (SAGE)

Interactive workflow for slab preparation, geometric quality control, adsorption-site generation, and ML-assisted electrocatalyst screening.

SAGE was developed to reduce the fragmentation of conventional script-based surface-screening workflows, where structure preparation, adsorption-site generation, screening, thermodynamic analysis, and result export are often handled separately.  
It integrates these steps into a single Streamlit-based interface for more consistent, reproducible, and interpretable surface-based electrocatalyst screening.

---

## Why SAGE?

Surface-adsorption screening is often carried out through a sequence of loosely connected scripts.  
In practice, this can lead to inconsistency in:

- slab construction
- adsorption-site definition
- geometric quality control
- thermodynamic post-processing
- metadata tracking and export

SAGE was designed to address this workflow fragmentation by combining structure input, slab preparation, QC, adsorption-site generation, screening, post-processing, and ranked export in one interface.

Rather than introducing a completely new atomistic engine, SAGE emphasizes **workflow-level advancement**:

- **single-interface execution** of the main screening steps
- **built-in geometric QC** before final interpretation
- **migration-aware result interpretation**
- **deterministic seed control** for reproducibility
- **metadata-aware CSV export** for downstream analysis

This positioning is important because many practical failures in catalyst screening do not arise from the calculator alone, but from inconsistency in preprocessing, site generation, post-relaxation interpretation, and result organization.

---

## Current Scope

### Supported material classes
- Metals
- Selected oxide surfaces

### Supported reaction modes
- **HER**
- **CO2RR**
- **ORR** *(experimental; still under development)*

### Current focus
The current release is designed primarily for **surface-based electrocatalyst screening**, with particular emphasis on adsorption-energy workflows for HER and related surface intermediates.

---

## SAGE App Interface

<img width="1280" height="543" alt="SAGE main interface" src="https://github.com/user-attachments/assets/1b3de53e-4bc5-4b19-abb9-3ae9f91aff15" />

---

## Installation

```bash
git clone https://github.com/Junhepark/Surface-Adsorption-and-Geometry-Evaluator-SAGE-.git
cd Surface-Adsorption-and-Geometry-Evaluator-SAGE-
pip install -r requirements.txt
```

### Optional model setup for CHE calculation

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

If other models are preferred, the model selection can be modified in the relevant configuration file before running the application.

---

## Usage

```bash
streamlit run app/Home.py
```

The application will open in your default browser.

---

## Quick Start

### 1. Load a structure
Enter a Materials Project ID (e.g., `mp-23` for Ni) or upload a `.cif` file.

<img width="2490" height="883" alt="Structure input" src="https://github.com/user-attachments/assets/60581ebf-e64c-42fe-baf5-ec308757d9a8" />

### 2. Select the system type and reaction mode
Choose the material type (**Metal / Oxide**) and target reaction (**HER / CO2RR / ORR**).

### 3. Prepare the slab
Set slab-related parameters such as supercell expansion, vacuum thickness, and composition-related options.

<img width="3133" height="1104" alt="Slab preparation" src="https://github.com/user-attachments/assets/9440ea60-9aca-4412-b9d1-e3199e97a5ee" />

### 4. Generate adsorption sites and evaluate candidates
Generate candidate sites using geometry-based logic or CHGNet-assisted screening, then run CHE-based evaluation.

<img width="3328" height="1651" alt="Site generation and CHE calculation" src="https://github.com/user-attachments/assets/91f704a4-cc05-447a-bb92-5e17836559a1" />

### 5. Review ranked outputs and export results
Inspect ranked candidates and export metadata-rich CSV files.

<img width="3324" height="1046" alt="Ranked results and export" src="https://github.com/user-attachments/assets/c32076e5-d336-4072-a008-16bba958f486" />

---

## Supported Reactions

| Reaction | Adsorbates | Thermodynamic framework |
|----------|------------|-------------------------|
| HER | H* | ΔG_H (CHE) |
| CO2RR | COOH*, CO*, HCOO*, OCHO* | ΔG_ads vs CO2/H2O/H2 |
| ORR* | OOH*, O*, OH* | 4e- Norskov CHE |

\* ORR support is currently experimental and should not yet be treated as the primary validated mode of the software.

---

## Key Features

- **Structure input**  
  `.cif` upload or Materials Project retrieval

- **Surface preparation**  
  supercell expansion, composition tuning, and vacuum control

- **Geometric quality control**  
  pairwise distance check, isolated-atom detection, and vacuum validation

- **Adsorption-site generation**  
  ontop / bridge / hollow-type site generation for metallic and oxide surfaces with PBC-aware deduplication

- **ML pre-screening**  
  optional CHGNet-based candidate filtering

- **CHE evaluation**  
  ZPE/entropic corrections, pH/potential adjustments, and migration-aware interpretation

- **Reproducibility support**  
  deterministic seed control for `numpy`, `random`, and `torch`

- **Result management**  
  session history, metadata export, and CSV download

- **LLM-ready interpretation support**  
  optional structured summarization of ranked outputs

---

## What Is Novel in SAGE?

The main contribution of SAGE is not simply that it performs adsorption-energy screening, but that it **organizes the full user-facing screening workflow into a reproducible and interpretable sequence**.

Its novelty lies in the integration of:

1. **structure retrieval or upload**
2. **slab preparation**
3. **surface QC**
4. **adsorption-site generation**
5. **ML-assisted pre-screening**
6. **CHE-based post-processing**
7. **migration-aware interpretation**
8. **ranked export with metadata**

This matters because adsorption screening is often limited not only by the underlying calculator, but also by poor consistency between the intermediate workflow steps.  
SAGE aims to reduce that inconsistency by enforcing a more unified workflow.

In this sense, SAGE should be viewed as a **workflow-level application for surface-screening standardization**, rather than merely a thin wrapper around an energy model.

---

## Benchmarking on Metallic Surfaces

To test whether the workflow yields physically reasonable adsorption trends on well-known metal surfaces, SAGE-calculated ΔG_H values were compared against representative literature values from the Norskov framework.

| Metal surface | Calculated ΔG_H (eV) | Norskov θ = 0.25 (eV) | Difference (eV) |
|---------------|----------------------|------------------------|-----------------|
| Ni(111) | -0.255 | -0.27 | +0.015 |
| Co(111) | -0.271 | -0.27 | -0.001 |
| Pt(111) | -0.100 | -0.09 | -0.010 |
| Pd(111) | -0.168 | -0.14 | -0.028 |
| Rh(111) | -0.117 | -0.10 | -0.017 |
| Cu(111) | +0.178 | +0.19 | -0.012 |
| Ag(111) | +0.555 | +0.51 | +0.045 |

Across these benchmark surfaces, the mean absolute deviation is approximately **0.018 eV**.  
This level of agreement suggests that the SAGE workflow can recover the expected relative adsorption trends for common metallic HER benchmarks while keeping the entire structure-to-ranking pipeline within one interface.

This benchmark is not intended to claim a new state-of-the-art calculator.  
Instead, it supports the more practical claim that **SAGE can reproduce well-established metallic screening trends within an integrated and reproducible workflow**.

---

## How SAGE Approaches Metal-Oxide Surfaces

Metal-oxide surfaces are intrinsically more difficult than close-packed metals because their adsorption energetics are more sensitive to:

- surface termination
- local stoichiometry
- cation/anion exposure
- slab polarity
- post-relaxation migration of adsorbates

For this reason, SAGE does **not** position oxide screening as a fully black-box problem.

Instead, the current oxide workflow is designed as a **structured and inspectable screening route**, where the user can:

- explicitly choose oxide mode
- construct oxide slabs under user-controlled settings
- generate adsorption candidates on oxide surfaces within the same interface
- apply geometric QC before interpretation
- inspect final relaxed-site behavior rather than relying only on the initial site label
- export ranked outputs for further verification

This is an important distinction.

For metals, SAGE benchmarking can be discussed in terms of agreement with established ΔG_H trends.  
For oxides, the main contribution at the current stage is **workflow support for controlled screening and interpretation**, especially in systems where termination and post-relaxation behavior strongly affect the final result.

Accordingly, oxide outputs should be interpreted as **curated screening results within a controlled workflow**, rather than as a universal black-box guarantee across all oxide terminations.

---

## Recommended Interpretation Strategy

A practical way to interpret SAGE outputs is:

- **metal surfaces**  
  use benchmarked adsorption trends as a reference for workflow validity

- **oxide surfaces**  
  use SAGE as a structured workflow for candidate generation, QC, ranking, and post-relaxation interpretation, while retaining explicit user judgment on termination validity

This distinction is intentional and reflects the physical complexity of oxide surfaces.

---

## Example Workflow

A typical HER workflow in SAGE is:

1. load a metallic structure from Materials Project or from a CIF file
2. prepare the slab with selected supercell and vacuum settings
3. run geometric QC
4. generate adsorption sites
5. run CHE-based evaluation
6. review ranked results and export CSV data

For oxide systems, the same workflow can be applied, but the final interpretation should be made with explicit attention to termination and relaxed-site behavior.

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
- oxide results remain more termination-sensitive than metallic benchmarks
- some oxide systems may require additional user validation after slab construction and relaxation
- certain energy calculators may require separate external access approval
- runtime can increase significantly for large supercells

---

## Citation

If you use SAGE in your research, please cite the software repository.

```text
Junhe Park. Surface Adsorption and Geometry Evaluator (SAGE). GitHub repository.
```

A manuscript citation will be added here after publication.

---

## Acknowledgements

This research was supported by the National Research Foundation of Korea (NRF) funded by the Ministry of Education  
(Grant No. NRF-2023R1A2C2003796; NRF-2020R1A6A1A03042742).

---

## License

This project is distributed under the **AGPL-3.0** license.

