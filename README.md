# Surface Adsorption and Geometry Evaluator(SAGE)

An integrated workflow application for machine-learning-assisted electrocatalyst screening.

SAGE App combines structure preparation, structural quality control, adsorption-site generation, ML-assisted pre-screening (CHGNet), CHE-based thermodynamic evaluation (Equiformer), and result management within a single interactive environment. Supports HER, CO₂RR, and ORR on both metallic and oxide surfaces.

# [SAGE App interface]
<img width="1559" height="729" alt="Image" src="https://github.com/user-attachments/assets/fdd9ec61-2b5d-4db0-9670-bf6358d07170" />
## Installation
```bash
git clone https://github.com/Junhepark/Surface-Adsorption-and-Geometry-Evaluator-SAGE-.git
cd Open-Catalyst-Project-application_prototype
pip install -r requirements.txt
```

### UMA model setup (required for CHE calculation)

The CHE evaluation stage uses Meta's UMA model as the default energy calculator. 
The model weights are distributed separately and require access approval:

1. Create a HuggingFace account at https://huggingface.co
2. Request access at https://huggingface.co/facebook/UMA
3. After approval, log in locally:
```bash
pip install huggingface_hub
huggingface-cli login
```
Otherwise, The user could use other models(Including OC20, OC22, or etc...).
If you want to use other model, please modify it in common.py before usage.

The model weights will be downloaded automatically on first use.
## Usage
```bash
streamlit run app/Home.py
```

The application will open in your default browser.

### Quick start

1. Enter a Materials Project ID (e.g., mp-23 for Ni) or upload a .cif file
<img width="2490" height="883" alt="Image" src="https://github.com/user-attachments/assets/60581ebf-e64c-42fe-baf5-ec308757d9a8" />

2. Select material type (Metal / Oxide) and reaction mode (HER / CO₂RR / ORR)

3. Prepare the surface (supercell expansion, vacuum, composition tuning)
<img width="3133" height="1104" alt="Image" src="https://github.com/user-attachments/assets/9440ea60-9aca-4412-b9d1-e3199e97a5ee" />

4. Generate adsorption sites(CHGnet or Geometry) and run CHE calculation
<img width="3328" height="1651" alt="Image" src="https://github.com/user-attachments/assets/91f704a4-cc05-447a-bb92-5e17836559a1" />

5. View ranked results and export CSV
<img width="3324" height="1046" alt="Image" src="https://github.com/user-attachments/assets/c32076e5-d336-4072-a008-16bba958f486" />

## Supported reactions

| Reaction | Adsorbates | Thermodynamic framework |
|----------|-----------|------------------------|
| HER | H* | ΔG_H (CHE) |
| CO₂RR | COOH*, CO*, HCOO*, OCHO* | ΔG_ads vs CO₂/H₂O/H₂ |
| ORR(On modifying) | OOH*, O*, OH* | 4e⁻ Nørskov CHE | 

## Key features

- **Structure input** — .cif upload or Materials Project API retrieval
- **Surface preparation** — supercell expansion, composition tuning, vacuum control
- **Quality control** — pairwise distance check, isolated atom detection, vacuum validation
- **Site generation** — ontop/bridge/hollow for metal and oxide surfaces, PBC-aware deduplication
- **ML pre-screening** — optional CHGNet-based candidate filtering
- **CHE evaluation** — ZPE/entropic corrections, pH/potential adjustments, migration detection
- **Reproducibility** — deterministic seed control (numpy, random, torch)
- **Result management** — session history, metadata export, CSV download
- **LLM interpretation** — optional structured result summarization

```

## Dependencies

- Python >= 3.10
- ASE, pymatgen, CHGNet, PyTorch, Streamlit, py3Dmol, NumPy, pandas

Full list in `requirements.txt`.

## Citation

If you use SAGE App in your research, please cite:
```
.
```

## Acknowledgements

This research was supported by the National Research Foundation of Korea (NRF) funded by the Ministry of Education (Grant No.: NRF-2023R1A2C2003796; NRF-2020R1A6A1A03042742).
