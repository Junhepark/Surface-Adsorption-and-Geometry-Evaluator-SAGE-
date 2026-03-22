# Open-Catalyst-Project-application_prototype

An integrated workflow application for machine-learning-assisted electrocatalyst screening.

OCP App combines structure preparation, structural quality control, adsorption-site generation, ML-assisted pre-screening (CHGNet), CHE-based thermodynamic evaluation (Equiformer), and result management within a single interactive environment. Supports HER, CO₂RR, and ORR on both metallic and oxide surfaces.

## Installation
```bash
git clone https://github.com/Junhepark/Open-Catalyst-Project-application_prototype.git
cd Open-Catalyst-Project-application_prototype
pip install -r requirements.txt
```

## Usage
```bash
streamlit run app/Home.py
```

The application will open in your default browser.

### Quick start

1. Enter a Materials Project ID (e.g., mp-23 for Ni) or upload a .cif file
2. Select material type (Metal / Oxide) and reaction mode (HER / CO₂RR / ORR)
3. Prepare the surface (supercell expansion, vacuum, composition tuning)
4. Generate adsorption sites and run CHE calculation
5. View ranked results and export CSV

## Supported reactions

| Reaction | Adsorbates | Thermodynamic framework |
|----------|-----------|------------------------|
| HER | H* | ΔG_H (CHE) |
| CO₂RR | COOH*, CO*, HCOO*, OCHO* | ΔG_ads vs CO₂/H₂O/H₂ |
| ORR | OOH*, O*, OH* | 4e⁻ Nørskov CHE |

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

## Project structure
```
├── app/
│   └── Home.py                  # Main Streamlit application
├── ocp_app/
│   └── core/
│       ├── ads_sites.py         # Adsorption site detection
│       ├── adsorbml_lite_screening.py  # CHGNet ML pre-screening
│       ├── CHE_mode.py          # CHE thermodynamic evaluation
│       ├── cifgen.py            # Structure generation and tuning
│       ├── gas_refs.py          # Gas-phase reference energies
│       ├── run_history.py       # Session history management
│       ├── seeds.py             # Reproducibility seed control
│       └── structure_check.py   # Structural quality control
├── ref_gas/                     # Gas reference CIF templates
├── requirements.txt
├── LICENSE
└── README.md
```

## Dependencies

- Python >= 3.10
- ASE, pymatgen, CHGNet, PyTorch, Streamlit, py3Dmol, NumPy, pandas

Full list in `requirements.txt`.

## Citation

If you use OCP App in your research, please cite:
```
J. Park, D. Kim, "OCP App: An integrated workflow application for 
machine-learning-assisted electrocatalyst screening," SoftwareX (submitted).
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

This research was supported by the National Research Foundation of Korea (NRF) funded by the Ministry of Education (Grant No.: NRF-2023R1A2C2003796; NRF-2020R1A6A1A03042742).
