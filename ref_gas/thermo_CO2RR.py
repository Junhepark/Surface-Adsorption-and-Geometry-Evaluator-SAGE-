{
  "MODEL": "uma-s-1p1",
  "T (K)": 298.15,
  "E_gas (eV)": {
    "H2": -6.9696671472688925,
    "CO2": -22.3087728268725,
    "CO": -14.417911320665826,
    "H2O": -14.155283695074841
  },
  "E_ads_ref (eV)": {
    "CO": -14.417911320665826,
    "COOH": -23.363462716358157,
    "HCOO": -23.30369010569379
  },
  "ΔZPE_ads (eV)": {
    "COOH": 0.64,
    "HCOO": 0.62,
    "CO": 0.21
  },
  "TΔS_gas (eV)": {
    "H2": 0.4,
    "CO2": 0.66,
    "CO": 0.61,
    "H2O": 0.6
  },
  "correction_scope": "species_constant_deltaZPE_only",
  "correction_note": "Values under ΔZPE_ads (eV) are applied as species-level ZPE-only corrections. They are not complete Gibbs corrections; entropy, solvation, and interfacial-field effects are not included. Species absent from this block are reported with ΔG_ads = null/NaN.",
  "canonical_filename": "thermo_CO2RR.json"
}