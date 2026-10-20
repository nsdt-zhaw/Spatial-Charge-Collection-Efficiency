# EQE Fitting and φ(z) Extraction

A Streamlit application for extracting the depth-resolved charge collection probability $\varphi(z)$ — also known as the spatial collection efficiency (SCE) — from External Quantum Efficiency (EQE) measurements and optical generation profiles of photovoltaic devices.

## Try It Online

The app is deployed and accessible at: **https://sce-extraction.streamlit.app/**

Upload your data and start analyzing directly in the browser. For heavier workloads (large generation grids, batch processing, long cross-validation sweeps), we recommend running the app locally — see [Installation (Local)](#installation-local) below.

## Features

### Automatic φ(z) Extraction
- Tikhonov (ridge) regression with k-fold cross-validation to find the optimal regularization strength $\gamma$
- First or second derivative regularization matrix
- Interactive γ slider to explore the regularization landscape after fitting

### Batch Processing
- Analyze multiple EQE files simultaneously against a shared generation profile
- Comparison plots for CV curves, φ profiles, and EQE fits side by side
- Summary table with per-file metrics (γ, MSE, Jsc)

### Weighted Fitting
- Prioritize specific wavelength regions during fitting with a configurable weight factor
- Useful when certain parts of the EQE spectrum are more reliable or important

### Bounds Screening
- During cross-validation, φ solutions with values outside physically valid bounds are excluded from the optimal γ search
- Enabled by default with [0, 1] bounds, ensuring only physically meaningful φ profiles are considered
- Configurable bounds for advanced use cases

### Reference φ Comparison
- Upload reference φ profiles to overlay on results
- Adjustable horizontal shift for alignment

### Manual φ Explorer
- Upload a pre-existing φ profile and immediately evaluate how well it reconstructs the measured EQE
- Displays MSE, R² score, and side-by-side φ and EQE comparison plots

### TMM Generation Profile Calculator
- Built-in Transfer Matrix Method (TMM) calculator for computing optical generation profiles from device layer stacks, based on the [tmm](https://github.com/sbyrnes321/tmm) Python package
- Define multi-layer structures with per-layer thickness, coherence, and n-k optical constants
- Supports coherent, incoherent, and mixed optical modeling
- Outputs generation profiles in the correct format for direct use in the analysis
- Visualization of absorptance, reflectance, and transmittance spectra

### Flexible Input Options
- Built-in example data files (EQE, generation profiles, n-k data, AM1.5G spectrum)
- File upload for custom data
- Support for standard spectrum files or direct photon flux files
- White light mode for flat-spectrum analysis

### Export
- Download all results as a ZIP archive (φ profiles, CV data, EQE fits, generation analysis, settings summary)
- Individual file downloads available
- Batch mode produces per-file results in a single ZIP

## Installation (Local)

Running locally is recommended for larger datasets or when the online app feels slow.

**1. Clone the repository**

```bash
git clone https://github.com/nsdt-zhaw/Spatial-Charge-Collection-Efficiency.git
cd Spatial-Charge-Collection-Efficiency
```

**2. (Optional but recommended) Create a virtual environment**

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

**3. Install the dependencies**

```bash
pip install -r requirements.txt
```

**4. Launch the Streamlit app**

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

### Input File Formats

| File | Format |
|---|---|
| EQE | Two-column (tab/space separated): wavelength (nm), EQE value |
| Generation profile | Tab-separated: `x (nm)` header, then one column per wavelength (with "nm" suffix in header) |
| Solar spectrum | Two-column: wavelength (nm), irradiance (W/m²/nm) |
| Photon flux | Two-column: wavelength (nm), photon flux (photons/s/cm²) |
| n-k data (TMM) | Three-column: wavelength, n, k |
| φ profile | Two-column: position (nm), φ value |

## Example Data

The app ships two built-in example datasets:

### 1. Simulated example (drift-diffusion with known ground-truth φ(z))

| Setting | Value |
|---|---|
| EQE | `simulated_example_EQE.txt` and/or `simulated_example_EQE_noions.txt` |
| Generation | `simulated_example_generation.txt` |
| Spectrum | Select **Photon flux file** under *Incident spectrum* and upload `simulated_example_photon_flux.txt` |
| Boundaries | x1 = 50 nm, x2 = 600 nm |
| Reference φ | `simulated_example_SCE.txt` / `simulated_example_SCE_noions.txt` |

The two EQE files compare a device with and without mobile ions, sharing the same generation profile and photon flux spectrum. The reference φ profiles in `sce data/` provide the ground truth for validation.

### 2. Experimental example — PSC7 pulsed EQE at Vpre = 0.7 V (reproduces one case of main-text Fig. 6)

| Setting | Value |
|---|---|
| EQE | `experimental_pulsed_EQE_Vpre_0p7V.txt` (pulsed EQE at V_MPP = 0.77 V after 0.7 V preconditioning) |
| Generation | `experimental_generation.txt` (Beer-Lambert, D = 700 nm, α_k from Dasgupta `FACsPb_Br010_Eg1.48eV.nk`, built at 1 sun AM1.5G) |
| Spectrum | Select **Default (AM1.5G)** under *Incident spectrum* (the generation file is built at the same 1 sun scaling) |
| Boundaries | x1 = 0 nm, x2 = 700 nm (absorber only) |
| Fit range | 350–800 nm; recommend down-weighting 750–800 nm |

Reproduces the Vpre = 0.7 V case from the main-text Fig. 6 pulsed-EQE series.

## Project Structure

```
app.py                  - Main Streamlit application
manual_sce_fitting.py   - Manual φ explorer module
tmm_generator.py        - Transfer Matrix Method generator
resources/
  eqe data/             - Example EQE measurements
  gen data/             - Example generation profiles (coherent and incoherent)
  sce data/             - Reference φ profiles for validation
  nk data/              - Optical constants for TMM
    dasgupta/           - Perovskite + SAM/SnO2/glass n,k from Dasgupta et al.
  Sunspectrum.sp        - AM1.5G solar spectrum
```

## Dependencies

- Python 3.9+
- streamlit >= 1.28.0
- numpy >= 1.24.0
- scipy >= 1.11.0
- scikit-learn >= 1.3.0
- plotly >= 5.17.0
- tmm >= 0.1.8

## Acknowledgments

The Streamlit GUI for this app was built with assistance from [Claude](https://claude.ai) by Anthropic.

The perovskite, SAM, SnO₂ and glass n,k datasets under `resources/nk data/dasgupta/` are redistributed from the [Perovskite-Dielectric-Constants-Repository](https://github.com/akashdasgupta/Perovskite-Dielectric-Constants-Repository) by Dasgupta et al.

Several of the common-material n,k files at the top level of `resources/nk data/` (e.g. Au, C60, FTO, ITO, MgF₂, PEDOT, TiO₂, glass) are taken from the [SIMsalabim data directory](https://github.com/kostergroup/SIMsalabim/tree/master/Data).

## License

MIT License
