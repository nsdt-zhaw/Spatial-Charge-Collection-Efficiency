# EQE Fitting and SCE Extraction

A Streamlit application for extracting Spatial Collection Efficiency (SCE) profiles from External Quantum Efficiency (EQE) measurements and optical generation profiles of photovoltaic devices.

## Try It Online

The app is deployed and accessible at: **https://sce-extraction.streamlit.app/**

No installation required — upload your data and start analyzing directly in the browser.

## Features

### Automatic SCE Extraction
- Tikhonov (ridge) regression with k-fold cross-validation to find the optimal regularization strength
- First or second derivative regularization matrix
- Interactive alpha slider to explore the regularization landscape after fitting
- Solver options: direct solve with optional clipping to [0,1], or bounded optimization (scipy `lsq_linear`)

### Batch Processing
- Analyze multiple EQE files simultaneously against a shared generation profile
- Comparison plots for CV curves, SCE profiles, and EQE fits side by side
- Summary table with per-file metrics (alpha, MSE, Jsc)

### Weighted Averaging Mode
- 1/MSE weighted average across multiple alpha solutions instead of single CV-optimal
- Uncertainty band visualization
- Configurable saturation detection with MSE derivative threshold

### Weighted Fitting
- Prioritize specific wavelength regions during fitting with a configurable weight factor
- Useful when certain parts of the EQE spectrum are more reliable or important

### Bounds Screening
- During cross-validation, SCE solutions with values outside physically valid bounds are excluded from the optimal alpha search
- Enabled by default with [0, 1] bounds, ensuring only physically meaningful SCE profiles are considered
- Configurable bounds for advanced use cases

### SCE Smoothing
- Post-extraction smoothing with Savitzky-Golay filter or smoothing spline
- Real-time MSE feedback showing the effect of smoothing on fit quality

### Reference SCE Comparison
- Upload reference SCE profiles to overlay on results
- Adjustable horizontal shift for alignment

### Manual SCE Explorer
- Upload a pre-existing SCE profile and immediately evaluate how well it reconstructs the measured EQE
- Displays MSE, R² score, and side-by-side SCE and EQE comparison plots

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
- Download all results as a ZIP archive (SCE profiles, CV data, EQE fits, generation analysis, settings summary)
- Individual file downloads available
- Batch mode produces per-file results in a single ZIP

## Installation (Local)

```bash
pip install -r requirements.txt
```

Run the app locally:

```bash
streamlit run app.py
```

### Input File Formats

| File | Format |
|---|---|
| EQE | Two-column (tab/space separated): wavelength (nm), EQE value |
| Generation profile | Tab-separated: `x (nm)` header, then one column per wavelength (with "nm" suffix in header) |
| Solar spectrum | Two-column: wavelength (nm), irradiance (W/m²/nm) |
| Photon flux | Two-column: wavelength (nm), photon flux (photons/s/cm²) |
| n-k data (TMM) | Three-column: wavelength, n, k |
| SCE profile | Two-column: position (nm), SCE value |

## Project Structure

```
app.py                  - Main Streamlit application
manual_sce_fitting.py   - Manual SCE explorer module
tmm_generator.py        - Transfer Matrix Method generator (optional)
resources/
  eqe data/             - Example EQE measurements
  gen data/             - Example generation profiles
  nk data/              - Optical constants for TMM
  Sunspectrum.sp        - AM1.5G solar spectrum
```

## Dependencies

- Python 3.9+
- streamlit >= 1.28.0
- numpy >= 1.24.0
- scipy >= 1.11.0
- scikit-learn >= 1.3.0
- plotly >= 5.17.0
- tmm >= 0.1.8 (optional, enables the TMM generator)

## Acknowledgments

The Streamlit GUI for this app was built with assistance from [Claude](https://claude.ai) by Anthropic.

## License

MIT License
