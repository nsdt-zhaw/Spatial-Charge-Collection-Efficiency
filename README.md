# SCE Extraction Analysis App

A Streamlit application for extracting Spatial Collection Efficiency (SCE) profiles from External Quantum Efficiency (EQE) measurements and optical generation profiles.

## Features

- **Automatic SCE Extraction**: Ridge regression with cross-validation to extract SCE profiles
- **Manual SCE Fitting**: Interactive sliders to explore different SCE profiles
- **TMM Generator**: Built-in Transfer Matrix Method calculator for optical generation profiles
- **Real-time Visualization**: Interactive plots for EQE fitting, SCE profiles, and generation analysis
- **Export Functionality**: Download extracted SCE profiles and fitted EQE data

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

## Project Structure

- `app.py` - Main Streamlit application
- `manual_sce_fitting.py` - Manual SCE fitting module
- `tmm_generator.py` - Transfer Matrix Method generator
- `resources/` - Example data files and optical constants
  - `eqe data/` - Example EQE measurements
  - `gen data/` - Example generation profiles
  - `nk data/` - Optical constants (n,k) for various materials
  - `Sunspectrum.sp` - AM1.5G solar spectrum

## Requirements

- Python 3.8+
- Streamlit
- NumPy
- SciPy
- scikit-learn
- Plotly
- tmm (optional, for TMM generator)

## License

MIT License
