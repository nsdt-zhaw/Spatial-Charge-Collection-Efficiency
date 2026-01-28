# SCE Extraction Analysis App

A Streamlit application for extracting Spatial Collection Efficiency (SCE) profiles from External Quantum Efficiency (EQE) measurements and optical generation profiles.

## Features

- **Automatic SCE Extraction**: Ridge regression with cross-validation to find optimal regularization
- **Batch Processing**: Analyze multiple EQE files with option to use same α across all samples
- **Manual SCE Testing**: Upload custom SCE profiles to see resulting EQE fit
- **TMM Generator**: Built-in Transfer Matrix Method calculator for optical generation profiles
- **Flexible Input**: Support for photon flux files or standard spectrum files
- **Weighted Fitting**: Prioritize specific wavelength regions during fitting
- **Export Functionality**: Download all results as ZIP (SCE profiles, CV data, EQE fits)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

### Input Files

- **EQE data**: Two columns (wavelength [nm], EQE)
- **Generation profile**: Position [nm] + generation rates at each wavelength
- **Spectrum**: Wavelength [nm] + flux (or photon flux)

## Project Structure

- `app.py` - Main Streamlit application
- `manual_sce_fitting.py` - Manual SCE fitting module
- `tmm_generator.py` - Transfer Matrix Method generator
- `resources/` - Example data files and optical constants

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
