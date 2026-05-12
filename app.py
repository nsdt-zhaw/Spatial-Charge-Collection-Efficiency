import streamlit as st
import numpy as np
# NumPy 2.0+ removed trapz entirely; use trapezoid with fallback
trapz_func = getattr(np, 'trapezoid', None) or np.trapz
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.constants import physical_constants as pc
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, LeaveOneOut
import io
import os
import zipfile
from pathlib import Path
from datetime import datetime

try:
    from tmm_generator import generation_profile_creator
    TMM_AVAILABLE = True
except ImportError:
    TMM_AVAILABLE = False
    st.warning("""
    ⚠️ TMM generator not available. To enable:
    1. Place `tmm_generator.py` in the same folder as `app.py`
    2. Run: `pip install tmm`
    """)

try:
    from manual_sce_fitting import manual_sce_fitting_tab
    MANUAL_FITTING_AVAILABLE = True
except ImportError:
    MANUAL_FITTING_AVAILABLE = False

# Helper functions for wavelength header detection
import re

def is_wavelength_header(line):
    """Check if a line is a wavelength header using multiple criteria."""
    line_lower = line.lower()
    # Check for common position column names
    has_position_col = any(x in line_lower for x in ['x (nm)', 'depth (nm)', 'depth(nm)', 'position'])
    # Check if line contains multiple 'nm' occurrences (wavelength columns)
    nm_count = line_lower.count(' nm') + line_lower.count('\tnm')
    has_multiple_nm = nm_count >= 2
    # Check for numeric values followed by nm (e.g., "315.0 nm")
    has_wavelength_pattern = bool(re.search(r'\d+\.?\d*\s*nm', line_lower))
    return has_position_col or has_multiple_nm or (has_wavelength_pattern and nm_count >= 1)

def extract_wavelengths_from_line(line, expected_count):
    """Extract wavelength values from a header line."""
    parts = line.replace('#', '').strip().split()
    if len(parts) <= 1:
        return None
    try:
        wavelengths = []
        for part in parts[1:]:
            # Try to extract number from string like "315.0" or "315.0nm"
            num_str = ''.join(c for c in part if c.isdigit() or c == '.')
            if num_str:
                wavelengths.append(float(num_str))
        wavelengths = np.array(wavelengths)
        if len(wavelengths) == expected_count:
            return wavelengths
    except:
        pass
    return None

# Helper functions for file management
def list_files_in_directory(directory, extensions=['.txt', '.csv', '.dat', '.sp']):
    """List all files with given extensions in a directory."""
    path = Path(directory)
    if not path.exists():
        return []
    files = []
    for ext in extensions:
        files.extend([f.name for f in path.glob(f'*{ext}')])
    return sorted(files)

def load_file_data(file_source):
    """Load data from either uploaded file or local path."""
    if isinstance(file_source, str):
        # Local file path
        return open(file_source, 'rb')
    else:
        # Uploaded file object
        return file_source

def create_results_zip(alphas, mse, lam, pos, inc_flux, y, y_fit_current, y_fit_optimal,
                       sce_current_raw, current_alpha, best_alpha,
                       eqe_original_wavelengths, eqe_original_values, gen_filtered,
                       settings_info, sce_current_smooth=None, mse_pure=None,
                       oob_mask=None):
    """Create an in-memory ZIP file containing all analysis results.

    Parameters:
    -----------
    settings_info : dict
        Dictionary containing all settings and file information for the analysis_info.txt file
    sce_current_smooth : array, optional
        Smoothed SCE profile (if smoothing is enabled)
    mse_pure : dict, optional
        Pure MSE values without penalty (when penalties are active)
    """
    zip_buffer = io.BytesIO()

    # Check if penalties were used (mse_pure differs from mse)
    penalties_active = mse_pure is not None and mse_pure != mse

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. CV results (alpha, mse, and optionally mse_pure, in_bounds)
        in_bounds_col = np.array([int(not oob_mask.get(a, False)) for a in alphas]) if oob_mask else np.ones(len(alphas), dtype=int)
        if penalties_active:
            cv_data = np.column_stack((alphas, [mse_pure[a] for a in alphas], [mse[a] for a in alphas], in_bounds_col))
            cv_buffer = io.StringIO()
            np.savetxt(cv_buffer, cv_data, fmt='%.6e\t%.6e\t%.6e\t%d', header='alpha\tMSE_pure\tCV_score_combined\tin_bounds')
        else:
            cv_data = np.column_stack((alphas, [mse[a] for a in alphas], in_bounds_col))
            cv_buffer = io.StringIO()
            np.savetxt(cv_buffer, cv_data, fmt='%.6e\t%.6e\t%d', header='alpha\tMSE\tin_bounds')
        zf.writestr('cv_results.txt', cv_buffer.getvalue())

        # 2. EQE original (wavelength, eqe)
        eqe_orig_data = np.column_stack((eqe_original_wavelengths, eqe_original_values))
        eqe_orig_buffer = io.StringIO()
        np.savetxt(eqe_orig_buffer, eqe_orig_data, fmt='%.6f\t%.6f', header='wavelength(nm)\tEQE')
        zf.writestr('eqe_original.txt', eqe_orig_buffer.getvalue())

        # 3. EQE interpolated (wavelength, eqe)
        eqe_interp_data = np.column_stack((lam, y / inc_flux))
        eqe_interp_buffer = io.StringIO()
        np.savetxt(eqe_interp_buffer, eqe_interp_data, fmt='%.6f\t%.6f', header='wavelength(nm)\tEQE_interpolated')
        zf.writestr('eqe_interpolated.txt', eqe_interp_buffer.getvalue())

        # 4. EQE fit current alpha (wavelength, eqe_fit)
        eqe_fit_current_data = np.column_stack((lam, y_fit_current / inc_flux))
        eqe_fit_current_buffer = io.StringIO()
        np.savetxt(eqe_fit_current_buffer, eqe_fit_current_data, fmt='%.6f\t%.6f',
                   header=f'wavelength(nm)\tEQE_fit (alpha={current_alpha:.2e})')
        zf.writestr('eqe_fit_current.txt', eqe_fit_current_buffer.getvalue())

        # 5. EQE fit optimal alpha (wavelength, eqe_fit)
        eqe_fit_optimal_data = np.column_stack((lam, y_fit_optimal / inc_flux))
        eqe_fit_optimal_buffer = io.StringIO()
        np.savetxt(eqe_fit_optimal_buffer, eqe_fit_optimal_data, fmt='%.6f\t%.6f',
                   header=f'wavelength(nm)\tEQE_fit (optimal alpha={best_alpha:.2e})')
        zf.writestr('eqe_fit_optimal.txt', eqe_fit_optimal_buffer.getvalue())

        # 6. SCE current alpha (position, sce - include both raw and smoothed if available)
        if sce_current_smooth is not None:
            smooth_method = settings_info.get('smooth_method', 'Unknown')
            if smooth_method == "Savitzky-Golay":
                smoothing_note = f"Savgol window={settings_info.get('smooth_window')}, poly={settings_info.get('smooth_polyorder')}"
            else:
                smoothing_note = f"Spline s={settings_info.get('spline_smoothing'):.4f}"
            sce_current_data = np.column_stack((pos, sce_current_raw, sce_current_smooth))
            sce_current_buffer = io.StringIO()
            np.savetxt(sce_current_buffer, sce_current_data, fmt='%.6f\t%.6f\t%.6f',
                       header=f'pos(nm)\tSCE_raw (alpha={current_alpha:.2e})\tSCE_smoothed ({smoothing_note})')
        else:
            sce_current_data = np.column_stack((pos, sce_current_raw))
            sce_current_buffer = io.StringIO()
            np.savetxt(sce_current_buffer, sce_current_data, fmt='%.6f\t%.6f',
                       header=f'pos(nm)\tSCE (alpha={current_alpha:.2e})')
        zf.writestr('sce_current.txt', sce_current_buffer.getvalue())

        # 7. Generation analysis (position, total_gen, collected_gen, cumulative_jsc)
        # Use smoothed SCE if available, otherwise raw
        sce_for_gen = sce_current_smooth if sce_current_smooth is not None else sce_current_raw
        total_generation = trapz_func(gen_filtered, lam, axis=1)
        collected_generation = sce_for_gen * total_generation

        # Calculate cumulative Jsc
        cumulative_jsc = np.zeros_like(pos)
        for i in range(1, len(pos)):
            cumulative_jsc[i] = trapz_func(collected_generation[:i+1], pos[:i+1])

        # Convert to mA/cm^2
        q = 1.602e-19  # Coulombs
        cumulative_jsc = cumulative_jsc * q * 1e-7 * 1000  # Convert to mA/cm^2

        gen_analysis_data = np.column_stack((pos, total_generation, collected_generation, cumulative_jsc))
        gen_analysis_buffer = io.StringIO()
        np.savetxt(gen_analysis_buffer, gen_analysis_data, fmt='%.6f\t%.6e\t%.6e\t%.6f',
                   header='pos(nm)\ttotal_generation(cm-3s-1)\tcollected_generation(cm-3s-1)\tcumulative_Jsc(mA/cm2)')
        zf.writestr('generation_analysis.txt', gen_analysis_buffer.getvalue())

        # 9. Create comprehensive analysis_info.txt with all settings and file information
        info_lines = [
            "=" * 60,
            "SCE EXTRACTION ANALYSIS - SETTINGS AND RESULTS SUMMARY",
            "=" * 60,
            "",
            "INPUT FILES",
            "-" * 40,
            f"EQE file:        {settings_info.get('eqe_file', 'N/A')}",
            f"Generation file: {settings_info.get('gen_file', 'N/A')}",
            f"Spectrum source: {settings_info.get('sun_source', 'N/A')}",
            "",
            "BOUNDARIES",
            "-" * 40,
            f"CTL boundary x1: {settings_info.get('x1', 'N/A')} nm",
            f"CTL boundary x2: {settings_info.get('x2', 'N/A')} nm",
            "",
            "PROCESSING OPTIONS",
            "-" * 40,
            f"Use white light spectrum:      {settings_info.get('use_white', 'N/A')}",
            f"Apply 0-1 clipping:            {settings_info.get('use_clipping', 'N/A')}",
            f"Use bounded optimization:      {settings_info.get('use_bounded', 'N/A')}",
            f"Use 1st derivative reg.:       {settings_info.get('use_first_derivative', 'N/A')}",
            f"Cross-validation:              {'LOO' if settings_info.get('n_splits') == 'LOO' else f\"{settings_info.get('n_splits', 'N/A')}-fold\"}",
            "",
            "WAVELENGTH FILTER",
            "-" * 40,
            f"Enabled:         {settings_info.get('use_wl_filter', False)}",
        ]
        if settings_info.get('use_wl_filter', False):
            info_lines.extend([
                f"Min wavelength:  {settings_info.get('wl_min', 'N/A')} nm",
                f"Max wavelength:  {settings_info.get('wl_max', 'N/A')} nm",
            ])

        info_lines.extend([
            "",
            "WEIGHTED FITTING",
            "-" * 40,
            f"Enabled:         {settings_info.get('use_weighted', False)}",
        ])
        if settings_info.get('use_weighted', False):
            info_lines.extend([
                f"Min wavelength:  {settings_info.get('weight_wl_min', 'N/A')} nm",
                f"Max wavelength:  {settings_info.get('weight_wl_max', 'N/A')} nm",
                f"Weight factor:   {settings_info.get('weight_factor', 'N/A')}",
            ])

        info_lines.extend([
            "",
            "REGULARIZATION",
            "-" * 40,
            f"Alpha range (log10): {settings_info.get('alpha_min', 'N/A')} to {settings_info.get('alpha_max', 'N/A')}",
            f"Number of alphas:    {settings_info.get('n_alphas', 'N/A')}",
        ])

        # SCE Smoothing section
        info_lines.extend([
            "",
            "SCE SMOOTHING",
            "-" * 40,
            f"Enabled:             {settings_info.get('use_smoothing', False)}",
        ])
        if settings_info.get('use_smoothing', False):
            smooth_method = settings_info.get('smooth_method', 'Savitzky-Golay')
            info_lines.append(f"Method:              {smooth_method}")
            if smooth_method == "Savitzky-Golay":
                info_lines.append(f"Window size:         {settings_info.get('smooth_window', 'N/A')}")
                info_lines.append(f"Polynomial order:    {settings_info.get('smooth_polyorder', 'N/A')}")
            else:  # Smoothing Spline
                info_lines.append(f"Smoothing parameter: {settings_info.get('spline_smoothing', 'N/A')}")

        # Bounds screening settings
        info_lines.extend([
            "",
            "BOUNDS SCREENING",
            "-" * 40,
            f"Screen by bounds:    {settings_info.get('screen_out_of_bounds', True)}",
        ])
        if settings_info.get('screen_out_of_bounds', True):
            info_lines.extend([
                f"  SCE min:           {settings_info.get('screen_bounds_min', 0.0)}",
                f"  SCE max:           {settings_info.get('screen_bounds_max', 1.0)}",
            ])

        # Weighted averaging settings
        info_lines.extend([
            "",
            "WEIGHTED AVERAGING",
            "-" * 40,
            f"Enabled:             {settings_info.get('use_weighted_avg', False)}",
        ])
        if settings_info.get('use_weighted_avg', False):
            info_lines.extend([
                f"  SCE bounds min:    {settings_info.get('sce_bounds_min', -0.001)}",
                f"  SCE bounds max:    {settings_info.get('sce_bounds_max', 1.001)}",
                f"  MSE sat. factor:   {settings_info.get('mse_saturation_factor', 10.0)}",
                f"  Deriv. threshold:  {settings_info.get('mse_deriv_threshold', 0.01)}",
                f"  Use CV for MSE:    {settings_info.get('use_cv_for_mse', True)}",
            ])

        info_lines.extend([
            "",
            "RESULTS",
            "-" * 40,
            f"Optimal alpha:       {best_alpha:.4e}",
        ])
        if penalties_active:
            info_lines.extend([
                f"Optimal pure MSE:    {mse_pure[best_alpha]:.4e}",
                f"Optimal CV score:    {mse[best_alpha]:.4e}",
            ])
        else:
            info_lines.append(f"Optimal MSE:         {mse[best_alpha]:.4e}")
        info_lines.extend([
            f"Current alpha:       {current_alpha:.4e}",
            f"Current MSE:         {mse.get(current_alpha, 'N/A') if current_alpha in mse else 'interpolated'}",
            f"Final Jsc:           {cumulative_jsc[-1]:.2f} mA/cm²",
            "",
            "DATA RANGES",
            "-" * 40,
            f"Wavelength range:    {lam.min():.1f} - {lam.max():.1f} nm ({len(lam)} points)",
            f"Depth range:         {pos.min():.1f} - {pos.max():.1f} nm ({len(pos)} points)",
            "",
            "=" * 60,
            "FILES IN THIS ARCHIVE",
            "=" * 60,
        ])
        if penalties_active:
            info_lines.append("cv_results.txt          - CV alpha vs pure MSE and combined CV score")
        else:
            info_lines.append("cv_results.txt          - Cross-validation alpha vs MSE")
        info_lines.extend([
            "eqe_original.txt        - Original EQE data",
            "eqe_interpolated.txt    - EQE interpolated to generation wavelengths",
            "eqe_fit_current.txt     - Fitted EQE at current alpha",
            "eqe_fit_optimal.txt     - Fitted EQE at optimal alpha",
            "sce_current.txt         - SCE profile at current alpha",
            "sce_optimal.txt         - SCE profile at optimal alpha (with std)",
            "generation_analysis.txt - Generation and Jsc analysis",
            "analysis_info.txt       - This file",
            "",
        ])

        zf.writestr('analysis_info.txt', '\n'.join(info_lines))

    zip_buffer.seek(0)
    return zip_buffer

def preview_eqe_data(eqe_file):
    """Preview EQE data with a simple plot."""
    try:
        # Load the EQE data
        data = np.loadtxt(eqe_file)

        # Extract wavelength and EQE values
        wavelengths = data[:, 0]
        eqe_values = data[:, 1]

        # Create plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=wavelengths,
            y=eqe_values,
            mode='lines+markers',
            line=dict(color='rgb(0, 100, 200)', width=2),
            marker=dict(size=4),
            name='EQE',
            hovertemplate='<b>Wavelength:</b> %{x:.1f} nm<br>' +
                        '<b>EQE:</b> %{y:.4f}<br>' +
                        '<extra></extra>'
        ))

        fig.update_layout(
            title=dict(text='External Quantum Efficiency (EQE)', font=dict(size=18, family='Arial Black')),
            xaxis=dict(title='Wavelength (nm)', gridcolor='lightgray'),
            yaxis=dict(title='EQE (-)', gridcolor='lightgray', range=[0, max(1, eqe_values.max()*1.1)]),
            template='plotly_white',
            hovermode='x unified',
            height=400,
            showlegend=False,
            plot_bgcolor='white',
            margin=dict(l=60, r=60, t=60, b=60)
        )

        return fig, wavelengths, eqe_values

    except Exception as e:
        st.error(f"Error previewing EQE data: {str(e)}")
        return None, None, None

def preview_generation_profile(gen_file):
    """Preview generation profile with plots similar to TMM generator."""
    try:
        # Load the generation data (skip comment lines)
        data = np.loadtxt(gen_file)

        # Extract position and generation profiles
        z_positions = data[:, 0]
        generation = data[:, 1:]  # All wavelength columns

        # Try to extract wavelengths from header
        wavelengths = None
        if isinstance(gen_file, str):
            # Local file - can read header
            with open(gen_file, 'r') as f:
                lines = f.readlines()
                # Look for the header line with wavelengths
                for line in lines:
                    if is_wavelength_header(line):
                        wavelengths = extract_wavelengths_from_line(line, generation.shape[1])
                        if wavelengths is not None:
                            break
        else:
            # Uploaded file - try to read header
            try:
                gen_file.seek(0)
                content = gen_file.read().decode('utf-8')
                gen_file.seek(0)  # Reset for np.loadtxt
                lines = content.split('\n')
                for line in lines:
                    if is_wavelength_header(line):
                        wavelengths = extract_wavelengths_from_line(line, generation.shape[1])
                        if wavelengths is not None:
                            break
            except:
                pass

        # If no wavelengths found, create dummy array
        if wavelengths is None:
            wavelengths = np.arange(generation.shape[1])

        # Calculate total generation (integrated over wavelength)
        if len(wavelengths) == generation.shape[1]:
            total_generation = trapz_func(generation, wavelengths, axis=1)
        else:
            total_generation = generation.sum(axis=1)

        # Create 2-subplot layout
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                'Total Generation Profile G(z)',
                'Wavelength-Dependent Generation G(z,λ)'
            ),
            specs=[[{"type": "scatter"}, {"type": "heatmap"}]],
            horizontal_spacing=0.15,
            column_widths=[0.45, 0.55]
        )

        # PLOT 1: Total Generation Profile
        fig.add_trace(
            go.Scatter(
                x=z_positions,
                y=total_generation / 1e21,
                mode='lines',
                line=dict(color='rgb(220, 20, 60)', width=3),
                fill='tozeroy',
                fillcolor='rgba(220, 20, 60, 0.3)',
                name='Total G(z)',
                hovertemplate='<b>Depth:</b> %{x:.1f} nm<br>' +
                            '<b>Generation:</b> %{y:.2f}×10²¹ cm⁻³s⁻¹<br>' +
                            '<extra></extra>',
                showlegend=False
            ),
            row=1, col=1
        )

        fig.update_xaxes(
            title_text="Depth (nm)",
            row=1, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )

        fig.update_yaxes(
            title_text="Generation Rate (×10²¹ cm⁻³s⁻¹)",
            row=1, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )

        # PLOT 2: Heatmap G(z, λ)
        # Note: generation is (n_depths x n_wavelengths), so we transpose for heatmap
        fig.add_trace(
            go.Heatmap(
                x=wavelengths if len(wavelengths) == generation.shape[1] else np.arange(generation.shape[1]),
                y=z_positions,
                z=generation / 1e21,  # Don't transpose - z_positions are rows, wavelengths are columns
                colorscale='Plasma',  # Changed to Plasma for better visibility
                colorbar=dict(
                    title="G(z,λ)<br>(×10²¹ cm⁻³s⁻¹)",
                    x=1.02,
                    len=0.9,
                    thickness=15
                ),
                hovertemplate='<b>Wavelength:</b> %{x:.1f} nm<br>' +
                            '<b>Depth:</b> %{y:.1f} nm<br>' +
                            '<b>Generation:</b> %{z:.2f}×10²¹ cm⁻³s⁻¹<br>' +
                            '<extra></extra>'
            ),
            row=1, col=2
        )

        fig.update_xaxes(
            title_text="Wavelength (nm)" if len(wavelengths) == generation.shape[1] else "Wavelength Index",
            row=1, col=2,
            showgrid=False
        )

        fig.update_yaxes(
            title_text="Depth (nm)",
            row=1, col=2,
            showgrid=False
        )

        # Update overall layout
        fig.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor='white',
            margin=dict(l=60, r=60, t=60, b=60)
        )

        return fig, z_positions, generation, wavelengths, total_generation

    except Exception as e:
        st.error(f"Error previewing generation data: {str(e)}")
        return None, None, None, None, None

# Page config
st.set_page_config(page_title="SCE Extraction Analysis", layout="wide", page_icon="⚡")

# Initialize manual fitting state to closed on first load
if 'manual_fitting_open' not in st.session_state:
    st.session_state.manual_fitting_open = False

st.title("EQE fitting and SCE extraction")
st.markdown("Upload your experimental EQE data and simulated generation profiles to extract the spatial collection efficiency (SCE)")

with st.sidebar:
    if st.button("🔄 Clear All", use_container_width=True, type="secondary", help="Reset the app and clear all data"):
        st.session_state.clear()
        st.rerun()

# Custom CSS for smaller headers on small screens and sidebar width
st.markdown("""
<style>
    /* Widen sidebar for settings */
    [data-testid="stSidebar"] {
        min-width: 420px;
        max-width: 480px;
    }

    /* Reduce header sizes */
    h1 {
        font-size: 2rem !important;
    }
    h2 {
        font-size: 1.5rem !important;
    }
    h3 {
        font-size: 1.2rem !important;
    }

    /* For smaller screens, reduce even more */
    @media (max-width: 768px) {
        h1 {
            font-size: 1.6rem !important;
        }
        h2 {
            font-size: 1.3rem !important;
        }
        h3 {
            font-size: 1.1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Custom Ridge Regression Model
class CustomRidgeDirect(BaseEstimator, RegressorMixin):
    """Direct ridge regression solver with coefficient clipping or bounded optimization."""
    def __init__(self, alpha=1.0, L=None, constraint=True, use_bounded=False):
        self.alpha = alpha
        self.L = L
        self.constraint = constraint
        self.use_bounded = use_bounded
        self.coef_ = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        L = self.L if self.L is not None else np.eye(X.shape[1])

        if self.use_bounded:
            # Use bounded optimization with scipy.optimize.lsq_linear
            # Formulate the problem: minimize ||Xw - y||^2 + alpha * ||Lw||^2
            # This is equivalent to: minimize ||[X; sqrt(alpha)*L] w - [y; 0]||^2
            X_aug = np.vstack([X, np.sqrt(self.alpha) * L])
            y_aug = np.concatenate([y, np.zeros(L.shape[0])])

            # Solve with bounds [0, 1]
            result = lsq_linear(X_aug, y_aug, bounds=(0, 1), method='bvls', verbose=0)
            w = result.x
        else:
            # Original direct solve method
            A = X.T @ X + self.alpha * (L.T @ L)
            b = X.T @ y
            w = np.linalg.solve(A, b)
            if self.constraint:
                w = np.clip(w, 0, 1)

        self.coef_ = w
        return self

    def predict(self, X):
        check_is_fitted(self, 'coef_')
        X = check_array(X)
        return X @ self.coef_



def compute_oscillation_penalty(coef, threshold=0):
    """Compute penalty for oscillations (direction changes) in the solution.

    Counts the number of sign changes in the first derivative, which indicates
    how many times the solution changes direction (peaks/valleys).
    Only penalizes oscillations beyond the threshold.

    Parameters:
    -----------
    coef : array
        Coefficient values (SCE profile)
    threshold : int
        Number of direction changes allowed before penalizing (default: 0)

    Returns:
    --------
    excess_oscillations : int
        Number of direction changes beyond the threshold (0 if below threshold)
    """
    # Compute first derivative (differences)
    diff = np.diff(coef)
    # Count sign changes: where diff[i] * diff[i+1] < 0
    sign_changes = np.sum(diff[:-1] * diff[1:] < 0)
    # Only penalize oscillations beyond threshold
    return max(0, sign_changes - threshold)


def fit_sce_unclipped(X, y, L, alpha):
    """Fit SCE without any clipping (for weighted averaging approach)."""
    A = X.T @ X + alpha * (L.T @ L)
    b = X.T @ y
    return np.linalg.solve(A, b)


def compute_mse(X, y, sce):
    """Compute mean squared error for a solution."""
    y_pred = X @ sce
    return np.mean((y_pred - y) ** 2)


def weighted_mse_averaging(X, y, L, alphas, sce_bounds_min=-0.001, sce_bounds_max=1.001,
                           mse_saturation_factor=10.0, mse_deriv_threshold=0.01,
                           use_cv=False, n_splits=5, progress_callback=None):
    """Apply weighted MSE averaging with MSE-derivative-based saturation detection.

    This approach:
    1. Extracts SCE WITHOUT clipping for all alpha values
    2. Screens solutions by physical bounds (SCE within sce_bounds_min to sce_bounds_max)
    3. Starts from first in-bounds solution (lowest alpha with valid SCE)
    4. Continues until MSE > mse_saturation_factor * min_MSE, then checks for saturation
    5. Stops when |d(log MSE)/d(log alpha)| < mse_deriv_threshold (MSE curve flattens)
    6. Averages all valid solutions using 1/MSE as weights

    Parameters:
    -----------
    X : array
        Design matrix (wavelengths x positions)
    y : array
        Target values (EQE * photon_flux)
    L : array
        Regularization matrix (first or second derivative)
    alphas : array
        Alpha values to test (should be in increasing order)
    sce_bounds_min, sce_bounds_max : float
        Physical bounds for SCE screening (default: -0.001 to 1.001 for numerical tolerance)
    mse_saturation_factor : float
        Only check for saturation after MSE > factor * min_MSE (default: 10.0)
    mse_deriv_threshold : float
        Stop when |d(log MSE)/d(log alpha)| < threshold (default: 0.01)
    use_cv : bool
        If True, use k-fold CV to compute MSE (slower but more robust). Default: False
    n_splits : int or str
        Number of CV folds if use_cv=True. Use "LOO" for leave-one-out CV. Default: 5
    progress_callback : callable, optional
        Function to call with progress (0-1) for UI updates

    Returns:
    --------
    dict with keys:
        'sce_avg': Weighted average SCE profile
        'sce_std': Weighted standard deviation at each position
        'best_alpha': First in-bounds alpha (equivalent to CV optimal without clipping)
        'all_sce': All SCE profiles
        'all_mse': All MSE values
        'bounds_mask': Boolean mask for in-bounds solutions
        'final_mask': Boolean mask for solutions included in averaging
        'min_mse': Minimum MSE among in-bounds solutions
        'avg_mse': MSE of the weighted average solution
        'saturation_alpha': Alpha where saturation was detected
        'passing_alphas': Alpha values that passed all criteria
    """
    all_sce = []
    all_mse = []
    bounds_mask = []

    # Set up CV if requested
    if use_cv:
        if n_splits == "LOO":
            kf = LeaveOneOut()
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Step 1 & 2: Extract without clipping, screen by physical bounds
    for idx, alpha in enumerate(alphas):
        sce = fit_sce_unclipped(X, y, L, alpha)
        all_sce.append(sce)

        if use_cv:
            # Compute CV-based MSE
            fold_mse_list = []
            for tr, val in kf.split(X):
                sce_fold = fit_sce_unclipped(X[tr], y[tr], L, alpha)
                y_val_pred = X[val] @ sce_fold
                fold_mse_list.append(np.mean((y[val] - y_val_pred) ** 2))
            mse = np.mean(fold_mse_list)
        else:
            # Full-data MSE (faster)
            mse = compute_mse(X, y, sce)

        all_mse.append(mse)
        in_bounds = np.all(sce >= sce_bounds_min) and np.all(sce <= sce_bounds_max)
        bounds_mask.append(in_bounds)

        if progress_callback:
            progress_callback((idx + 1) / len(alphas))

    all_sce = np.array(all_sce)
    all_mse = np.array(all_mse)
    bounds_mask = np.array(bounds_mask)

    if not np.any(bounds_mask):
        return {
            'sce_avg': None, 'sce_std': None, 'best_alpha': None,
            'all_sce': all_sce, 'all_mse': all_mse, 'bounds_mask': bounds_mask,
            'final_mask': bounds_mask, 'min_mse': None, 'avg_mse': None,
            'saturation_alpha': None, 'passing_alphas': np.array([]),
            'error': 'No solutions within physical bounds'
        }

    # Step 3: Find first in-bounds solution and minimum MSE
    in_bounds_indices = np.where(bounds_mask)[0]
    start_idx = in_bounds_indices[0]

    screened_mse = np.where(bounds_mask, all_mse, np.inf)
    min_idx = np.argmin(screened_mse)
    min_screened_mse = screened_mse[min_idx]
    best_alpha = alphas[min_idx]  # CV optimal = min MSE among in-bounds

    # Step 4 & 5: Find saturation point
    log_alphas = np.log10(alphas)
    log_mse = np.log10(all_mse)
    mse_threshold_for_saturation = mse_saturation_factor * min_screened_mse

    saturation_idx = len(alphas) - 1  # Default: end of array

    for i in range(min_idx + 1, len(alphas) - 1):
        if not bounds_mask[i]:
            continue

        # Only check for saturation after MSE > threshold
        if all_mse[i] < mse_threshold_for_saturation:
            continue

        # Compute local derivative using central difference
        d_log_mse = log_mse[i + 1] - log_mse[i - 1]
        d_log_alpha = log_alphas[i + 1] - log_alphas[i - 1]
        deriv = d_log_mse / d_log_alpha

        if abs(deriv) < mse_deriv_threshold:
            saturation_idx = i
            break

    saturation_idx = max(saturation_idx, min_idx)

    # Create final mask: from start_idx to saturation_idx, in-bounds only
    final_mask = np.zeros(len(alphas), dtype=bool)
    for i in range(start_idx, saturation_idx + 1):
        if bounds_mask[i]:
            final_mask[i] = True

    passing_sce = all_sce[final_mask]
    passing_alphas = alphas[final_mask]
    passing_mse = all_mse[final_mask]

    if len(passing_sce) == 0:
        return {
            'sce_avg': None, 'sce_std': None, 'best_alpha': best_alpha,
            'all_sce': all_sce, 'all_mse': all_mse, 'bounds_mask': bounds_mask,
            'final_mask': final_mask, 'min_mse': min_screened_mse, 'avg_mse': None,
            'saturation_alpha': alphas[saturation_idx], 'passing_alphas': np.array([]),
            'error': 'No solutions in valid range'
        }

    # Step 6: Weighted average using 1/MSE as weights
    weights = 1.0 / passing_mse
    weights = weights / np.sum(weights)

    sce_avg = np.sum(weights[:, np.newaxis] * passing_sce, axis=0)

    # Weighted standard deviation
    sce_mean_diff = passing_sce - sce_avg
    sce_var = np.sum(weights[:, np.newaxis] * sce_mean_diff**2, axis=0)
    sce_std = np.sqrt(sce_var)

    avg_mse = compute_mse(X, y, sce_avg)
    saturation_alpha = alphas[saturation_idx]

    return {
        'sce_avg': sce_avg,
        'sce_std': sce_std,
        'best_alpha': best_alpha,
        'all_sce': all_sce,
        'all_mse': all_mse,
        'bounds_mask': bounds_mask,
        'final_mask': final_mask,
        'min_mse': min_screened_mse,
        'avg_mse': avg_mse,
        'saturation_alpha': saturation_alpha,
        'passing_alphas': passing_alphas,
        'n_averaged': len(passing_alphas),
    }


def read_uploaded_data(eqe_file, gen_file, sun_file, x1, x2):
    """Load and process uploaded files or local file paths."""
    try:
        # Reset file position for uploaded files before reading
        if hasattr(eqe_file, 'seek'):
            eqe_file.seek(0)
        if hasattr(gen_file, 'seek'):
            gen_file.seek(0)
        if hasattr(sun_file, 'seek'):
            sun_file.seek(0)

        # Handle both file paths (strings) and uploaded file objects
        # np.loadtxt can handle both directly
        abs_data = np.loadtxt(gen_file)
        pos_raw, gen_raw = abs_data[:, 0], abs_data[:, 1:]

        # Extract wavelengths from generation file header
        gen_wavelengths = None
        if isinstance(gen_file, str):
            # Local file - can read header
            with open(gen_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if is_wavelength_header(line):
                        gen_wavelengths = extract_wavelengths_from_line(line, gen_raw.shape[1])
                        if gen_wavelengths is not None:
                            break
        else:
            # Uploaded file - try to read header
            try:
                gen_file.seek(0)
                content = gen_file.read().decode('utf-8')
                gen_file.seek(0)
                lines = content.split('\n')
                for line in lines:
                    if is_wavelength_header(line):
                        gen_wavelengths = extract_wavelengths_from_line(line, gen_raw.shape[1])
                        if gen_wavelengths is not None:
                            break
            except:
                pass

        # CTL interface boundaries
        idx_x1s = np.where(pos_raw == x1)[0]
        idx_x2s = np.where(pos_raw == x2)[0]
        idx_x1 = (idx_x1s[1] if idx_x1s.size >= 2 else
                  (idx_x1s[0] if idx_x1s.size == 1 else np.argmin(np.abs(pos_raw - x1))))
        idx_x2 = (idx_x2s[0] if idx_x2s.size >= 1 else
                  np.argmin(np.abs(pos_raw - x2)))

        mask = (pos_raw > x1) & (pos_raw < x2)
        mask[[idx_x1, idx_x2]] = True
        pos = pos_raw[mask]; gen = gen_raw[mask]

        unique_mask = np.concatenate(([True], pos[1:] != pos[:-1]))
        pos, gen = pos[unique_mask], gen[unique_mask]

        # Reset file position for uploaded files before reading
        if hasattr(eqe_file, 'seek'):
            eqe_file.seek(0)
        if hasattr(sun_file, 'seek'):
            sun_file.seek(0)

        eqe = np.loadtxt(eqe_file, ndmin=2)
        sun_spec = np.loadtxt(sun_file, ndmin=2)
        return pos, gen, eqe, sun_spec, gen_wavelengths
    except Exception as e:
        st.error(f"Error reading data: {str(e)}")
        return None, None, None, None, None


def prepare_input(pos, gen, eqe, sun_spec, use_white_light, gen_wavelengths=None, wl_min=None, wl_max=None, use_first_derivative=False, is_photon_flux=False):
    """Build feature matrix X, response vector y, and regularization L.

    Parameters:
    -----------
    pos : array
        Position array (depth in nm)
    gen : array
        Generation profile data (n_positions x n_wavelengths)
    eqe : array
        EQE data with columns [wavelength, EQE]
    sun_spec : array
        Solar spectrum with columns [wavelength, flux] OR delta photon flux with columns [wavelength, photons/s/cm²]
    use_white_light : bool
        Whether to use white light (flat spectrum)
    gen_wavelengths : array, optional
        Wavelengths from generation file header. If provided, EQE will be interpolated to these wavelengths.
    wl_min : float, optional
        Minimum wavelength (nm) to include in analysis
    wl_max : float, optional
        Maximum wavelength (nm) to include in analysis
    use_first_derivative : bool, optional
        If True, use first derivative operator for regularization; if False, use second derivative (default: False)
    is_photon_flux : bool, optional
        If True, sun_spec contains photon flux [photons/s/cm²] instead of spectrum [W/m²/nm] (default: False)

    Returns:
    --------
    X : array
        Feature matrix
    y : array
        Response vector
    L : array
        Regularization matrix
    (lam, pos, photon_flux, gen) : tuple
        Wavelengths, positions, incident photon flux, and filtered generation profile
    """
    if use_white_light and not is_photon_flux:
        sun_spec = sun_spec.copy()  # Don't modify original
        sun_spec[:, 1] = 1  # WHITE light

    h = pc['Planck constant'][0]
    c_nm = pc['speed of light in vacuum'][0] * 1e9

    # Get EQE data
    eqe_wavelengths, eqe_vals = eqe[:, 0], eqe[:, 1]

    # If generation wavelengths are provided, interpolate EQE to match
    if gen_wavelengths is not None and len(gen_wavelengths) == gen.shape[1]:
        st.info(f"📊 Interpolating EQE from {len(eqe_wavelengths)} points ({eqe_wavelengths.min():.0f}-{eqe_wavelengths.max():.0f} nm) "
                f"to {len(gen_wavelengths)} generation wavelengths ({gen_wavelengths.min():.0f}-{gen_wavelengths.max():.0f} nm)")

        # Interpolate EQE onto generation wavelengths
        interp_eqe = interp1d(eqe_wavelengths, eqe_vals, kind='linear',
                             bounds_error=False, fill_value=0)
        lam = gen_wavelengths
        eqe_vals_interp = interp_eqe(lam)
    else:
        # Use EQE wavelengths as-is (original behavior)
        lam = eqe_wavelengths
        eqe_vals_interp = eqe_vals
        if gen_wavelengths is None:
            st.warning("⚠️ Could not extract wavelengths from generation file header. "
                      "Assuming EQE and generation wavelengths match exactly.")

    # Apply wavelength filter if specified
    if wl_min is not None or wl_max is not None:
        wl_mask = np.ones(len(lam), dtype=bool)
        if wl_min is not None:
            wl_mask &= (lam >= wl_min)
        if wl_max is not None:
            wl_mask &= (lam <= wl_max)

        n_filtered = np.sum(~wl_mask)
        if n_filtered > 0:
            st.info(f"🔍 Wavelength filtering: excluding {n_filtered} wavelengths outside "
                   f"[{wl_min if wl_min else 'min'}, {wl_max if wl_max else 'max'}] nm range. "
                   f"Using {np.sum(wl_mask)} wavelengths from {lam[wl_mask].min():.0f} to {lam[wl_mask].max():.0f} nm.")

        # Apply mask
        lam = lam[wl_mask]
        eqe_vals_interp = eqe_vals_interp[wl_mask]
        gen = gen[:, wl_mask]  # Filter generation columns

    # Build feature matrix with filtered data
    dp = np.diff(pos)
    weights = np.concatenate(([0.5*dp[0]], 0.5*(dp[:-1]+dp[1:]), [0.5*dp[-1]]))
    X = (gen / 1e21).T * weights

    # Interpolate spectrum/photon_flux onto the target wavelengths
    interp_flux = interp1d(sun_spec[:, 0], sun_spec[:, 1], bounds_error=False, fill_value=0)
    if is_photon_flux:
        # Photon flux file: values are already in photons/s/cm², just scale to 10^18
        photon_flux = interp_flux(lam) / 1e18
    else:
        # Spectrum file: convert from W/m²/nm to 10^18 photons/s/cm²
        photon_flux = interp_flux(lam) / (h * c_nm / lam) / 1e18
    y = eqe_vals_interp * photon_flux

    N = pos.size

    # Build regularization matrix L based on derivative order
    if use_first_derivative:
        # First derivative operator: penalizes gradient/slope
        L = np.zeros((N-1, N))
        for i in range(N-1):
            d = pos[i+1] - pos[i]
            L[i, i] = -1.0/d
            L[i, i+1] = 1.0/d
    else:
        # Second derivative operator: penalizes curvature (default)
        L = np.zeros((N-2, N))
        for i in range(1, N-1):
            d1, d2 = pos[i]-pos[i-1], pos[i+1]-pos[i]
            L[i-1, i-1] =  2.0/(d1*(d1+d2))
            L[i-1, i  ] = -2.0/(d1*d2)
            L[i-1, i+1] =  2.0/(d2*(d1+d2))

    return X, y, L, (lam, pos, photon_flux, gen)


# Get available files from resources
eqe_files = list_files_in_directory("resources/eqe data")
gen_files = list_files_in_directory("resources/gen data")
default_sun_path = "resources/Sunspectrum.sp"

# --- Sidebar: Data selection ---
st.sidebar.header("📁 Data selection")

st.sidebar.subheader("EQE data")
eqe_source = st.sidebar.radio("Source:", ["Local file", "Upload"], key="eqe_source", horizontal=True)
if eqe_source == "Local file" and eqe_files:
    eqe_selected_list = st.sidebar.multiselect("Select EQE file(s):", eqe_files, key="eqe_select",
                                        help="Select one or more EQE files for batch processing")
    eqe_file_list = [f"resources/eqe data/{f}" for f in eqe_selected_list] if eqe_selected_list else []
else:
    uploaded_eqe = st.sidebar.file_uploader("Upload EQE data", type=['txt', 'csv', 'dat'],
                                     accept_multiple_files=True, key="eqe_upload",
                                     help="Upload one or more EQE files for batch processing")
    eqe_file_list = uploaded_eqe if uploaded_eqe else []

# Show count of selected files
if eqe_file_list:
    st.sidebar.caption(f"{len(eqe_file_list)} EQE file(s) selected")

# Preview button for EQE
show_eqe_preview = len(eqe_file_list) >= 1 and st.sidebar.button("Preview EQE", key="preview_eqe")
# For compatibility, set eqe_file to first file or None
eqe_file = eqe_file_list[0] if eqe_file_list else None

st.sidebar.subheader("Generation profile")
gen_source = st.sidebar.radio("Source:", ["Local file", "Upload"], key="gen_source", horizontal=True)
if gen_source == "Local file" and gen_files:
    gen_selected = st.sidebar.selectbox("Select generation file:", [""] + gen_files, key="gen_select")
    gen_file = f"resources/gen data/{gen_selected}" if gen_selected else None
else:
    gen_file = st.sidebar.file_uploader("Upload generation profile", type=['txt', 'csv', 'dat'], key="gen_upload")

    # Show format requirements for uploaded files
    st.sidebar.info("""
    📄 **File Format Requirements:**
    - Header: `x (nm)` followed by wavelengths
    - Data: `position(nm)` then generation values
    - Tab or space separated
    """)

# Preview button (shows preview in main area)
show_gen_preview = gen_file and st.sidebar.button("Preview generation", key="preview_gen")

st.sidebar.subheader("Incident spectrum")
sun_source = st.sidebar.radio("Source:", ["Default (Sunspectrum.sp)", "Upload spectrum", "Photon flux file"], key="sun_source")
is_photon_flux = (sun_source == "Photon flux file")
if sun_source == "Default (Sunspectrum.sp)":
    if Path(default_sun_path).exists():
        sun_file = default_sun_path
        st.sidebar.success("✓ Using default AM1.5G spectrum")
    else:
        st.sidebar.error("Default spectrum not found!")
        sun_file = None
elif sun_source == "Upload spectrum":
    sun_file = st.sidebar.file_uploader("Upload spectrum", type=['txt', 'csv', 'dat', 'sp'], key="sun_upload")
else:
    sun_file = st.sidebar.file_uploader("Upload photon flux file", type=['txt', 'csv', 'dat'], key="photon_flux_upload",
                                help="File with wavelength [nm] and photon flux [photons/s/cm²]")
    if sun_file:
        st.sidebar.info("📊 Using photon flux directly")

# Display EQE preview (full width outside columns)
if 'show_eqe_preview' in locals() and show_eqe_preview and eqe_file_list:
    with st.expander("📊 EQE data preview", expanded=True):
        with st.spinner("Loading preview..."):
            if len(eqe_file_list) == 1:
                fig, wls, eqe_vals = preview_eqe_data(eqe_file_list[0])
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # Batch preview: overlay all EQE files on one plot
                fig = go.Figure()
                colors = ['darkblue', 'darkgreen', 'darkred', 'purple', 'orange', 'brown', 'pink', 'gray']
                for idx, f in enumerate(eqe_file_list):
                    try:
                        data = np.loadtxt(f)
                        wavelengths = data[:, 0]
                        eqe_values = data[:, 1]
                        name = Path(f).stem if isinstance(f, str) else f.name.rsplit('.', 1)[0]
                        fig.add_trace(go.Scatter(
                            x=wavelengths, y=eqe_values,
                            mode='lines+markers',
                            line=dict(color=colors[idx % len(colors)], width=2),
                            marker=dict(size=3),
                            name=name,
                            hovertemplate=f'<b>{name}</b><br>λ: %{{x:.1f}} nm<br>EQE: %{{y:.4f}}<extra></extra>'
                        ))
                    except Exception as e:
                        st.warning(f"Could not load {f}: {e}")
                fig.update_layout(
                    title=dict(text='External Quantum Efficiency (EQE)', font=dict(size=18, family='Arial Black')),
                    xaxis=dict(title='Wavelength (nm)', gridcolor='lightgray'),
                    yaxis=dict(title='EQE (-)', gridcolor='lightgray'),
                    template='plotly_white',
                    hovermode='x unified',
                    height=400,
                    plot_bgcolor='white',
                    legend=dict(yanchor='top', y=1, xanchor='left', x=1.02),
                    margin=dict(l=60, r=150, t=60, b=60)
                )
                st.plotly_chart(fig, use_container_width=True)

# Display generation profile preview (full width outside columns)
if 'show_gen_preview' in locals() and show_gen_preview and gen_file:
    with st.expander("📊 Generation profile preview", expanded=True):
        with st.spinner("Loading preview..."):
            fig, z_pos, gen_data, wls, total_gen = preview_generation_profile(gen_file)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                col_info1, col_info2, col_info3, col_info4, col_info5 = st.columns(5)
                col_info1.metric("Depth points", len(z_pos))
                col_info2.metric("Wavelengths", gen_data.shape[1])
                col_info3.metric("Depth range", f"{z_pos.min():.1f} - {z_pos.max():.1f} nm")
                if len(wls) == gen_data.shape[1]:  # Check if wavelengths were parsed
                    col_info4.metric("Wavelength range", f"{wls.min():.0f} - {wls.max():.0f} nm")
                else:
                    col_info4.metric("Wavelength range", "N/A")
                col_info5.metric("Max Generation", f"{(gen_data.max()/1e21):.2f}×10²¹ cm⁻³s⁻¹")

if TMM_AVAILABLE:
    st.markdown("---")
    generation_profile_creator()

# --- Sidebar: Settings ---
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Settings")

# Row 1: Boundaries
sb_col1, sb_col2 = st.sidebar.columns(2)
with sb_col1:
    x1 = st.number_input("x1 (nm)", value=10, min_value=0, help="Front CTL boundary")
with sb_col2:
    x2 = st.number_input("x2 (nm)", value=550, min_value=0, help="Back CTL boundary")

# Row 2: Alpha range
sb_col3, sb_col4, sb_col5 = st.sidebar.columns(3)
with sb_col3:
    alpha_min = st.number_input("log₁₀(α) min", value=0, help="Min regularization strength")
with sb_col4:
    alpha_max = st.number_input("log₁₀(α) max", value=14, help="Max regularization strength")
with sb_col5:
    n_alphas = st.number_input("# alphas", value=1000, min_value=100, max_value=2000, help="Alpha values to test")

# Processing options (stacked vertically in sidebar)
st.sidebar.markdown("**Processing options**")
use_white = st.sidebar.checkbox("White light", value=True, help="Override with flat spectrum (enable for white light generation profiles)")
use_weighted_avg = st.sidebar.checkbox("Weighted avg", value=False, help="Use 1/MSE weighted averaging instead of CV optimal (incompatible with clipping)")
# Bounds screening (incompatible with weighted averaging, which has its own bounds)
if not use_weighted_avg:
    screen_out_of_bounds = st.sidebar.checkbox("Screen by bounds", value=True,
                                       help="Exclude solutions with SCE outside bounds from CV optimal search")
    if screen_out_of_bounds:
        sb_sc1, sb_sc2 = st.sidebar.columns(2)
        with sb_sc1:
            screen_bounds_min = st.number_input("SCE min", value=0.0, format="%.3f", key="screen_min",
                                               help="Lower bound for valid SCE")
        with sb_sc2:
            screen_bounds_max = st.number_input("SCE max", value=1.0, format="%.3f", key="screen_max",
                                               help="Upper bound for valid SCE")
    else:
        screen_bounds_min, screen_bounds_max = 0.0, 1.0
else:
    screen_out_of_bounds = False
    screen_bounds_min, screen_bounds_max = -0.001, 1.001
use_clipping = st.sidebar.checkbox("Clip 0-1", value=False, disabled=use_weighted_avg,
                           help="Constrain SCE to 0-1 range" + (" (disabled with weighted avg)" if use_weighted_avg else ""))
use_bounded_opt = st.sidebar.checkbox("Bounded opt", value=False, disabled=use_weighted_avg,
                              help="Use 0-1 constrained optimization" + (" (disabled with weighted avg)" if use_weighted_avg else ""))
use_second_derivative = st.sidebar.checkbox("2nd deriv.", value=False, help="Use 2nd derivative regularization (curvature) instead of 1st (slope)")
use_first_derivative = not use_second_derivative  # Flip logic: default is 1st derivative
cv_options = [3, 4, 5, 6, 7, 8, 9, 10, "LOO"]
n_splits = st.sidebar.selectbox("CV folds", cv_options, index=2, help="Cross-validation folds (LOO = Leave-One-Out)")

# Override clipping settings when weighted averaging is enabled
if use_weighted_avg:
    use_clipping = False
    use_bounded_opt = False

# Weighted averaging settings (only show when enabled)
if use_weighted_avg:
    st.sidebar.markdown("**Weighted Averaging Settings:**")
    sb_wa1, sb_wa2 = st.sidebar.columns(2)
    with sb_wa1:
        sce_bounds_min = st.number_input("SCE min", value=-0.001, format="%.3f",
                                         help="Lower bound for valid SCE (with numerical tolerance)")
    with sb_wa2:
        sce_bounds_max = st.number_input("SCE max", value=1.001, format="%.3f",
                                         help="Upper bound for valid SCE (with numerical tolerance)")
    sb_wa3, sb_wa4 = st.sidebar.columns(2)
    with sb_wa3:
        mse_saturation_factor = st.number_input("MSE factor", value=10.0, format="%.1f",
                                                help="Check saturation after MSE > factor × min_MSE")
    with sb_wa4:
        mse_deriv_threshold = st.number_input("Deriv. threshold", value=0.01,
                                              format="%.3f", help="Saturation when |d(log MSE)/d(log α)| < threshold")
    use_cv_for_mse = st.sidebar.checkbox("Use CV for MSE", value=True,
                                 help="Use k-fold CV to compute MSE (more robust to overfitting)")
else:
    sce_bounds_min, sce_bounds_max = -0.001, 1.001
    mse_saturation_factor, mse_deriv_threshold = 10.0, 0.01
    use_cv_for_mse = True

# Optional features
use_wavelength_filter = st.sidebar.checkbox("Restrict λ range", value=False, help="Limit analysis to wavelength range")
if use_wavelength_filter:
    sb_wl1, sb_wl2 = st.sidebar.columns(2)
    with sb_wl1:
        wl_min = st.number_input("λ min", value=400, min_value=300, max_value=1200)
    with sb_wl2:
        wl_max = st.number_input("λ max", value=800, min_value=300, max_value=1200)
else:
    wl_min, wl_max = None, None

use_weighted_fitting = st.sidebar.checkbox("Weighted fitting", value=False, help="Prioritize a wavelength region")
if use_weighted_fitting:
    sb_w1, sb_w2 = st.sidebar.columns(2)
    with sb_w1:
        weight_wl_min = st.number_input("Weight λ min", value=430, min_value=300, max_value=1200)
    with sb_w2:
        weight_wl_max = st.number_input("Weight λ max", value=445, min_value=300, max_value=1200)
    weight_factor = st.sidebar.number_input("Factor", value=10.0, min_value=1.0, max_value=1000000.0)
else:
    weight_wl_min, weight_wl_max, weight_factor = None, None, None

# Bounds screening defaults (set by UI above, in processing options)
# When weighted averaging is active, screening is handled by its own bounds

# Clipping penalty disabled - use screen by bounds instead
use_clipping_penalty = False
clipping_penalty_weight = 0.0

# Oscillation penalty disabled - set defaults
use_oscillation_penalty = False
oscillation_threshold = 0
oscillation_penalty_weight = 0.0

st.sidebar.markdown("---")
run_analysis = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)
if st.sidebar.button("Open Manual Fitting", type="secondary", use_container_width=True):
    st.session_state.manual_fitting_open = True

# Manual Fitting Section (right below buttons)
if st.session_state.get('manual_fitting_open', False) and eqe_file and gen_file and sun_file:
    with st.expander("Manual SCE Explorer", expanded=True):
        if MANUAL_FITTING_AVAILABLE:
            # Load data if not already loaded for manual fitting
            if 'manual_data_loaded' not in st.session_state or st.session_state.get('manual_reload_needed', True):
                with st.spinner("Loading data for manual fitting..."):
                    pos_manual, gen_manual, eqe_manual, sun_spec_manual, gen_wavelengths_manual = read_uploaded_data(
                        eqe_file, gen_file, sun_file, x1, x2
                    )

                    if pos_manual is None:
                        st.error("Failed to load data")
                        st.stop()

                    # Store data in session state
                    st.session_state.pos_manual = pos_manual
                    st.session_state.gen_manual = gen_manual
                    st.session_state.eqe_manual = eqe_manual
                    st.session_state.sun_spec_manual = sun_spec_manual
                    st.session_state.gen_wavelengths_manual = gen_wavelengths_manual
                    st.session_state.manual_data_loaded = True
                    st.session_state.manual_reload_needed = False

                    st.success(f"✅ Data loaded: {len(pos_manual)} depth points, {gen_manual.shape[1]} generation wavelengths, {eqe_manual.shape[0]} EQE wavelengths")

            # Retrieve data from session state
            pos_manual = st.session_state.pos_manual
            gen_manual = st.session_state.gen_manual
            eqe_manual = st.session_state.eqe_manual
            sun_spec_manual = st.session_state.sun_spec_manual
            gen_wavelengths_manual = st.session_state.gen_wavelengths_manual

            if st.button("Close", key="close_manual"):
                st.session_state.manual_fitting_open = False
                st.rerun()

            # Call the manual fitting function
            manual_sce_fitting_tab(
                pos=pos_manual,
                gen=gen_manual,
                eqe=eqe_manual,
                sun_spec=sun_spec_manual,
                use_white_light=use_white,
                gen_wavelengths=gen_wavelengths_manual,
                wl_min=wl_min,
                wl_max=wl_max,
                is_photon_flux=is_photon_flux
            )
        else:
            st.error("Manual SCE fitting module not available. Please ensure manual_sce_fitting.py is in the same directory as app.py")

# Separator before automatic analysis
if st.session_state.get('manual_fitting_open', False):
    st.markdown("---")

# Main content - Automatic Analysis (supports batch processing)
if run_analysis and eqe_file_list and gen_file and sun_file:
    n_files = len(eqe_file_list)
    is_batch = n_files > 1

    if is_batch:
        st.subheader(f"Batch processing: {n_files} EQE files")

    # Dictionary to store results for each file
    batch_results = {}

    # First, load the generation profile (shared across all files)
    with st.spinner("Loading generation profile..."):
        # Load gen data once (we'll reload EQE data per file)
        pos_init, gen_init, _, sun_spec, gen_wavelengths = read_uploaded_data(
            eqe_file_list[0], gen_file, sun_file, x1, x2
        )
        if pos_init is None:
            st.stop()

    # Process each EQE file
    use_same_alpha = False  # Feature removed - each file uses its own optimal alpha
    shared_alpha = None
    alphas = np.logspace(alpha_min, alpha_max, n_alphas)
    if n_splits == "LOO":
        kf = LeaveOneOut()
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for file_idx, eqe_file_item in enumerate(eqe_file_list):
        # Get file name for display
        if isinstance(eqe_file_item, str):
            file_name = Path(eqe_file_item).name
        else:
            file_name = eqe_file_item.name

        with st.spinner(f"Processing {file_name} ({file_idx + 1}/{n_files})..."):
            # Load data for this EQE file
            pos, gen, eqe, _, _ = read_uploaded_data(eqe_file_item, gen_file, sun_file, x1, x2)

            if pos is None:
                st.error(f"Failed to load {file_name}")
                continue

            # Store original EQE data
            eqe_original_wavelengths = eqe[:, 0].copy()
            eqe_original_values = eqe[:, 1].copy()

            # Prepare input
            X, y, L, (lam, pos, inc_flux, gen_filtered) = prepare_input(
                pos, gen, eqe, sun_spec, use_white, gen_wavelengths,
                wl_min, wl_max, use_first_derivative, is_photon_flux
            )

            # Create wavelength weights for weighted fitting
            sample_weights = np.ones(len(lam))
            if weight_wl_min is not None and weight_wl_max is not None and weight_factor is not None:
                weight_mask = (lam >= weight_wl_min) & (lam <= weight_wl_max)
                n_weighted = np.sum(weight_mask)
                if n_weighted == 0:
                    st.error(f"No wavelengths found in weight range for {file_name}")
                    continue
                sample_weights[weight_mask] = weight_factor

            # Apply weights
            sqrt_weights = np.sqrt(sample_weights)
            X_weighted = X * sqrt_weights[:, np.newaxis]
            y_weighted = y * sqrt_weights

            # Weighted averaging mode vs CV mode
            if use_weighted_avg:
                # Use weighted MSE averaging approach
                if is_batch:
                    progress_text = st.empty()
                    progress_text.text(f"Computing weighted average for {file_name}...")
                progress_bar = st.progress(0)

                # Run weighted averaging with progress callback
                wa_result = weighted_mse_averaging(
                    X_weighted, y_weighted, L, alphas,
                    sce_bounds_min=sce_bounds_min,
                    sce_bounds_max=sce_bounds_max,
                    mse_saturation_factor=mse_saturation_factor,
                    mse_deriv_threshold=mse_deriv_threshold,
                    use_cv=use_cv_for_mse,
                    n_splits=n_splits,
                    progress_callback=lambda p: progress_bar.progress(p)
                )
                progress_bar.empty()
                if is_batch and 'progress_text' in dir():
                    progress_text.empty()

                if wa_result.get('error'):
                    st.error(f"{file_name}: {wa_result['error']}")
                    continue

                best_alpha = wa_result['best_alpha']
                # Convert all_mse array to dict format for compatibility
                mse = {a: wa_result['all_mse'][i] for i, a in enumerate(alphas)}
                mse_pure = mse.copy()
                penalty_values = {a: 0.0 for a in alphas}
                oscillation_values = {a: 0.0 for a in alphas}

                # Store weighted averaging specific results
                wa_results = wa_result

                n_avg = wa_result.get('n_averaged', 0)
                st.success(f"**{file_name}**: CV optimal α = {best_alpha:.2e}, "
                          f"Averaged {n_avg} solutions (α: {wa_result['passing_alphas'].min():.2e} to {wa_result['saturation_alpha']:.2e})")

                wa_results = wa_result  # Store for later use

            # Cross-validation for optimal alpha (standard mode)
            elif use_same_alpha and shared_alpha is not None:
                wa_results = None  # Not using weighted averaging
                # Use the shared alpha from first file
                best_alpha = shared_alpha
                mse = {}
                mse_pure = {}
                penalty_values = {}
                oscillation_values = {}
                # Compute MSE for this file at the shared alpha
                fold_mse_list = []
                for tr, val in kf.split(X_weighted):
                    m = CustomRidgeDirect(alpha=shared_alpha, L=L, constraint=use_clipping, use_bounded=use_bounded_opt)
                    m.fit(X_weighted[tr], y_weighted[tr])
                    y_val_pred = m.predict(X_weighted[val])
                    fold_mse_list.append(mean_squared_error(y_weighted[val], y_val_pred))
                mse_pure[shared_alpha] = np.mean(fold_mse_list)
                penalty_values[shared_alpha] = 0.0
                oscillation_values[shared_alpha] = 0.0
                mse[shared_alpha] = mse_pure[shared_alpha]
            else:
                # Find optimal alpha for this file
                if is_batch:
                    progress_text = st.empty()
                    progress_text.text(f"Finding optimal α for {file_name}...")
                progress_bar = st.progress(0)
                mse = {}
                mse_pure = {}
                penalty_values = {}
                oscillation_values = {}
                oob_mask = {}  # Out-of-bounds mask for screening

                for idx, a in enumerate(alphas):
                    # Check if solution is in-bounds (for screening mode)
                    if screen_out_of_bounds:
                        # Fit on full data to check bounds
                        m_full = CustomRidgeDirect(alpha=a, L=L, constraint=False, use_bounded=False)
                        m_full.fit(X_weighted, y_weighted)
                        sce_full = m_full.coef_
                        is_out_of_bounds = np.any(sce_full < screen_bounds_min) or np.any(sce_full > screen_bounds_max)
                        oob_mask[a] = is_out_of_bounds

                    fold_mse_list = []
                    for tr, val in kf.split(X_weighted):
                        m = CustomRidgeDirect(alpha=a, L=L, constraint=use_clipping, use_bounded=use_bounded_opt)
                        m.fit(X_weighted[tr], y_weighted[tr])
                        y_val_pred = m.predict(X_weighted[val])
                        fold_mse_list.append(mean_squared_error(y_weighted[val], y_val_pred))

                    mse_pure[a] = np.mean(fold_mse_list)
                    penalty_values[a] = 0.0
                    oscillation_values[a] = 0.0
                    mse[a] = mse_pure[a]
                    progress_bar.progress((idx + 1) / len(alphas))

                # Find best alpha (screening out-of-bounds if enabled)
                if screen_out_of_bounds:
                    # Only consider in-bounds solutions
                    in_bounds_mse = {a: m for a, m in mse.items() if not oob_mask.get(a, False)}
                    if in_bounds_mse:
                        best_alpha = min(in_bounds_mse, key=in_bounds_mse.get)
                    else:
                        st.warning(f"{file_name}: No solutions within [0,1] bounds. Using minimum MSE alpha.")
                        best_alpha = min(mse, key=mse.get)
                else:
                    best_alpha = min(mse, key=mse.get)
                progress_bar.empty()
                if is_batch and 'progress_text' in dir():
                    progress_text.empty()

                # Set shared alpha from first file if using same alpha mode
                if use_same_alpha and shared_alpha is None:
                    shared_alpha = best_alpha
                    st.info(f"Using α = {shared_alpha:.2e} from {file_name} for all files")

                wa_results = None  # Not using weighted averaging

            # Store results for this file
            batch_results[file_name] = {
                'eqe_file': eqe_file_item,
                'X': X,
                'y': y,
                'L': L,
                'lam': lam,
                'pos': pos,
                'inc_flux': inc_flux,
                'gen_filtered': gen_filtered,
                'X_weighted': X_weighted,
                'y_weighted': y_weighted,
                'sample_weights': sample_weights,
                'mse': mse,
                'mse_pure': mse_pure,
                'penalty_values': penalty_values,
                'oscillation_values': oscillation_values,
                'oob_mask': oob_mask if 'oob_mask' in dir() else {},  # Out-of-bounds mask for screening
                'best_alpha': best_alpha,
                'alphas': alphas,
                'eqe_original_wavelengths': eqe_original_wavelengths,
                'eqe_original_values': eqe_original_values,
                'wa_results': wa_results,  # Weighted averaging results (None if not used)
            }

            if not is_batch and not use_weighted_avg:
                weight_str = f" (weighted: {weight_wl_min}-{weight_wl_max} nm, {weight_factor}×)" if weight_factor is not None else ""
                st.success(f"Optimal alpha: **{best_alpha:.2e}** (CV score: {mse[best_alpha]:.2e}){weight_str}")

    # Store batch results in session state
    st.session_state.analysis_complete = True
    st.session_state.is_batch = is_batch
    st.session_state.batch_results = batch_results
    st.session_state.kf = kf
    st.session_state.use_clipping = use_clipping
    st.session_state.use_bounded_opt = use_bounded_opt
    st.session_state.use_same_alpha = use_same_alpha if is_batch else False
    st.session_state.shared_alpha = shared_alpha
    st.session_state.sun_spec = sun_spec
    st.session_state.gen_wavelengths = gen_wavelengths
    st.session_state.use_white = use_white
    st.session_state.wl_min = wl_min
    st.session_state.wl_max = wl_max
    st.session_state.use_first_derivative = use_first_derivative
    st.session_state.weight_wl_min_used = weight_wl_min
    st.session_state.weight_wl_max_used = weight_wl_max
    st.session_state.weight_factor_used = weight_factor
    st.session_state.screen_out_of_bounds = screen_out_of_bounds
    st.session_state.screen_bounds_min = screen_bounds_min
    st.session_state.screen_bounds_max = screen_bounds_max
    st.session_state.use_oscillation_penalty = use_oscillation_penalty
    st.session_state.oscillation_penalty_weight = oscillation_penalty_weight
    st.session_state.oscillation_threshold = oscillation_threshold
    st.session_state.use_weighted_avg = use_weighted_avg
    st.session_state.sce_bounds_min = sce_bounds_min
    st.session_state.sce_bounds_max = sce_bounds_max
    st.session_state.mse_saturation_factor = mse_saturation_factor
    st.session_state.mse_deriv_threshold = mse_deriv_threshold
    st.session_state.use_cv_for_mse = use_cv_for_mse
    # For backward compatibility with single file
    if not is_batch and batch_results:
        first_result = list(batch_results.values())[0]
        st.session_state.alphas = first_result['alphas']
        st.session_state.mse = first_result['mse']
        st.session_state.mse_pure = first_result['mse_pure']
        st.session_state.penalty_values = first_result['penalty_values']
        st.session_state.oscillation_values = first_result['oscillation_values']
        st.session_state.best_alpha = first_result['best_alpha']
        st.session_state.X = first_result['X']
        st.session_state.y = first_result['y']
        st.session_state.L = first_result['L']
        st.session_state.lam = first_result['lam']
        st.session_state.pos = first_result['pos']
        st.session_state.inc_flux = first_result['inc_flux']
        st.session_state.gen_filtered = first_result['gen_filtered']
        st.session_state.eqe_original_wavelengths = first_result['eqe_original_wavelengths']
        st.session_state.eqe_original_values = first_result['eqe_original_values']
        st.session_state.X_weighted = first_result['X_weighted']
        st.session_state.y_weighted = first_result['y_weighted']
        st.session_state.sample_weights = first_result['sample_weights']
        st.session_state.oob_mask = first_result.get('oob_mask', {})
        st.session_state.wa_results = first_result.get('wa_results', None)

    if is_batch:
        st.success(f"Batch processing complete: {len(batch_results)} files processed")

# Display results if analysis has been run
if 'analysis_complete' in st.session_state and st.session_state.analysis_complete:
    # Check if this is batch mode
    is_batch_results = st.session_state.get('is_batch', False)
    batch_results = st.session_state.get('batch_results', {})

    # Retrieve common session state values
    kf = st.session_state.kf
    use_clipping = st.session_state.get('use_clipping', True)
    use_bounded_opt = st.session_state.get('use_bounded_opt', False)
    weight_wl_min_stored = st.session_state.get('weight_wl_min_used', None)
    weight_wl_max_stored = st.session_state.get('weight_wl_max_used', None)
    weight_factor_stored = st.session_state.get('weight_factor_used', None)

    if is_batch_results and len(batch_results) > 1:
        # === BATCH MODE DISPLAY ===
        st.markdown("---")
        st.subheader("Batch Results Comparison")

        # Check if using same alpha mode
        use_same_alpha_mode = st.session_state.get('use_same_alpha', False)
        shared_alpha = st.session_state.get('shared_alpha', None)

        # Get first result for alpha range reference
        first_result = list(batch_results.values())[0]
        alphas = first_result['alphas']

        # Alpha slider for same-alpha mode
        if use_same_alpha_mode and shared_alpha is not None:
            st.subheader("Regularization strength (applied to all files)")

            log_alpha_min = np.log10(alphas.min())
            log_alpha_max = np.log10(alphas.max())
            log_shared_alpha = np.log10(shared_alpha)

            log_current_alpha = st.slider(
                "log₁₀(α) - Adjust to explore different regularization strengths",
                min_value=float(log_alpha_min),
                max_value=float(log_alpha_max),
                value=float(log_shared_alpha),
                step=0.01,
                format="%.2f",
                key="batch_alpha_slider"
            )
            current_alpha = 10**log_current_alpha

            col_info1, col_info2 = st.columns(2)
            col_info1.metric("Current α", f"{current_alpha:.2e}")
            col_info2.metric("Optimal α (from first file)", f"{shared_alpha:.2e}")
        else:
            current_alpha = None  # Each file uses its own optimal alpha

        # Summary table
        summary_data = []
        file_names = list(batch_results.keys())

        # Compute SCE for each file
        sce_profiles = {}
        # Check if weighted averaging mode was used
        use_weighted_avg_batch = st.session_state.get('use_weighted_avg', False)

        for file_name, result in batch_results.items():
            wa_result = result.get('wa_results')

            # Determine which alpha to use and get SCE profile
            if use_weighted_avg_batch and wa_result is not None and wa_result.get('sce_avg') is not None:
                # Use weighted average SCE
                sce_profile = wa_result['sce_avg']
                alpha_to_use = result['best_alpha']
                sce_std = wa_result.get('sce_std')
            else:
                # Standard mode: fit with specific alpha
                if use_same_alpha_mode and current_alpha is not None:
                    alpha_to_use = current_alpha
                else:
                    alpha_to_use = result['best_alpha']

                model = CustomRidgeDirect(
                    alpha=alpha_to_use,
                    L=result['L'],
                    constraint=use_clipping,
                    use_bounded=use_bounded_opt
                )
                model.fit(result['X_weighted'], result['y_weighted'])
                sce_profile = model.coef_
                sce_std = None

            sce_profiles[file_name] = sce_profile

            # Calculate Jsc
            total_gen = trapz_func(result['gen_filtered'], result['lam'], axis=1)
            collected_gen = sce_profile * total_gen
            jsc = trapz_func(collected_gen, result['pos']) * 1.602e-19 * 1e-7 * 1000  # mA/cm²

            # Calculate MSE
            y_fit = result['X'] @ sce_profile
            current_mse = np.mean((result['y'] - y_fit)**2)

            if use_weighted_avg_batch and wa_result is not None:
                n_avg = wa_result.get('n_averaged', 0)
                summary_data.append({
                    'File': file_name,
                    'CV opt α': f"{alpha_to_use:.2e}",
                    '# Averaged': n_avg,
                    'Avg MSE': f"{current_mse:.2e}",
                    'Jsc (mA/cm²)': f"{jsc:.2f}"
                })
            else:
                summary_data.append({
                    'File': file_name,
                    'α used': f"{alpha_to_use:.2e}",
                    'MSE': f"{current_mse:.2e}",
                    'Jsc (mA/cm²)': f"{jsc:.2f}"
                })

        # Display summary table
        st.dataframe(summary_data, use_container_width=True)

        # CV comparison plot
        st.subheader("Cross-Validation Comparison")

        use_oscillation_penalty_batch = False
        any_penalty_active = False

        fig_cv = go.Figure()
        colors = ['darkblue', 'darkgreen', 'darkred', 'purple', 'orange', 'brown', 'pink', 'gray']

        for idx, (file_name, result) in enumerate(batch_results.items()):
            color = colors[idx % len(colors)]
            file_alphas = result['alphas']
            mse = result['mse']
            mse_pure = result.get('mse_pure', mse)  # Fall back to mse if mse_pure not available
            best_alpha = result['best_alpha']
            wa_result = result.get('wa_results')

            alphas_plot = [a for a in file_alphas if a in mse]

            # Weighted averaging mode visualization
            if use_weighted_avg_batch and wa_result is not None:
                mse_values = [mse[a] for a in alphas_plot]
                wa_bounds_mask = wa_result.get('bounds_mask', np.ones(len(file_alphas), dtype=bool))
                wa_final_mask = wa_result.get('final_mask', np.zeros(len(file_alphas), dtype=bool))

                # Plot out-of-bounds points (grey)
                out_of_bounds_mask = ~wa_bounds_mask[:len(alphas_plot)]
                if np.any(out_of_bounds_mask):
                    oob_alphas = np.array(alphas_plot)[out_of_bounds_mask]
                    oob_mse = np.array(mse_values)[out_of_bounds_mask]
                    fig_cv.add_trace(go.Scatter(
                        x=oob_alphas,
                        y=oob_mse,
                        mode='markers',
                        name=f'{file_name} (out of bounds)',
                        marker=dict(color='lightgray', size=4),
                        showlegend=False,
                        hovertemplate=f'{file_name} (out of bounds)<br>α: %{{x:.2e}}<br>MSE: %{{y:.2e}}<extra></extra>'
                    ))

                # Plot in-bounds but not averaged (hollow circles)
                in_bounds_not_avg = wa_bounds_mask[:len(alphas_plot)] & ~wa_final_mask[:len(alphas_plot)]
                if np.any(in_bounds_not_avg):
                    ib_alphas = np.array(alphas_plot)[in_bounds_not_avg]
                    ib_mse = np.array(mse_values)[in_bounds_not_avg]
                    fig_cv.add_trace(go.Scatter(
                        x=ib_alphas,
                        y=ib_mse,
                        mode='markers',
                        name=f'{file_name} (in bounds)',
                        marker=dict(color=color, size=5, symbol='circle-open'),
                        showlegend=False,
                        hovertemplate=f'{file_name} (in bounds, not averaged)<br>α: %{{x:.2e}}<br>MSE: %{{y:.2e}}<extra></extra>'
                    ))

                # Plot MSE curve through in-bounds points
                ib_mask = wa_bounds_mask[:len(alphas_plot)]
                ib_all_alphas = np.array(alphas_plot)[ib_mask]
                ib_all_mse = np.array(mse_values)[ib_mask]
                if len(ib_all_alphas) > 0:
                    fig_cv.add_trace(go.Scatter(
                        x=ib_all_alphas,
                        y=ib_all_mse,
                        mode='lines',
                        name=f'{file_name}',
                        line=dict(color=color, width=1.5),
                        hovertemplate=f'{file_name}<br>α: %{{x:.2e}}<br>MSE: %{{y:.2e}}<extra></extra>'
                    ))

                # Highlight averaged range (filled circles)
                avg_mask = wa_final_mask[:len(alphas_plot)]
                avg_alphas = np.array(alphas_plot)[avg_mask]
                avg_mse = np.array(mse_values)[avg_mask]
                if len(avg_alphas) > 0:
                    fig_cv.add_trace(go.Scatter(
                        x=avg_alphas,
                        y=avg_mse,
                        mode='markers',
                        name=f'{file_name} (avg range)',
                        marker=dict(color=color, size=6),
                        showlegend=False,
                        hovertemplate=f'{file_name} (averaged)<br>α: %{{x:.2e}}<br>MSE: %{{y:.2e}}<extra></extra>'
                    ))

                # Mark CV optimal
                fig_cv.add_trace(go.Scatter(
                    x=[best_alpha],
                    y=[mse[best_alpha]],
                    mode='markers',
                    name=f'{file_name} CV opt',
                    marker=dict(color=color, size=10, symbol='star'),
                    showlegend=False,
                    hovertemplate=f'{file_name} CV opt<br>α: %{{x:.2e}}<br>MSE: %{{y:.2e}}<extra></extra>'
                ))

                # Mark weighted average MSE
                avg_mse_val = wa_result.get('avg_mse')
                if avg_mse_val is not None:
                    fig_cv.add_trace(go.Scatter(
                        x=[best_alpha],
                        y=[avg_mse_val],
                        mode='markers',
                        name=f'{file_name} avg MSE',
                        marker=dict(color=color, size=10, symbol='diamond'),
                        showlegend=False,
                        hovertemplate=f'{file_name} weighted avg<br>MSE: %{{y:.2e}}<extra></extra>'
                    ))

            # If any penalty was used, show both curves
            elif any_penalty_active and mse_pure != mse:
                # Pure MSE curve (dashed)
                mse_pure_values = [mse_pure[a] for a in alphas_plot]
                fig_cv.add_trace(go.Scatter(
                    x=alphas_plot,
                    y=mse_pure_values,
                    mode='lines',
                    name=f'{file_name} (pure MSE)',
                    line=dict(color=color, width=1, dash='dash'),
                    hovertemplate=f'{file_name}<br>α: %{{x:.2e}}<br>Pure MSE: %{{y:.2e}}<extra></extra>'
                ))

                # Combined score curve (solid)
                mse_combined_values = [mse[a] for a in alphas_plot]
                fig_cv.add_trace(go.Scatter(
                    x=alphas_plot,
                    y=mse_combined_values,
                    mode='lines',
                    name=f'{file_name} (CV score)',
                    line=dict(color=color, width=2),
                    hovertemplate=f'{file_name}<br>α: %{{x:.2e}}<br>CV Score: %{{y:.2e}}<extra></extra>'
                ))

                # Optimal point marker
                if not use_same_alpha_mode or current_alpha is None:
                    fig_cv.add_trace(go.Scatter(
                        x=[best_alpha],
                        y=[mse[best_alpha]],
                        mode='markers',
                        name=f'{file_name} optimal',
                        marker=dict(color=color, size=10, symbol='star'),
                        showlegend=False,
                        hovertemplate=f'{file_name} optimal<br>α: %{{x:.2e}}<br>CV Score: %{{y:.2e}}<extra></extra>'
                    ))
            else:
                # Single MSE curve - check if screening was used
                screen_oob_batch = st.session_state.get('screen_out_of_bounds', False)
                file_oob_mask = result.get('oob_mask', {})

                mse_values = [mse[a] for a in alphas_plot]

                if screen_oob_batch and file_oob_mask:
                    # Show out-of-bounds points in grey
                    oob_alphas = [a for a in alphas_plot if file_oob_mask.get(a, False)]
                    oob_mse = [mse[a] for a in oob_alphas]
                    if oob_alphas:
                        fig_cv.add_trace(go.Scatter(
                            x=oob_alphas,
                            y=oob_mse,
                            mode='markers',
                            name=f'{file_name} (out of bounds)',
                            marker=dict(color='lightgray', size=4),
                            showlegend=False,
                            hovertemplate=f'{file_name} (out of bounds)<br>α: %{{x:.2e}}<br>MSE: %{{y:.2e}}<extra></extra>'
                        ))

                    # Show in-bounds curve
                    ib_alphas = [a for a in alphas_plot if not file_oob_mask.get(a, False)]
                    ib_mse = [mse[a] for a in ib_alphas]
                    if ib_alphas:
                        fig_cv.add_trace(go.Scatter(
                            x=ib_alphas,
                            y=ib_mse,
                            mode='lines',
                            name=file_name,
                            line=dict(color=color, width=2),
                            hovertemplate=f'{file_name}<br>α: %{{x:.2e}}<br>MSE: %{{y:.2e}}<extra></extra>'
                        ))
                else:
                    # Normal MSE curve
                    fig_cv.add_trace(go.Scatter(
                        x=alphas_plot,
                        y=mse_values,
                        mode='lines',
                        name=file_name,
                        line=dict(color=color, width=2),
                        hovertemplate=f'{file_name}<br>α: %{{x:.2e}}<br>MSE: %{{y:.2e}}<extra></extra>'
                    ))

                # Optimal point marker (only show if not using same alpha with different current)
                if not use_same_alpha_mode or current_alpha is None:
                    fig_cv.add_trace(go.Scatter(
                        x=[best_alpha],
                        y=[mse[best_alpha]],
                        mode='markers',
                        name=f'{file_name} optimal',
                        marker=dict(color=color, size=10, symbol='star'),
                        showlegend=False,
                        hovertemplate=f'{file_name} optimal<br>α: %{{x:.2e}}<br>CV Score: %{{y:.2e}}<extra></extra>'
                    ))

        # Add current alpha vertical line if using same alpha mode
        if use_same_alpha_mode and current_alpha is not None:
            fig_cv.add_vline(x=current_alpha, line_dash="dash", line_color="red",
                            annotation_text=f"Current α = {current_alpha:.2e}")
            # Also show optimal alpha line if different
            if abs(current_alpha - shared_alpha) / shared_alpha > 0.01:
                fig_cv.add_vline(x=shared_alpha, line_dash="dot", line_color="gray",
                                annotation_text=f"Optimal α = {shared_alpha:.2e}")

        # Update title based on options used
        if use_weighted_avg_batch:
            plot_title = 'MSE vs Alpha (Weighted Averaging)'
            yaxis_title = 'MSE'
        else:
            plot_title = 'Cross-Validation: MSE vs Alpha'
            yaxis_title = 'Average MSE'

        fig_cv.update_layout(
            title=dict(text=plot_title, font=dict(size=16, family='Arial Black')),
            xaxis=dict(title='Alpha', type='log', gridcolor='lightgray'),
            yaxis=dict(title=yaxis_title, type='log', gridcolor='lightgray'),
            template='plotly_white',
            hovermode='closest',
            height=500,
            showlegend=True,
            legend=dict(yanchor='top', y=1, xanchor='left', x=1.02),
            margin=dict(r=150)
        )

        st.plotly_chart(fig_cv, use_container_width=True)

        # Comparison plot - SCE profiles overlaid
        st.subheader("SCE Profile Comparison")

        # Store raw SCE profiles for reference
        sce_profiles_raw = {k: v.copy() for k, v in sce_profiles.items()}

        # Option to upload reference SCE profiles for comparison
        with st.expander("Upload reference SCE profiles (e.g., for validation)"):
            ref_sce_files_batch = st.file_uploader(
                "Upload SCE files (2 columns: position in nm, SCE value)",
                type=['txt', 'csv', 'dat'],
                accept_multiple_files=True,
                key="ref_sce_batch",
                help="Upload one or more reference SCE profiles to overlay on the plot"
            )
            ref_x_shift_batch = st.number_input(
                "X-axis shift (nm)",
                value=0.0,
                step=1.0,
                format="%.1f",
                key="ref_x_shift_batch",
                help="Shift the position axis of uploaded reference profiles (positive = shift right)"
            )
            ref_sce_profiles_batch = {}
            if ref_sce_files_batch:
                for ref_file in ref_sce_files_batch:
                    try:
                        ref_data = np.loadtxt(ref_file, comments='#')
                        if ref_data.ndim == 1:
                            st.warning(f"Skipped {ref_file.name}: needs 2 columns")
                            continue
                        ref_sce_profiles_batch[ref_file.name] = {
                            'pos': ref_data[:, 0] + ref_x_shift_batch,
                            'sce': ref_data[:, 1]
                        }
                    except Exception as e:
                        st.warning(f"Could not load {ref_file.name}: {e}")

        # Smoothing controls next to SCE plot
        smooth_col1, smooth_col2, smooth_col3 = st.columns([1, 1, 1])
        with smooth_col1:
            use_smoothing_batch = st.checkbox("Smooth SCE", value=False, key="batch_smooth_checkbox",
                                              help="Apply smoothing filter to remove oscillations")

        # Get max allowed window from first result
        max_window_batch = len(first_result['pos']) // 2 * 2 - 1
        # Initialize default values
        smooth_method_batch = "Savitzky-Golay"
        smooth_window_batch = 51
        smooth_poly_batch = 3
        spline_smoothing_batch = 0.01

        if use_smoothing_batch:
            with smooth_col2:
                smooth_method_batch = st.selectbox("Method", ["Savitzky-Golay", "Smoothing Spline"],
                                                   key="batch_smooth_method",
                                                   help="Savgol: local polynomial. Spline: global fit, preserves peak positions better.")

            if smooth_method_batch == "Savitzky-Golay":
                with smooth_col3:
                    smooth_window_batch = st.slider("Window", min_value=5, max_value=min(501, max_window_batch),
                                                    value=min(51, max_window_batch), step=2, key="batch_smooth_window",
                                                    help="Larger = more smoothing")
                    smooth_poly_batch = st.slider("Poly order", min_value=1, max_value=min(smooth_window_batch-1, 15),
                                                  value=min(3, smooth_window_batch-1), key="batch_smooth_poly",
                                                  help="Lower = smoother, higher = preserves more detail")
            else:  # Smoothing Spline
                with smooth_col3:
                    spline_smoothing_batch = st.slider("Smoothing", min_value=0.0001, max_value=1.0,
                                                       value=0.001, step=0.0001, format="%.4f",
                                                       key="batch_spline_smoothing",
                                                       help="Higher = smoother curve. Preserves peak positions.")

        # Compute smoothed SCE if enabled
        sce_profiles_smoothed = {}
        if use_smoothing_batch:
            for file_name, sce_raw in sce_profiles_raw.items():
                pos_batch = batch_results[file_name]['pos']
                if smooth_method_batch == "Savitzky-Golay":
                    sce_smooth = savgol_filter(sce_raw, smooth_window_batch, smooth_poly_batch)
                else:  # Smoothing Spline
                    n_points = len(sce_raw)
                    s_param = spline_smoothing_batch * n_points
                    spline = UnivariateSpline(pos_batch, sce_raw, s=s_param)
                    sce_smooth = spline(pos_batch)
                if use_clipping:
                    sce_smooth = np.clip(sce_smooth, 0, 1)
                sce_profiles_smoothed[file_name] = sce_smooth

        fig_compare = go.Figure()

        # Color mapping for named colors
        color_map = {'darkblue': (0, 0, 139), 'darkgreen': (0, 100, 0), 'darkred': (139, 0, 0),
                     'purple': (128, 0, 128), 'orange': (255, 165, 0), 'brown': (165, 42, 42),
                     'pink': (255, 192, 203), 'gray': (128, 128, 128)}

        for idx, (file_name, sce_raw) in enumerate(sce_profiles_raw.items()):
            pos = batch_results[file_name]['pos']
            color = colors[idx % len(colors)]

            # Raw SCE line (solid if no smoothing, dashed if smoothing enabled)
            line_style = dict(color=color, width=1, dash='dash') if use_smoothing_batch else dict(color=color, width=2)
            name_suffix = ' (raw)' if use_smoothing_batch else ''
            fig_compare.add_trace(go.Scatter(
                x=pos,
                y=sce_raw,
                mode='lines',
                name=f'{file_name}{name_suffix}',
                line=line_style,
                hovertemplate=f'{file_name}<br>Depth: %{{x:.1f}} nm<br>SCE: %{{y:.4f}}<extra></extra>'
            ))

            # Smoothed SCE line (if smoothing enabled)
            if use_smoothing_batch:
                fig_compare.add_trace(go.Scatter(
                    x=pos,
                    y=sce_profiles_smoothed[file_name],
                    mode='lines',
                    name=f'{file_name} (smoothed)',
                    line=dict(color=color, width=2),
                    hovertemplate=f'{file_name} smoothed<br>Depth: %{{x:.1f}} nm<br>SCE: %{{y:.4f}}<extra></extra>'
                ))

        # Add reference SCE profiles if uploaded
        ref_colors = ['black', 'dimgray', 'darkslategray', 'slategray']
        for idx, (ref_name, ref_data) in enumerate(ref_sce_profiles_batch.items()):
            ref_color = ref_colors[idx % len(ref_colors)]
            fig_compare.add_trace(go.Scatter(
                x=ref_data['pos'],
                y=ref_data['sce'],
                mode='lines',
                name=f'Ref: {ref_name}',
                line=dict(color=ref_color, width=2, dash='dot'),
                hovertemplate=f'{ref_name}<br>Depth: %{{x:.1f}} nm<br>SCE: %{{y:.4f}}<extra></extra>'
            ))

        plot_title = 'SCE Profile Comparison (Raw vs Smoothed)' if use_smoothing_batch else 'SCE Profile Comparison'
        fig_compare.update_layout(
            title=dict(text=plot_title, font=dict(size=18, family='Arial Black')),
            xaxis=dict(title='Depth (nm)', gridcolor='lightgray'),
            yaxis=dict(title='Collection Efficiency (-)', range=[0, 1], gridcolor='lightgray'),
            template='plotly_white',
            hovermode='x unified',
            height=600,
            showlegend=True,
            legend=dict(yanchor='top', y=1, xanchor='left', x=1.02),
            margin=dict(r=150)
        )

        st.plotly_chart(fig_compare, use_container_width=True)

        # EQE comparison plot
        st.subheader("EQE Fit Comparison")

        fig_eqe_compare = go.Figure()

        for idx, (file_name, result) in enumerate(batch_results.items()):
            color = colors[idx % len(colors)]
            lam = result['lam']
            inc_flux = result['inc_flux']
            y = result['y']

            # Measured EQE
            fig_eqe_compare.add_trace(go.Scatter(
                x=lam,
                y=y/inc_flux,
                mode='lines',
                name=f'{file_name} (measured)',
                line=dict(color=color, width=1, dash='dash'),
                hovertemplate=f'{file_name}<br>λ: %{{x:.1f}} nm<br>EQE: %{{y:.4f}}<extra></extra>'
            ))

            # Fitted EQE from SCE profile
            sce_for_fit = sce_profiles_smoothed[file_name] if use_smoothing_batch else sce_profiles_raw[file_name]
            y_fit = result['X'] @ sce_for_fit

            # Determine label based on mode
            if use_weighted_avg_batch:
                fit_label = f'{file_name} (weighted avg fit)'
            elif use_smoothing_batch:
                fit_label = f'{file_name} (smoothed fit)'
            else:
                fit_label = f'{file_name} (fit)'

            fig_eqe_compare.add_trace(go.Scatter(
                x=lam,
                y=y_fit/inc_flux,
                mode='lines',
                name=fit_label,
                line=dict(color=color, width=2),
                hovertemplate=f'{file_name} fit<br>λ: %{{x:.1f}} nm<br>EQE: %{{y:.4f}}<extra></extra>'
            ))

        fig_eqe_compare.update_layout(
            title=dict(text='EQE Comparison (Measured vs Fit)', font=dict(size=16, family='Arial Black')),
            xaxis=dict(title='Wavelength (nm)', gridcolor='lightgray'),
            yaxis=dict(title='EQE', gridcolor='lightgray'),
            template='plotly_white',
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(yanchor='top', y=1, xanchor='left', x=1.02),
            margin=dict(r=150)
        )

        st.plotly_chart(fig_eqe_compare, use_container_width=True)

        # Download batch results
        st.subheader("Download Batch Results")

        # Store default name in session state so it doesn't change on rerun
        if 'batch_results_name' not in st.session_state:
            st.session_state.batch_results_name = f"sce_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        results_name = st.text_input(
            "Results name (for ZIP file)",
            key="batch_results_name",
            help="Enter a unique name for this batch analysis."
        )

        # Create batch ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Analysis info
            info_lines = [
                "SCE Batch Analysis Results",
                "=" * 50,
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Number of files: {len(batch_results)}",
                f"Used same α for all: {use_same_alpha_mode}",
            ]
            if use_same_alpha_mode and current_alpha is not None:
                info_lines.append(f"  - Current α (slider): {current_alpha:.2e}")
                info_lines.append(f"  - Optimal α (from first file): {shared_alpha:.2e}")
            info_lines.extend([
                "",
                "Settings:",
                f"  - Boundaries: x1={x1} nm, x2={x2} nm",
                f"  - White light: {use_white}",
                f"  - Clipping: {use_clipping}",
                f"  - Bounded optimization: {use_bounded_opt}",
                f"  - CV: {'LOO' if n_splits == 'LOO' else f'{n_splits}-fold'}",
                f"  - SCE smoothing: {use_smoothing_batch}",
            ])
            if use_smoothing_batch:
                info_lines.append(f"    - Method: {smooth_method_batch}")
                if smooth_method_batch == "Savitzky-Golay":
                    info_lines.append(f"    - Window size: {smooth_window_batch}")
                    info_lines.append(f"    - Polynomial order: {smooth_poly_batch}")
                else:
                    info_lines.append(f"    - Smoothing parameter: {spline_smoothing_batch:.4f}")
            info_lines.append(f"  - Screen by bounds: {screen_out_of_bounds}")
            if screen_out_of_bounds:
                info_lines.extend([
                    f"    - SCE min: {screen_bounds_min}",
                    f"    - SCE max: {screen_bounds_max}",
                ])
            info_lines.append(f"  - Weighted averaging: {use_weighted_avg}")
            if use_weighted_avg:
                info_lines.extend([
                    f"    - SCE bounds min: {sce_bounds_min}",
                    f"    - SCE bounds max: {sce_bounds_max}",
                    f"    - MSE sat. factor: {mse_saturation_factor}",
                    f"    - Deriv. threshold: {mse_deriv_threshold}",
                    f"    - Use CV for MSE: {use_cv_for_mse}",
                ])
            info_lines.extend([
                "",
                "Results Summary:",
                "-" * 50,
            ])
            for item in summary_data:
                if 'α used' in item:
                    info_lines.append(f"{item['File']}: α={item['α used']}, MSE={item['MSE']}, Jsc={item['Jsc (mA/cm²)']} mA/cm²")
                else:
                    # Weighted averaging mode
                    info_lines.append(f"{item['File']}: CV opt α={item['CV opt α']}, # Averaged={item['# Averaged']}, Avg MSE={item['Avg MSE']}, Jsc={item['Jsc (mA/cm²)']} mA/cm²")

            zf.writestr("analysis_info.txt", "\n".join(info_lines))

            # Export data for each file
            for file_name, result in batch_results.items():
                safe_file_name = "".join(c for c in file_name if c.isalnum() or c in ('_', '-', '.')).strip()

                # Use current alpha if same-alpha mode, otherwise use file's best alpha
                if use_same_alpha_mode and current_alpha is not None:
                    alpha_used = current_alpha
                else:
                    alpha_used = result['best_alpha']

                # 1. SCE profile (include both raw and smoothed if smoothing is enabled)
                sce_raw = sce_profiles_raw[file_name]
                pos = result['pos']

                if use_smoothing_batch and file_name in sce_profiles_smoothed:
                    sce_smooth = sce_profiles_smoothed[file_name]
                    if smooth_method_batch == "Savitzky-Golay":
                        smoothing_note = f"Savgol window={smooth_window_batch}, poly={smooth_poly_batch}"
                    else:
                        smoothing_note = f"Spline s={spline_smoothing_batch:.4f}"
                    # Include both raw and smoothed
                    sce_data = np.column_stack((pos, sce_raw, sce_smooth))
                    sce_buffer = io.StringIO()
                    np.savetxt(sce_buffer, sce_data, fmt='%.6f\t%.6f\t%.6f',
                               header=f'pos(nm)\tSCE_raw (alpha={alpha_used:.2e})\tSCE_smoothed ({smoothing_note})')
                else:
                    sce_data = np.column_stack((pos, sce_raw))
                    sce_buffer = io.StringIO()
                    np.savetxt(sce_buffer, sce_data, fmt='%.6f\t%.6f',
                               header=f'pos(nm)\tSCE (alpha={alpha_used:.2e})')
                zf.writestr(f"SCE_{safe_file_name}.txt", sce_buffer.getvalue())

                # 2. CV data (alpha vs MSE, and optionally pure MSE, in_bounds)
                alphas_file = result['alphas']
                mse_file = result['mse']
                mse_pure_file = result.get('mse_pure', mse_file)

                # Build in_bounds column from wa_results or oob_mask
                file_wa = result.get('wa_results', None)
                file_oob = result.get('oob_mask', {})
                if file_wa is not None:
                    wa_bm = file_wa.get('bounds_mask', np.ones(len(alphas_file), dtype=bool))
                    in_bounds_col = np.array([int(wa_bm[i]) for i in range(len(alphas_file))])
                elif file_oob:
                    in_bounds_col = np.array([int(not file_oob.get(a, False)) for a in alphas_file])
                else:
                    in_bounds_col = np.ones(len(alphas_file), dtype=int)

                # Check if penalties were active (mse_pure differs from mse)
                penalties_active_file = any_penalty_active and mse_pure_file != mse_file

                if penalties_active_file:
                    cv_data = np.column_stack((alphas_file,
                                               [mse_pure_file[a] for a in alphas_file],
                                               [mse_file[a] for a in alphas_file],
                                               in_bounds_col))
                    cv_buffer = io.StringIO()
                    np.savetxt(cv_buffer, cv_data, fmt='%.6e\t%.6e\t%.6e\t%d',
                               header=f'alpha\tMSE_pure\tCV_score_combined\tin_bounds (optimal_alpha={result["best_alpha"]:.2e})')
                else:
                    cv_data = np.column_stack((alphas_file, [mse_file[a] for a in alphas_file], in_bounds_col))
                    cv_buffer = io.StringIO()
                    np.savetxt(cv_buffer, cv_data, fmt='%.6e\t%.6e\t%d',
                               header=f'alpha\tMSE\tin_bounds (optimal_alpha={result["best_alpha"]:.2e})')
                zf.writestr(f"CV_{safe_file_name}.txt", cv_buffer.getvalue())

                # 3. EQE data (original + fit, include both raw and smoothed fits if smoothing enabled)
                lam = result['lam']
                inc_flux = result['inc_flux']
                y = result['y']
                eqe_measured = y / inc_flux

                # Compute EQE fit from raw SCE
                y_fit_raw = result['X'] @ sce_raw
                eqe_fit_raw = y_fit_raw / inc_flux

                if use_smoothing_batch and file_name in sce_profiles_smoothed:
                    # Also compute EQE fit from smoothed SCE
                    y_fit_smooth = result['X'] @ sce_smooth
                    eqe_fit_smooth = y_fit_smooth / inc_flux
                    eqe_data = np.column_stack((lam, eqe_measured, eqe_fit_raw, eqe_fit_smooth))
                    eqe_buffer = io.StringIO()
                    np.savetxt(eqe_buffer, eqe_data, fmt='%.6f\t%.6f\t%.6f\t%.6f',
                               header=f'wavelength(nm)\tEQE_measured\tEQE_fit_raw (alpha={alpha_used:.2e})\tEQE_fit_smoothed ({smoothing_note})')
                else:
                    eqe_data = np.column_stack((lam, eqe_measured, eqe_fit_raw))
                    eqe_buffer = io.StringIO()
                    np.savetxt(eqe_buffer, eqe_data, fmt='%.6f\t%.6f\t%.6f',
                               header=f'wavelength(nm)\tEQE_measured\tEQE_fit (alpha={alpha_used:.2e})')
                zf.writestr(f"EQE_{safe_file_name}.txt", eqe_buffer.getvalue())

        zip_buffer.seek(0)

        safe_name = "".join(c for c in results_name if c.isalnum() or c in ('_', '-')).strip()
        if not safe_name:
            safe_name = "sce_batch_results"

        st.download_button(
            label="Download Batch Results (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"{safe_name}.zip",
            mime="application/zip",
            use_container_width=True
        )

        if any_penalty_active:
            st.caption("ZIP includes: analysis_info.txt, and for each file: SCE profile, CV data (alpha vs pure MSE & combined CV score), EQE data (measured + fit)")
        else:
            st.caption("ZIP includes: analysis_info.txt, and for each file: SCE profile, CV data (alpha vs MSE), EQE data (measured + fit)")

    else:
        # === SINGLE FILE MODE DISPLAY ===
        # Retrieve from session state
        alphas = st.session_state.alphas
        mse = st.session_state.mse
        mse_pure = st.session_state.get('mse_pure', mse)  # Fallback to mse if not available
        penalty_values = st.session_state.get('penalty_values', {a: 0 for a in alphas})
        any_penalty_stored = False
        best_alpha = st.session_state.best_alpha
        X = st.session_state.X
        y = st.session_state.y
        L = st.session_state.L
        lam = st.session_state.lam
        pos = st.session_state.pos
        inc_flux = st.session_state.inc_flux
        gen_filtered = st.session_state.gen_filtered
        eqe_original_wavelengths = st.session_state.eqe_original_wavelengths
        eqe_original_values = st.session_state.eqe_original_values
        X_weighted = st.session_state.get('X_weighted', X)
        y_weighted = st.session_state.get('y_weighted', y)
        sample_weights = st.session_state.get('sample_weights', np.ones(len(lam)))

        # Interactive Alpha Slider
        st.markdown("---")
        st.subheader("Regularization strength")

        # Calculate log10 range for slider
        log_alpha_min = np.log10(alphas.min())
        log_alpha_max = np.log10(alphas.max())
        log_best_alpha = np.log10(best_alpha)

        # Slider for log10(alpha)
        log_current_alpha = st.slider(
            "log₁₀(α) - Adjust to explore different regularization strengths",
            min_value=float(log_alpha_min),
            max_value=float(log_alpha_max),
            value=float(log_best_alpha),
            step=0.01,
            format="%.2f"
        )

        current_alpha = 10**log_current_alpha

        # Display current alpha info - adjust columns based on what's enabled
        n_cols = 3
        if weight_factor_stored is not None:
            n_cols += 1
        if any_penalty_stored:
            n_cols += 1

        info_cols = st.columns(n_cols)
        col_idx = 0

        info_cols[col_idx].metric("Current α", f"{current_alpha:.2e}")
        col_idx += 1
        info_cols[col_idx].metric("Optimal α", f"{best_alpha:.2e}")
        col_idx += 1

        # Find MSE/score for current alpha (interpolate if necessary)
        if current_alpha in mse:
            current_mse = mse[current_alpha]
            current_pure_mse = mse_pure.get(current_alpha, current_mse)
            current_penalty = penalty_values.get(current_alpha, 0)
        else:
            # Interpolate
            log_alphas_array = np.log10(alphas)
            mse_array = np.array([mse[a] for a in alphas])
            current_mse = 10**np.interp(log_current_alpha, log_alphas_array, np.log10(mse_array))
            mse_pure_array = np.array([mse_pure.get(a, mse[a]) for a in alphas])
            current_pure_mse = 10**np.interp(log_current_alpha, log_alphas_array, np.log10(mse_pure_array))
            penalty_array = np.array([penalty_values.get(a, 0) for a in alphas])
            current_penalty = np.interp(log_current_alpha, log_alphas_array, penalty_array)

        if any_penalty_stored:
            # Show pure MSE
            info_cols[col_idx].metric("Pure MSE", f"{current_pure_mse:.2e}")
            col_idx += 1
            # Show CV Score (combined)
            info_cols[col_idx].metric("CV Score", f"{current_mse:.2e}",
                            delta=f"penalty: {current_penalty:.2e}",
                            delta_color="off")
            col_idx += 1
        else:
            info_cols[col_idx].metric("Current MSE", f"{current_mse:.2e}",
                            delta=f"{((current_mse/mse[best_alpha] - 1)*100):.1f}%",
                            delta_color="inverse")
            col_idx += 1

        # Show weighted fitting info if it was used
        if weight_factor_stored is not None:
            info_cols[col_idx].metric("Weight", f"{weight_wl_min_stored}-{weight_wl_max_stored} nm ({weight_factor_stored}×)")

        # Get weighted averaging results if available
        use_weighted_avg_stored = st.session_state.get('use_weighted_avg', False)
        wa_results = st.session_state.get('wa_results', None)

        if use_weighted_avg_stored and wa_results is not None and wa_results.get('sce_avg') is not None:
            # Use weighted averaging results
            sce_optimal = wa_results['sce_avg']
            sce_optimal_std = wa_results['sce_std']
            y_fit_optimal = X @ sce_optimal

            # For current alpha, use the SCE from all_sce at the closest alpha
            alpha_idx = np.argmin(np.abs(alphas - current_alpha))
            sce_current = wa_results['all_sce'][alpha_idx]
            y_fit_current = X @ sce_current
        else:
            sce_optimal_std = None
            # Fit model with current alpha (use weighted data for fitting, unweighted for prediction/display)
            model_current = CustomRidgeDirect(alpha=current_alpha, L=L, constraint=use_clipping, use_bounded=use_bounded_opt)
            model_current.fit(X_weighted, y_weighted)
            y_fit_current = model_current.predict(X)  # Predict on unweighted X for display
            sce_current = model_current.coef_

            # Fit model with optimal alpha
            model_optimal = CustomRidgeDirect(alpha=best_alpha, L=L, constraint=use_clipping, use_bounded=use_bounded_opt)
            model_optimal.fit(X_weighted, y_weighted)
            y_fit_optimal = model_optimal.predict(X)  # Predict on unweighted X for display
            sce_optimal = model_optimal.coef_

        # Plot CV curve
        col1, col2 = st.columns(2)

        with col1:
            # Create interactive CV plot with Plotly
            fig1 = go.Figure()

            if use_weighted_avg_stored and wa_results is not None:
                # Weighted averaging mode - show MSE curve with averaging range
                mse_values = [mse[a] for a in alphas]
                bounds_mask = wa_results.get('bounds_mask', np.ones(len(alphas), dtype=bool))
                final_mask = wa_results.get('final_mask', np.zeros(len(alphas), dtype=bool))

                # Plot out-of-bounds points in gray
                out_of_bounds_alphas = alphas[~bounds_mask]
                out_of_bounds_mse = np.array(mse_values)[~bounds_mask]
                if len(out_of_bounds_alphas) > 0:
                    fig1.add_trace(go.Scatter(
                        x=out_of_bounds_alphas,
                        y=out_of_bounds_mse,
                        mode='markers',
                        name='Out of bounds',
                        marker=dict(color='lightgray', size=4),
                        hovertemplate='Alpha: %{x:.2e}<br>MSE: %{y:.2e}<br>(out of bounds)<extra></extra>'
                    ))

                # Plot in-bounds points in blue
                in_bounds_alphas = alphas[bounds_mask]
                in_bounds_mse = np.array(mse_values)[bounds_mask]
                fig1.add_trace(go.Scatter(
                    x=in_bounds_alphas,
                    y=in_bounds_mse,
                    mode='markers',
                    name='In bounds',
                    marker=dict(color='royalblue', size=5),
                    hovertemplate='Alpha: %{x:.2e}<br>MSE: %{y:.2e}<extra></extra>'
                ))

                # Highlight averaging range
                avg_alphas = alphas[final_mask]
                avg_mse = np.array(mse_values)[final_mask]
                if len(avg_alphas) > 0:
                    fig1.add_trace(go.Scatter(
                        x=avg_alphas,
                        y=avg_mse,
                        mode='markers',
                        name=f'Averaged (n={len(avg_alphas)})',
                        marker=dict(color='green', size=7),
                        hovertemplate='Alpha: %{x:.2e}<br>MSE: %{y:.2e}<br>(included in avg)<extra></extra>'
                    ))

                # Mark CV optimal (min MSE)
                fig1.add_trace(go.Scatter(
                    x=[best_alpha],
                    y=[mse[best_alpha]],
                    mode='markers',
                    name='CV optimal (min MSE)',
                    marker=dict(color='blue', size=12, symbol='star'),
                    hovertemplate='CV optimal α: %{x:.2e}<br>MSE: %{y:.2e}<extra></extra>'
                ))

                # Mark saturation point
                sat_alpha = wa_results.get('saturation_alpha')
                if sat_alpha is not None:
                    sat_idx = np.argmin(np.abs(alphas - sat_alpha))
                    fig1.add_trace(go.Scatter(
                        x=[sat_alpha],
                        y=[mse_values[sat_idx]],
                        mode='markers',
                        name='Saturation',
                        marker=dict(color='purple', size=10, symbol='x'),
                        hovertemplate='Saturation α: %{x:.2e}<br>MSE: %{y:.2e}<extra></extra>'
                    ))

                # Mark averaged MSE
                avg_mse_val = wa_results.get('avg_mse')
                if avg_mse_val is not None:
                    fig1.add_trace(go.Scatter(
                        x=[best_alpha],
                        y=[avg_mse_val],
                        mode='markers',
                        name='Weighted avg MSE',
                        marker=dict(color='red', size=12, symbol='diamond'),
                        hovertemplate='Weighted avg MSE: %{y:.2e}<extra></extra>'
                    ))

                plot_title = 'MSE vs Alpha (Weighted Averaging)'
                yaxis_title = 'MSE'

            else:
                # Standard single curve
                mse_values = [mse[a] for a in alphas]
                screen_oob_stored = st.session_state.get('screen_out_of_bounds', False)
                oob_mask_stored = st.session_state.get('oob_mask', {})

                if screen_oob_stored and oob_mask_stored:
                    # Show out-of-bounds points in gray
                    oob_alphas = [a for a in alphas if oob_mask_stored.get(a, False)]
                    oob_mse = [mse[a] for a in oob_alphas]
                    if oob_alphas:
                        fig1.add_trace(go.Scatter(
                            x=oob_alphas,
                            y=oob_mse,
                            mode='markers',
                            name='Out of bounds',
                            marker=dict(color='lightgray', size=4),
                            hovertemplate='Alpha: %{x:.2e}<br>MSE: %{y:.2e}<br>(out of bounds)<extra></extra>'
                        ))

                    # Show in-bounds curve
                    ib_alphas = [a for a in alphas if not oob_mask_stored.get(a, False)]
                    ib_mse = [mse[a] for a in ib_alphas]
                    if ib_alphas:
                        fig1.add_trace(go.Scatter(
                            x=ib_alphas,
                            y=ib_mse,
                            mode='lines',
                            name='CV Score (in bounds)',
                            line=dict(color='royalblue', width=2),
                            hovertemplate='Alpha: %{x:.2e}<br>CV Score: %{y:.2e}<extra></extra>'
                        ))
                else:
                    fig1.add_trace(go.Scatter(
                        x=alphas,
                        y=mse_values,
                        mode='lines',
                        name='CV Score',
                        line=dict(color='royalblue', width=2),
                        hovertemplate='Alpha: %{x:.2e}<br>CV Score: %{y:.2e}<extra></extra>'
                    ))

                # Optimal point
                fig1.add_trace(go.Scatter(
                    x=[best_alpha],
                    y=[mse[best_alpha]],
                    mode='markers',
                    name='Optimal α',
                    marker=dict(color='red', size=12, symbol='star'),
                    hovertemplate='Optimal α: %{x:.2e}<br>CV Score: %{y:.2e}<extra></extra>'
                ))

                plot_title = 'Cross-Validation: MSE vs Alpha'
                yaxis_title = 'Average CV Score'

            # Current alpha point (only if different from optimal)
            if abs(current_alpha - best_alpha) / best_alpha > 0.01:  # Show if >1% different
                fig1.add_trace(go.Scatter(
                    x=[current_alpha],
                    y=[current_mse],
                    mode='markers',
                    name='Current α',
                    marker=dict(color='orange', size=12, symbol='diamond'),
                    hovertemplate='Current α: %{x:.2e}<br>Score: %{y:.2e}<extra></extra>'
                ))

            fig1.update_layout(
                title=dict(text=plot_title, font=dict(size=16, family='Arial Black')),
                xaxis=dict(title='Alpha', type='log', gridcolor='lightgray'),
                yaxis=dict(title=yaxis_title, type='log', gridcolor='lightgray'),
                template='plotly_white',
                hovermode='closest',
                height=500,
                showlegend=True,
                legend=dict(yanchor='top', y=1, xanchor='left', x=1.02),
                margin=dict(r=150)
            )

            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # Create interactive EQE fit plot with Plotly
            fig2 = go.Figure()

            # Original imported EQE (before interpolation)
            fig2.add_trace(go.Scatter(
                x=eqe_original_wavelengths,
                y=eqe_original_values,
                mode='markers',
                name='Original EQE',
                marker=dict(color='blue', size=4, opacity=0.6),
                hovertemplate='λ: %{x:.1f} nm<br>EQE (original): %{y:.4f}<extra></extra>'
            ))

            # Measured EQE (interpolated to generation wavelengths)
            fig2.add_trace(go.Scatter(
                x=lam,
                y=y/inc_flux,
                mode='lines',
                name='Interpolated EQE',
                line=dict(color='gray', width=2, dash='dash'),
                hovertemplate='λ: %{x:.1f} nm<br>EQE (interp): %{y:.4f}<extra></extra>'
            ))

            # Optimal/weighted avg fit
            if use_weighted_avg_stored:
                # When weighted averaging is used, show weighted avg fit
                fig2.add_trace(go.Scatter(
                    x=lam,
                    y=y_fit_optimal/inc_flux,
                    mode='lines',
                    name='Weighted avg fit',
                    line=dict(color='darkgreen', width=2.5),
                    hovertemplate='λ: %{x:.1f} nm<br>EQE (weighted avg): %{y:.4f}<extra></extra>'
                ))
                # Also show current alpha fit for comparison
                fig2.add_trace(go.Scatter(
                    x=lam,
                    y=y_fit_current/inc_flux,
                    mode='lines',
                    name='Current α fit',
                    line=dict(color='orange', width=2, dash='dot'),
                    hovertemplate='λ: %{x:.1f} nm<br>EQE (current α): %{y:.4f}<extra></extra>'
                ))
            else:
                # Standard mode: show optimal alpha fit if different from current
                if abs(current_alpha - best_alpha) / best_alpha > 0.01:
                    fig2.add_trace(go.Scatter(
                        x=lam,
                        y=y_fit_optimal/inc_flux,
                        mode='lines',
                        name='Optimal α fit',
                        line=dict(color='lightgray', width=2),
                        hovertemplate='λ: %{x:.1f} nm<br>EQE (opt): %{y:.4f}<extra></extra>',
                        opacity=0.5
                    ))

                # Current alpha fit (solid line)
                fig2.add_trace(go.Scatter(
                    x=lam,
                    y=y_fit_current/inc_flux,
                    mode='lines',
                    name='Current α fit',
                    line=dict(color='darkgreen', width=2.5),
                    hovertemplate='λ: %{x:.1f} nm<br>EQE (current): %{y:.4f}<extra></extra>'
                ))

            fig2.update_layout(
                title=dict(text='Measured vs Fitted EQE', font=dict(size=16, family='Arial Black')),
                xaxis=dict(title='Wavelength (nm)', gridcolor='lightgray'),
                yaxis=dict(title='EQE', gridcolor='lightgray'),
                template='plotly_white',
                hovermode='x unified',
                height=500,
                showlegend=True,
                legend=dict(yanchor='top', y=1, xanchor='left', x=1.02),
                margin=dict(r=150)
            )

            st.plotly_chart(fig2, use_container_width=True)

        # Extract SCE with uncertainty
        st.subheader("Extracted spatial collection efficiency (SCE)")

        # Store raw SCE for reference
        sce_current_raw = sce_current.copy()
        sce_optimal_raw = sce_optimal.copy()
        current_mse = np.mean((y - y_fit_current)**2)

        # Smoothing controls next to SCE plot
        smooth_col1, smooth_col2, smooth_col3, smooth_col4 = st.columns([1, 1, 1, 1.5])
        with smooth_col1:
            use_smoothing = st.checkbox("Smooth SCE", value=False, key="single_smooth_checkbox",
                                        help="Apply smoothing filter to remove oscillations")

        # Get max allowed window for Savgol
        max_window = len(sce_current_raw) // 2 * 2 - 1
        # Initialize default values
        smooth_method = "Savitzky-Golay"
        smooth_window = 51
        smooth_poly = 3
        spline_smoothing = 0.01

        if use_smoothing:
            with smooth_col2:
                smooth_method = st.selectbox("Method", ["Savitzky-Golay", "Smoothing Spline"],
                                             key="single_smooth_method",
                                             help="Savgol: local polynomial. Spline: global fit, preserves peak positions better.")

            if smooth_method == "Savitzky-Golay":
                with smooth_col3:
                    smooth_window = st.slider("Window", min_value=5, max_value=min(501, max_window),
                                              value=min(51, max_window), step=2, key="single_smooth_window",
                                              help="Larger = more smoothing")
                    smooth_poly = st.slider("Poly order", min_value=1, max_value=min(smooth_window-1, 15),
                                            value=min(3, smooth_window-1), key="single_smooth_poly",
                                            help="Lower = smoother, higher = preserves more detail")

                # Apply Savitzky-Golay smoothing
                sce_current_smooth = savgol_filter(sce_current_raw, smooth_window, smooth_poly)
                sce_optimal_smooth = savgol_filter(sce_optimal_raw, smooth_window, smooth_poly)

            else:  # Smoothing Spline
                with smooth_col3:
                    spline_smoothing = st.slider("Smoothing", min_value=0.0001, max_value=1.0,
                                                 value=0.001, step=0.0001, format="%.4f",
                                                 key="single_spline_smoothing",
                                                 help="Higher = smoother curve. Preserves peak positions.")

                # Apply smoothing spline
                n_points = len(sce_current_raw)
                s_param = spline_smoothing * n_points
                spline_current = UnivariateSpline(pos, sce_current_raw, s=s_param)
                spline_optimal = UnivariateSpline(pos, sce_optimal_raw, s=s_param)
                sce_current_smooth = spline_current(pos)
                sce_optimal_smooth = spline_optimal(pos)

            if use_clipping:
                sce_current_smooth = np.clip(sce_current_smooth, 0, 1)
                sce_optimal_smooth = np.clip(sce_optimal_smooth, 0, 1)

            # Calculate smoothed EQE fit and MSE
            y_fit_current_smooth = X @ sce_current_smooth
            mse_smooth = np.mean((y - y_fit_current_smooth)**2)
            with smooth_col4:
                st.metric("Smoothed MSE", f"{mse_smooth:.2e}",
                         delta=f"{((mse_smooth/current_mse - 1)*100):+.1f}% vs raw",
                         delta_color="inverse")

        # Option to upload reference SCE profiles for comparison
        with st.expander("Upload reference SCE profiles (e.g., for validation)"):
            ref_sce_files_single = st.file_uploader(
                "Upload SCE files (2 columns: position in nm, SCE value)",
                type=['txt', 'csv', 'dat'],
                accept_multiple_files=True,
                key="ref_sce_single",
                help="Upload one or more reference SCE profiles to overlay on the plot"
            )
            ref_x_shift_single = st.number_input(
                "X-axis shift (nm)",
                value=0.0,
                step=1.0,
                format="%.1f",
                key="ref_x_shift_single",
                help="Shift the position axis of uploaded reference profiles (positive = shift right)"
            )
            ref_sce_profiles_single = {}
            if ref_sce_files_single:
                for ref_file in ref_sce_files_single:
                    try:
                        ref_data = np.loadtxt(ref_file, comments='#')
                        if ref_data.ndim == 1:
                            st.warning(f"Skipped {ref_file.name}: needs 2 columns")
                            continue
                        ref_sce_profiles_single[ref_file.name] = {
                            'pos': ref_data[:, 0] + ref_x_shift_single,
                            'sce': ref_data[:, 1]
                        }
                    except Exception as e:
                        st.warning(f"Could not load {ref_file.name}: {e}")

        # Checkbox for showing generation analysis
        show_gen_analysis = st.checkbox(
            "Show SCE × generation analysis",
            value=False,
            help="Display detailed analysis showing how SCE and generation profiles combine to produce photocurrent"
        )

        # Create columns for side-by-side plots
        if show_gen_analysis:
            col_sce, col_gen = st.columns(2)
        else:
            col_sce = st.container()

        with col_sce:
            # Create interactive SCE profile plot with Plotly
            fig3 = go.Figure()

            # For weighted averaging mode, show the averaged profile with uncertainty band
            if use_weighted_avg_stored and wa_results is not None and sce_optimal_std is not None:
                # Uncertainty band (fill between)
                fig3.add_trace(go.Scatter(
                    x=np.concatenate([pos, pos[::-1]]),
                    y=np.concatenate([sce_optimal_raw + sce_optimal_std, (sce_optimal_raw - sce_optimal_std)[::-1]]),
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Weighted avg ±1σ',
                    hoverinfo='skip'
                ))

                # Averaged profile
                fig3.add_trace(go.Scatter(
                    x=pos,
                    y=sce_optimal_raw,
                    mode='lines',
                    name=f'Weighted avg (n={wa_results.get("n_averaged", "?")})',
                    line=dict(color='red', width=3),
                    hovertemplate='Depth: %{x:.1f} nm<br>SCE (avg): %{y:.4f}<extra></extra>'
                ))

                # CV optimal profile for comparison (min MSE among in-bounds)
                cv_opt_idx = np.argmin(np.where(wa_results['bounds_mask'], wa_results['all_mse'], np.inf))
                sce_cv_opt = wa_results['all_sce'][cv_opt_idx]
                fig3.add_trace(go.Scatter(
                    x=pos,
                    y=sce_cv_opt,
                    mode='lines',
                    name=f'CV optimal α={best_alpha:.1e}',
                    line=dict(color='blue', width=2, dash='dash'),
                    hovertemplate='Depth: %{x:.1f} nm<br>SCE (CV opt): %{y:.4f}<extra></extra>'
                ))

            # Standard mode: Optimal alpha profile (grey, shown when different from current)
            elif abs(current_alpha - best_alpha) / best_alpha > 0.01:
                line_style = dict(color='gray', width=1, dash='dash') if use_smoothing else dict(color='gray', width=2)
                fig3.add_trace(go.Scatter(
                    x=pos,
                    y=sce_optimal_raw,
                    mode='lines',
                    name=f'Optimal α={best_alpha:.1e}' + (' (raw)' if use_smoothing else ''),
                    line=line_style,
                    hovertemplate='Depth: %{x:.1f} nm<br>SCE (optimal): %{y:.4f}<extra></extra>'
                ))

            # Current alpha profile - raw (dashed if smoothing enabled)
            line_style_current = dict(color='darkblue', width=1, dash='dash') if use_smoothing else dict(color='darkblue', width=3)
            fig3.add_trace(go.Scatter(
                x=pos,
                y=sce_current_raw,
                mode='lines',
                name=f'Current α={current_alpha:.1e}' + (' (raw)' if use_smoothing else ''),
                line=line_style_current,
                hovertemplate='Depth: %{x:.1f} nm<br>SCE (current): %{y:.4f}<extra></extra>'
            ))

            # Smoothed profiles (if smoothing enabled)
            if use_smoothing:
                if abs(current_alpha - best_alpha) / best_alpha > 0.01:
                    fig3.add_trace(go.Scatter(
                        x=pos,
                        y=sce_optimal_smooth,
                        mode='lines',
                        name=f'Optimal α={best_alpha:.1e} (smoothed)',
                        line=dict(color='gray', width=2),
                        hovertemplate='Depth: %{x:.1f} nm<br>SCE (optimal, smoothed): %{y:.4f}<extra></extra>'
                    ))
                fig3.add_trace(go.Scatter(
                    x=pos,
                    y=sce_current_smooth,
                    mode='lines',
                    name=f'Current α={current_alpha:.1e} (smoothed)',
                    line=dict(color='darkblue', width=3),
                    hovertemplate='Depth: %{x:.1f} nm<br>SCE (current, smoothed): %{y:.4f}<extra></extra>'
                ))

            # Add reference SCE profiles if uploaded
            ref_colors = ['black', 'dimgray', 'darkslategray', 'slategray']
            for idx, (ref_name, ref_data) in enumerate(ref_sce_profiles_single.items()):
                ref_color = ref_colors[idx % len(ref_colors)]
                fig3.add_trace(go.Scatter(
                    x=ref_data['pos'],
                    y=ref_data['sce'],
                    mode='lines',
                    name=f'Ref: {ref_name}',
                    line=dict(color=ref_color, width=2, dash='dot'),
                    hovertemplate=f'{ref_name}<br>Depth: %{{x:.1f}} nm<br>SCE: %{{y:.4f}}<extra></extra>'
                ))

            plot_title = 'Extracted SCE Profile (Raw vs Smoothed)' if use_smoothing else 'Extracted SCE Profile'
            fig3.update_layout(
                title=dict(text=plot_title, font=dict(size=18, family='Arial Black')),
                xaxis=dict(title='Depth (nm)', gridcolor='lightgray'),
                yaxis=dict(title='Collection Efficiency (-)', range=[0, 1], gridcolor='lightgray'),
                template='plotly_white',
                hovermode='x unified',
                height=600,
                showlegend=True,
                legend=dict(yanchor='top', y=1, xanchor='left', x=1.02),
                margin=dict(r=150)
            )

            st.plotly_chart(fig3, use_container_width=True)

        # SCE × Generation Analysis (only if checkbox is selected)
        if show_gen_analysis:
            with col_gen:
                # Use smoothed SCE for analysis if enabled
                sce_for_analysis = sce_current_smooth if use_smoothing else sce_current_raw

                # Calculate total generation profile (integrate over wavelengths)
                total_generation = trapz_func(gen_filtered, lam, axis=1)

                # Calculate collected generation (SCE × total generation)
                collected_generation_current = sce_for_analysis * total_generation

                # Calculate cumulative Jsc (integrate from left to right)
                cumulative_jsc_current = np.zeros_like(pos)
                for i in range(1, len(pos)):
                    cumulative_jsc_current[i] = trapz_func(collected_generation_current[:i+1], pos[:i+1])

                # Convert to mA/cm^2
                q = 1.602e-19  # Coulombs
                cumulative_jsc_current = cumulative_jsc_current * q * 1e-7 * 1000  # Convert to mA/cm^2

                # Create figure with three y-axes
                fig4 = go.Figure()

                # Add collected generation as filled area (on generation axis)
                fig4.add_trace(go.Scatter(
                    x=pos,
                    y=collected_generation_current / 1e21,
                    mode='lines',
                    name='Collected Generation (SCE×G)',
                    line=dict(width=0),
                    fill='tozeroy',
                    fillcolor='rgba(100, 149, 237, 0.4)',
                    hovertemplate='Depth: %{x:.1f} nm<br>Collected G: %{y:.2f}×10²¹ cm⁻³s⁻¹<extra></extra>',
                    yaxis='y2'
                ))

                # Add total generation line (on generation axis)
                fig4.add_trace(go.Scatter(
                    x=pos,
                    y=total_generation / 1e21,
                    mode='lines',
                    name='Total Generation',
                    line=dict(color='red', width=2, dash='dash'),
                    hovertemplate='Depth: %{x:.1f} nm<br>Total G: %{y:.2f}×10²¹ cm⁻³s⁻¹<extra></extra>',
                    yaxis='y2'
                ))

                # Add SCE line (on SCE axis)
                fig4.add_trace(go.Scatter(
                    x=pos,
                    y=sce_for_analysis,
                    mode='lines',
                    name='SCE' + (' (smoothed)' if use_smoothing else ''),
                    line=dict(color='darkblue', width=2),
                    hovertemplate='Depth: %{x:.1f} nm<br>SCE: %{y:.4f}<extra></extra>',
                    yaxis='y'
                ))

                # Add cumulative Jsc line (on Jsc axis)
                fig4.add_trace(go.Scatter(
                    x=pos,
                    y=cumulative_jsc_current,
                    mode='lines',
                    name='Cumulative Jsc',
                    line=dict(color='gray', width=2, dash='dash'),
                    hovertemplate='Depth: %{x:.1f} nm<br>Cumulative Jsc: %{y:.2f} mA/cm²<extra></extra>',
                    yaxis='y3'
                ))

                # Update layout with three y-axes
                fig4.update_layout(
                    title=dict(text='SCE × Generation Analysis', font=dict(size=18, family='Arial Black')),
                    xaxis=dict(title='Depth (nm)', gridcolor='lightgray'),
                    yaxis=dict(
                        title='Collection Efficiency (-)',
                        title_font=dict(color='darkblue'),
                        tickfont=dict(color='darkblue'),
                        range=[0, 1],
                        gridcolor='lightgray',
                        side='left'
                    ),
                    yaxis2=dict(
                        title='Generation (×10²¹ cm⁻³s⁻¹)',
                        title_font=dict(color='red'),
                        tickfont=dict(color='red'),
                        overlaying='y',
                        side='right',
                        anchor='x',
                        gridcolor='lightgray',
                        showgrid=False
                    ),
                    yaxis3=dict(
                        title='Cumulative Jsc (mA/cm²)',
                        title_font=dict(color='gray'),
                        tickfont=dict(color='gray'),
                        overlaying='y',
                        side='right',
                        anchor='free',
                        position=0.95,
                        showgrid=False
                    ),
                    template='plotly_white',
                    hovermode='x unified',
                    height=600,
                    showlegend=True,
                    legend=dict(yanchor='top', y=1, xanchor='left', x=1.08),
                    margin=dict(r=200)
                )

                st.plotly_chart(fig4, use_container_width=True)

                # Show final Jsc value
                final_jsc = cumulative_jsc_current[-1]
                st.metric("Total Collected Jsc", f"{final_jsc:.2f} mA/cm²")

        # Download results
        st.subheader("Download results")

        # Custom name for results - store in session state so it doesn't change on rerun
        if 'single_results_name' not in st.session_state:
            st.session_state.single_results_name = f"sce_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        results_name = st.text_input(
            "Results name (for ZIP file)",
            key="single_results_name",
            help="Enter a unique name for this analysis. This will be used as the ZIP filename."
        )

        # Get file names for settings info (use first file from list for single mode)
        first_eqe = eqe_file_list[0] if eqe_file_list else None
        eqe_file_name = first_eqe if isinstance(first_eqe, str) else (first_eqe.name if first_eqe else "N/A")
        gen_file_name = gen_file if isinstance(gen_file, str) else (gen_file.name if gen_file else "N/A")
        sun_file_name = sun_file if isinstance(sun_file, str) else (sun_file.name if sun_file else "N/A")

        # Gather all settings info for the analysis_info.txt
        settings_info = {
            'eqe_file': eqe_file_name,
            'gen_file': gen_file_name,
            'sun_source': f"{sun_source} ({sun_file_name})",
            'x1': x1,
            'x2': x2,
            'use_white': use_white,
            'use_clipping': use_clipping,
            'use_bounded': use_bounded_opt,
            'use_first_derivative': use_first_derivative,
            'n_splits': n_splits,
            'use_wl_filter': use_wavelength_filter,
            'wl_min': wl_min,
            'wl_max': wl_max,
            'use_weighted': use_weighted_fitting,
            'weight_wl_min': weight_wl_min,
            'weight_wl_max': weight_wl_max,
            'weight_factor': weight_factor,
            'alpha_min': alpha_min,
            'alpha_max': alpha_max,
            'n_alphas': n_alphas,
            'use_smoothing': use_smoothing,
            'smooth_method': smooth_method if use_smoothing else None,
            'smooth_window': smooth_window if use_smoothing and smooth_method == "Savitzky-Golay" else None,
            'smooth_polyorder': smooth_poly if use_smoothing and smooth_method == "Savitzky-Golay" else None,
            'spline_smoothing': spline_smoothing if use_smoothing and smooth_method == "Smoothing Spline" else None,
            'use_oscillation_penalty': use_oscillation_penalty,
            'screen_out_of_bounds': screen_out_of_bounds,
            'screen_bounds_min': screen_bounds_min,
            'screen_bounds_max': screen_bounds_max,
            'use_weighted_avg': use_weighted_avg,
            'sce_bounds_min': sce_bounds_min,
            'sce_bounds_max': sce_bounds_max,
            'mse_saturation_factor': mse_saturation_factor,
            'mse_deriv_threshold': mse_deriv_threshold,
            'use_cv_for_mse': use_cv_for_mse,
        }

        # Build oob_mask for export
        # In weighted averaging mode, derive from bounds_mask; otherwise use stored oob_mask
        if use_weighted_avg_stored and wa_results is not None:
            wa_bounds = wa_results.get('bounds_mask', np.ones(len(alphas), dtype=bool))
            export_oob_mask = {a: bool(not wa_bounds[i]) for i, a in enumerate(alphas)}
        else:
            export_oob_mask = st.session_state.get('oob_mask', {})

        # Save All Results button (ZIP with all data)
        zip_buffer = create_results_zip(
            alphas=alphas,
            mse=mse,
            lam=lam,
            pos=pos,
            inc_flux=inc_flux,
            y=y,
            y_fit_current=y_fit_current,
            y_fit_optimal=y_fit_optimal,
            sce_current_raw=sce_current_raw,
            current_alpha=current_alpha,
            best_alpha=best_alpha,
            eqe_original_wavelengths=eqe_original_wavelengths,
            eqe_original_values=eqe_original_values,
            gen_filtered=gen_filtered,
            sce_current_smooth=sce_current_smooth if use_smoothing else None,
            settings_info=settings_info,
            mse_pure=mse_pure,
            oob_mask=export_oob_mask
        )

        # Sanitize filename
        safe_name = "".join(c for c in results_name if c.isalnum() or c in ('_', '-')).strip()
        if not safe_name:
            safe_name = "sce_results"

        st.download_button(
            label="Save All Results (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"{safe_name}.zip",
            mime="application/zip",
            use_container_width=True
        )

        st.caption("ZIP includes: CV results, EQE data, SCE profiles, generation analysis, and analysis_info.txt with all settings")

        st.markdown("**Individual downloads:**")

        # Download current alpha SCE (include both raw and smoothed if smoothing enabled)
        if use_smoothing:
            if smooth_method == "Savitzky-Golay":
                smoothing_note = f"Savgol window={smooth_window}, poly={smooth_poly}"
            else:
                smoothing_note = f"Spline s={spline_smoothing:.4f}"
            out_current = np.column_stack((pos, sce_current_raw, sce_current_smooth))
            buffer_current = io.StringIO()
            np.savetxt(buffer_current, out_current, fmt='%.6f\t%.6f\t%.6f',
                       header=f'pos(nm)\tSCE_raw (alpha={current_alpha:.2e})\tSCE_smoothed ({smoothing_note})')
        else:
            out_current = np.column_stack((pos, sce_current_raw))
            buffer_current = io.StringIO()
            np.savetxt(buffer_current, out_current, fmt='%.6f\t%.6f',
                       header=f'pos(nm)\tSCE (alpha={current_alpha:.2e})')

        st.download_button(
            label=f"Download current α SCE data",
            data=buffer_current.getvalue(),
            file_name=f"SCE_alpha_{current_alpha:.2e}.txt",
            mime="text/plain"
        )

elif not (eqe_file_list and gen_file and sun_file):
    st.info("Please upload all three data files to begin analysis")
    
    st.markdown("---")
    st.markdown("### 📖 Instructions")
    st.markdown("""
    1. **Upload data files:**
       - EQE data: wavelength (nm) vs EQE values
       - Generation profile: position (nm) and generation rates
       - Solar spectrum: wavelength (nm) vs flux

    2. **Set parameters:**
       - CTL boundaries (x1, x2) define the active region
       - Choose white light or solar spectrum
       - Adjust cross-validation folds and alpha range

    3. **Run analysis:**
       - Click "Run analysis" to extract SCE profile
       - View cross-validation results and fitted EQE
       - Download extracted SCE data
    """)