import streamlit as st
import numpy as np
# NumPy 2.0+ uses trapezoid, older versions use trapz
trapz_func = getattr(np, 'trapezoid', np.trapz)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.constants import physical_constants as pc
from scipy.interpolate import interp1d
from scipy.optimize import lsq_linear
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
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
                       sce_current, sce_mean, sce_std, current_alpha, best_alpha,
                       eqe_original_wavelengths, eqe_original_values, gen_filtered,
                       settings_info):
    """Create an in-memory ZIP file containing all analysis results.

    Parameters:
    -----------
    settings_info : dict
        Dictionary containing all settings and file information for the analysis_info.txt file
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. CV results (alpha, mse)
        cv_data = np.column_stack((alphas, [mse[a] for a in alphas]))
        cv_buffer = io.StringIO()
        np.savetxt(cv_buffer, cv_data, fmt='%.6e\t%.6e', header='alpha\tMSE')
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

        # 6. SCE current alpha (position, sce)
        sce_current_data = np.column_stack((pos, sce_current))
        sce_current_buffer = io.StringIO()
        np.savetxt(sce_current_buffer, sce_current_data, fmt='%.6f\t%.6f',
                   header=f'pos(nm)\tSCE (alpha={current_alpha:.2e})')
        zf.writestr('sce_current.txt', sce_current_buffer.getvalue())

        # 7. SCE optimal alpha with uncertainty (position, sce_mean, sce_std)
        sce_optimal_data = np.column_stack((pos, sce_mean, sce_std))
        sce_optimal_buffer = io.StringIO()
        np.savetxt(sce_optimal_buffer, sce_optimal_data, fmt='%.6f\t%.6f\t%.6f',
                   header=f'pos(nm)\tSCE_mean\tSCE_std (optimal alpha={best_alpha:.2e})')
        zf.writestr('sce_optimal.txt', sce_optimal_buffer.getvalue())

        # 8. Generation analysis (position, total_gen, collected_gen, cumulative_jsc)
        total_generation = trapz_func(gen_filtered, lam, axis=1)
        collected_generation = sce_current * total_generation

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
            f"Cross-validation folds:        {settings_info.get('n_splits', 'N/A')}",
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
            "",
            "RESULTS",
            "-" * 40,
            f"Optimal alpha:       {best_alpha:.4e}",
            f"Optimal MSE:         {mse[best_alpha]:.4e}",
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
            "cv_results.txt          - Cross-validation alpha vs MSE",
            "eqe_original.txt        - Original EQE data",
            "eqe_interpolated.txt    - EQE interpolated to generation wavelengths",
            "eqe_fit_current.txt     - Fitted EQE at current alpha",
            "eqe_fit_optimal.txt     - Fitted EQE at optimal alpha",
            "sce_current.txt         - SCE profile at current alpha",
            "sce_optimal.txt         - SCE profile at optimal alpha (with std)",
            "generation_analysis.txt - Generation and Jsc analysis",
            "analysis_info.txt       - This file",
            "",
        ]
        )

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
                # Look for the header line with wavelengths (usually starts with "x (nm)")
                for line in lines:
                    if 'x (nm)' in line.lower() or ('nm' in line and not line.strip().startswith('#')):
                        # Parse the header line
                        parts = line.replace('#', '').strip().split()
                        if len(parts) > 1:
                            try:
                                # Extract numeric values from strings like "360 nm", "370 nm", etc.
                                wavelengths = []
                                for part in parts[1:]:
                                    # Try to extract number from string
                                    num_str = ''.join(c for c in part if c.isdigit() or c == '.')
                                    if num_str:
                                        wavelengths.append(float(num_str))
                                wavelengths = np.array(wavelengths)
                                if len(wavelengths) != generation.shape[1]:
                                    wavelengths = None  # Mismatch, ignore
                                break
                            except:
                                pass
        else:
            # Uploaded file - try to read header
            try:
                gen_file.seek(0)
                content = gen_file.read().decode('utf-8')
                gen_file.seek(0)  # Reset for np.loadtxt
                lines = content.split('\n')
                for line in lines:
                    if 'x (nm)' in line.lower() or ('nm' in line and not line.strip().startswith('#')):
                        parts = line.replace('#', '').strip().split()
                        if len(parts) > 1:
                            try:
                                wavelengths = []
                                for part in parts[1:]:
                                    num_str = ''.join(c for c in part if c.isdigit() or c == '.')
                                    if num_str:
                                        wavelengths.append(float(num_str))
                                wavelengths = np.array(wavelengths)
                                if len(wavelengths) != generation.shape[1]:
                                    wavelengths = None
                                break
                            except:
                                pass
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

col_title, col_clear = st.columns([6, 1])
with col_title:
    st.title("EQE fitting and SCE extraction")
    st.markdown("Upload your experimental EQE data and simulated generation profiles to extract the spatial collection efficiency (SCE)")
with col_clear:
    st.markdown("<br>", unsafe_allow_html=True)  # Spacing
    if st.button("🔄 Clear All", use_container_width=True, type="secondary", help="Reset the app and clear all data"):
        st.session_state.clear()
        st.rerun()

# Custom CSS for smaller headers on small screens
st.markdown("""
<style>
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


def read_uploaded_data(eqe_file, gen_file, sun_file, x1, x2):
    """Load and process uploaded files or local file paths."""
    try:
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
                    if 'x (nm)' in line.lower() or ('nm' in line and not line.strip().startswith('#')):
                        parts = line.replace('#', '').strip().split()
                        if len(parts) > 1:
                            try:
                                gen_wavelengths = []
                                for part in parts[1:]:
                                    num_str = ''.join(c for c in part if c.isdigit() or c == '.')
                                    if num_str:
                                        gen_wavelengths.append(float(num_str))
                                gen_wavelengths = np.array(gen_wavelengths)
                                if len(gen_wavelengths) != gen_raw.shape[1]:
                                    gen_wavelengths = None
                                break
                            except:
                                pass
        else:
            # Uploaded file - try to read header
            try:
                gen_file.seek(0)
                content = gen_file.read().decode('utf-8')
                gen_file.seek(0)
                lines = content.split('\n')
                for line in lines:
                    if 'x (nm)' in line.lower() or ('nm' in line and not line.strip().startswith('#')):
                        parts = line.replace('#', '').strip().split()
                        if len(parts) > 1:
                            try:
                                gen_wavelengths = []
                                for part in parts[1:]:
                                    num_str = ''.join(c for c in part if c.isdigit() or c == '.')
                                    if num_str:
                                        gen_wavelengths.append(float(num_str))
                                gen_wavelengths = np.array(gen_wavelengths)
                                if len(gen_wavelengths) != gen_raw.shape[1]:
                                    gen_wavelengths = None
                                break
                            except:
                                pass
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

        eqe = np.loadtxt(eqe_file)
        sun_spec = np.loadtxt(sun_file)
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


# Main window controls
st.markdown("---")
st.header("📁 Data selection")

# Get available files from resources
eqe_files = list_files_in_directory("resources/eqe data")
gen_files = list_files_in_directory("resources/gen data")
default_sun_path = "resources/Sunspectrum.sp"

col_upload1, col_upload2, col_upload3 = st.columns(3)

with col_upload1:
    st.subheader("EQE data")
    eqe_source = st.radio("Source:", ["Local file", "Upload"], key="eqe_source", horizontal=True)
    if eqe_source == "Local file" and eqe_files:
        eqe_selected_list = st.multiselect("Select EQE file(s):", eqe_files, key="eqe_select",
                                            help="Select one or more EQE files for batch processing")
        eqe_file_list = [f"resources/eqe data/{f}" for f in eqe_selected_list] if eqe_selected_list else []
    else:
        uploaded_eqe = st.file_uploader("Upload EQE data", type=['txt', 'csv', 'dat'],
                                         accept_multiple_files=True, key="eqe_upload",
                                         help="Upload one or more EQE files for batch processing")
        eqe_file_list = uploaded_eqe if uploaded_eqe else []

    # Show count of selected files
    if eqe_file_list:
        st.caption(f"{len(eqe_file_list)} EQE file(s) selected")

    # Preview button for EQE (only for single file)
    show_eqe_preview = len(eqe_file_list) == 1 and st.button("Preview EQE", key="preview_eqe")
    # For compatibility, set eqe_file to first file or None
    eqe_file = eqe_file_list[0] if eqe_file_list else None

with col_upload2:
    st.subheader("Generation profile")
    gen_source = st.radio("Source:", ["Local file", "Upload"], key="gen_source", horizontal=True)
    if gen_source == "Local file" and gen_files:
        gen_selected = st.selectbox("Select generation file:", [""] + gen_files, key="gen_select")
        gen_file = f"resources/gen data/{gen_selected}" if gen_selected else None
    else:
        gen_file = st.file_uploader("Upload generation profile", type=['txt', 'csv', 'dat'], key="gen_upload")

        # Show format requirements for uploaded files
        st.info("""
        📄 **File Format Requirements:**
        - Header line must contain: `x (nm)` followed by wavelengths
        - Wavelengths format: `360 nm  370 nm  380 nm  ...`
        - Data columns: `position(nm)` then generation values
        - Tab or space separated
        - Example:
        ```
        # Charge generation [cm^-3*s^-1]
        # Column format:
        # x (nm)  360 nm  370 nm  380 nm  ...
        0.0  4.678e19  5.104e19  5.442e19  ...
        10.0 3.526e19  3.845e19  4.088e19  ...
        ...
        ```
        """)

    # Preview button (shows preview outside columns)
    show_gen_preview = gen_file and st.button("Preview generation", key="preview_gen")

with col_upload3:
    st.subheader("Incident spectrum / Photon flux")
    sun_source = st.radio("Source:", ["Default (Sunspectrum.sp)", "Upload spectrum", "Photon flux file"], key="sun_source", horizontal=True)
    is_photon_flux = (sun_source == "Photon flux file")
    if sun_source == "Default (Sunspectrum.sp)":
        if Path(default_sun_path).exists():
            sun_file = default_sun_path
            st.success("✓ Using default AM1.5G spectrum")
        else:
            st.error("Default spectrum not found!")
            sun_file = None
    elif sun_source == "Upload spectrum":
        sun_file = st.file_uploader("Upload spectrum", type=['txt', 'csv', 'dat', 'sp'], key="sun_upload")
    else:
        sun_file = st.file_uploader("Upload photon flux file", type=['txt', 'csv', 'dat'], key="photon_flux_upload",
                                    help="File with wavelength [nm] and photon flux [photons/s/cm²]")
        if sun_file:
            st.info("📊 Using photon flux directly (no spectrum conversion)")

# Display EQE preview (full width outside columns)
if 'show_eqe_preview' in locals() and show_eqe_preview and eqe_file:
    with st.expander("📊 EQE data preview", expanded=True):
        with st.spinner("Loading preview..."):
            fig, wls, eqe_vals = preview_eqe_data(eqe_file)
            if fig is not None:
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

st.markdown("---")
st.header("⚙️ Settings")

col_param1, col_param2, col_param3 = st.columns(3)

with col_param1:
    st.subheader("Boundaries")
    x1 = st.number_input(
        "CTL boundary x1 (nm)",
        value=10,
        min_value=0,
        help="Starting position of the active layer (nm). Defines the front charge transport layer (CTL) boundary. "
             "SCE will be extracted between x1 and x2."
    )
    x2 = st.number_input(
        "CTL boundary x2 (nm)",
        value=550,
        min_value=0,
        help="End position of the active layer (nm). Defines the back charge transport layer (CTL) boundary. "
             "SCE will be extracted between x1 and x2."
    )

with col_param2:
    st.subheader("Processing options")
    use_white = st.checkbox(
        "Use white light spectrum",
        value=True,
        help="Enable this when your generation profile was calculated with flat/white light illumination. "
             "Sets constant irradiance at all wavelengths (note: photon flux will increase with wavelength). "
             "Disable to use the default AM1.5G or uploaded spectrum."
    )
    use_clipping = st.checkbox(
        "Apply 0-1 clipping to SCE",
        value=True,
        help="Constrains the extracted collection efficiency to physically meaningful values between 0 and 1. "
             "Uses post-processing clipping after solving."
    )
    use_bounded_opt = st.checkbox(
        "Use bounded optimization (0-1)",
        value=False,
        help="Solve the optimization problem directly with 0-1 bounds using constrained least squares. "
             "More rigorous than clipping, but may be slower. When enabled, overrides the clipping option."
    )
    use_first_derivative = st.checkbox(
        "Use 1st derivative regularization",
        value=False,
        help="Use first derivative (gradient penalty) for regularization instead of second derivative (smoothness penalty). "
             "First derivative penalizes slope, second derivative penalizes curvature. "
             "Second derivative is typically smoother but first derivative may preserve sharp features better."
    )
    n_splits = st.slider(
        "Cross-validation folds",
        3, 10, 5,
        help="Number of data splits for K-fold cross-validation when finding optimal α. "
             "More folds = more robust validation but longer computation. "
             "5 folds is standard practice."
    )

    # Batch mode alpha option (only show when multiple EQE files selected)
    if len(eqe_file_list) > 1:
        st.markdown("---")
        st.markdown("**Batch fitting options:**")
        use_same_alpha = st.checkbox(
            "Use same α for all files",
            value=False,
            help="When enabled, finds optimal α from the first EQE file and applies it to all others. "
                 "Faster and useful for comparing samples under identical conditions. "
                 "When disabled, finds optimal α independently for each file."
        )
    else:
        use_same_alpha = False

    use_wavelength_filter = st.checkbox(
        "Restrict wavelength range",
        value=False,
        help="Enable to limit the analysis to a specific wavelength range. "
             "Useful for excluding regions with poor EQE or generation data quality."
    )

    if use_wavelength_filter:
        wl_col1, wl_col2 = st.columns(2)
        with wl_col1:
            wl_min = st.number_input(
                "λ min (nm)",
                value=400,
                min_value=300,
                max_value=1200,
                help="Minimum wavelength to include in analysis"
            )
        with wl_col2:
            wl_max = st.number_input(
                "λ max (nm)",
                value=800,
                min_value=300,
                max_value=1200,
                help="Maximum wavelength to include in analysis"
            )
    else:
        wl_min, wl_max = None, None

    use_weighted_fitting = st.checkbox(
        "Use weighted fitting",
        value=False,
        help="Give higher importance to a specific wavelength region during fitting. "
             "The fit will prioritize matching EQE in the weighted region, potentially "
             "at the expense of fit quality elsewhere."
    )

    if use_weighted_fitting:
        weight_col1, weight_col2, weight_col3 = st.columns(3)
        with weight_col1:
            weight_wl_min = st.number_input(
                "Weight λ min (nm)",
                value=430,
                min_value=300,
                max_value=1200,
                help="Start of wavelength region to prioritize"
            )
        with weight_col2:
            weight_wl_max = st.number_input(
                "Weight λ max (nm)",
                value=445,
                min_value=300,
                max_value=1200,
                help="End of wavelength region to prioritize"
            )
        with weight_col3:
            weight_factor = st.number_input(
                "Weight factor",
                value=10.0,
                min_value=1.0,
                max_value=1000000.0,
                step=1.0,
                help="How much more important the selected region is (e.g., 10 = 10× weight)"
            )
    else:
        weight_wl_min, weight_wl_max, weight_factor = None, None, None

with col_param3:
    st.subheader("Regularization range")
    alpha_min = st.number_input(
        "Alpha range (log10 min)",
        value=0,
        help="Minimum regularization strength (log10 scale). "
             "Smaller α values allow fitting closer to data but may produce noisy SCE profiles. "
             "Start of the search range for optimal α."
    )
    alpha_max = st.number_input(
        "Alpha range (log10 max)",
        value=14,
        help="Maximum regularization strength (log10 scale). "
             "Larger α values produce smoother SCE profiles but may underfit the data. "
             "End of the search range for optimal α."
    )
    n_alphas = st.slider(
        "Number of alpha values",
        100, 2000, 1000,
        help="Number of α values to test via cross-validation between min and max. "
             "More values = finer search but longer computation time. "
             "Typically 1000 provides good balance."
    )

st.markdown("---")
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)
with col_btn2:
    if st.button("Open Manual Fitting", type="secondary", use_container_width=True):
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

            # Manual fitting settings
            col_manual1, col_manual2, col_manual3 = st.columns([1, 1, 1])

            with col_manual1:
                n_segments_manual = st.slider(
                    "Number of SCE segments",
                    min_value=3,
                    max_value=20,
                    value=10,
                    key="n_segments_manual",
                    help="Divide the absorber into segments, each with adjustable SCE"
                )

            with col_manual2:
                smoothing_factor_manual = st.slider(
                    "Smoothing factor",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.0,
                    step=0.1,
                    key="smoothing_manual",
                    help="0 = linear interpolation, higher = smoother curves"
                )

            with col_manual3:
                if st.button("Close Manual Fitting", key="close_manual"):
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
                n_segments_default=n_segments_manual,
                smoothing_factor_external=smoothing_factor_manual,
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
    shared_alpha = None  # Will be set if use_same_alpha is True
    alphas = np.logspace(alpha_min, alpha_max, n_alphas)
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

            # Cross-validation for optimal alpha
            if use_same_alpha and shared_alpha is not None:
                # Use the shared alpha from first file
                best_alpha = shared_alpha
                mse = {shared_alpha: 0}  # Placeholder, will compute actual MSE
                # Compute MSE for this file at the shared alpha
                errs = []
                for tr, val in kf.split(X_weighted):
                    m = CustomRidgeDirect(alpha=shared_alpha, L=L, constraint=use_clipping, use_bounded=use_bounded_opt)
                    m.fit(X_weighted[tr], y_weighted[tr])
                    y_val_pred = m.predict(X_weighted[val])
                    errs.append(mean_squared_error(y_weighted[val], y_val_pred))
                mse[shared_alpha] = np.mean(errs)
            else:
                # Find optimal alpha for this file
                if is_batch:
                    progress_text = st.empty()
                    progress_text.text(f"Finding optimal α for {file_name}...")
                progress_bar = st.progress(0)
                mse = {}

                for idx, a in enumerate(alphas):
                    errs = []
                    for tr, val in kf.split(X_weighted):
                        m = CustomRidgeDirect(alpha=a, L=L, constraint=use_clipping, use_bounded=use_bounded_opt)
                        m.fit(X_weighted[tr], y_weighted[tr])
                        y_val_pred = m.predict(X_weighted[val])
                        errs.append(mean_squared_error(y_weighted[val], y_val_pred))
                    mse[a] = np.mean(errs)
                    progress_bar.progress((idx + 1) / len(alphas))

                best_alpha = min(mse, key=mse.get)
                progress_bar.empty()
                if is_batch and 'progress_text' in dir():
                    progress_text.empty()

                # Set shared alpha from first file if using same alpha mode
                if use_same_alpha and shared_alpha is None:
                    shared_alpha = best_alpha
                    st.info(f"Using α = {shared_alpha:.2e} from {file_name} for all files")

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
                'best_alpha': best_alpha,
                'alphas': alphas,
                'eqe_original_wavelengths': eqe_original_wavelengths,
                'eqe_original_values': eqe_original_values,
            }

            if not is_batch:
                weight_str = f" (weighted: {weight_wl_min}-{weight_wl_max} nm, {weight_factor}×)" if weight_factor is not None else ""
                st.success(f"Optimal alpha: **{best_alpha:.2e}** (MSE: {mse[best_alpha]:.2e}){weight_str}")

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
    # For backward compatibility with single file
    if not is_batch and batch_results:
        first_result = list(batch_results.values())[0]
        st.session_state.alphas = first_result['alphas']
        st.session_state.mse = first_result['mse']
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

        # Summary table
        summary_data = []
        file_names = list(batch_results.keys())

        # Compute SCE for each file
        sce_profiles = {}
        for file_name, result in batch_results.items():
            model = CustomRidgeDirect(
                alpha=result['best_alpha'],
                L=result['L'],
                constraint=use_clipping,
                use_bounded=use_bounded_opt
            )
            model.fit(result['X_weighted'], result['y_weighted'])
            sce_profiles[file_name] = model.coef_

            # Calculate Jsc
            total_gen = trapz_func(result['gen_filtered'], result['lam'], axis=1)
            collected_gen = model.coef_ * total_gen
            jsc = trapz_func(collected_gen, result['pos']) * 1.602e-19 * 1e-7 * 1000  # mA/cm²

            summary_data.append({
                'File': file_name,
                'Optimal α': f"{result['best_alpha']:.2e}",
                'MSE': f"{result['mse'][result['best_alpha']]:.2e}",
                'Jsc (mA/cm²)': f"{jsc:.2f}"
            })

        # Display summary table
        st.dataframe(summary_data, use_container_width=True)

        # CV comparison plot
        st.subheader("Cross-Validation Comparison")

        fig_cv = go.Figure()
        colors = ['darkblue', 'darkgreen', 'darkred', 'purple', 'orange', 'brown', 'pink', 'gray']

        for idx, (file_name, result) in enumerate(batch_results.items()):
            color = colors[idx % len(colors)]
            alphas = result['alphas']
            mse = result['mse']
            best_alpha = result['best_alpha']

            # MSE curve
            mse_values = [mse[a] for a in alphas if a in mse]
            alphas_plot = [a for a in alphas if a in mse]

            fig_cv.add_trace(go.Scatter(
                x=alphas_plot,
                y=mse_values,
                mode='lines',
                name=file_name,
                line=dict(color=color, width=2),
                hovertemplate=f'{file_name}<br>α: %{{x:.2e}}<br>MSE: %{{y:.2e}}<extra></extra>'
            ))

            # Optimal point marker
            fig_cv.add_trace(go.Scatter(
                x=[best_alpha],
                y=[mse[best_alpha]],
                mode='markers',
                name=f'{file_name} optimal',
                marker=dict(color=color, size=10, symbol='star'),
                showlegend=False,
                hovertemplate=f'{file_name} optimal<br>α: %{{x:.2e}}<br>MSE: %{{y:.2e}}<extra></extra>'
            ))

        fig_cv.update_layout(
            title=dict(text='Cross-Validation: MSE vs Alpha', font=dict(size=16, family='Arial Black')),
            xaxis=dict(title='Alpha', type='log', gridcolor='lightgray'),
            yaxis=dict(title='Average MSE', type='log', gridcolor='lightgray'),
            template='plotly_white',
            hovermode='closest',
            height=500,
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )

        st.plotly_chart(fig_cv, use_container_width=True)

        # Comparison plot - SCE profiles overlaid
        st.subheader("SCE Profile Comparison")

        fig_compare = go.Figure()

        for idx, (file_name, sce) in enumerate(sce_profiles.items()):
            pos = batch_results[file_name]['pos']
            color = colors[idx % len(colors)]
            fig_compare.add_trace(go.Scatter(
                x=pos,
                y=sce,
                mode='lines',
                name=file_name,
                line=dict(color=color, width=2),
                hovertemplate=f'{file_name}<br>Depth: %{{x:.1f}} nm<br>SCE: %{{y:.4f}}<extra></extra>'
            ))

        fig_compare.update_layout(
            title=dict(text='SCE Profile Comparison', font=dict(size=18, family='Arial Black')),
            xaxis=dict(title='Depth (nm)', gridcolor='lightgray'),
            yaxis=dict(title='Collection Efficiency (-)', range=[0, 1], gridcolor='lightgray'),
            template='plotly_white',
            hovermode='x unified',
            height=600,
            showlegend=True,
            legend=dict(x=0.02, y=0.02, xanchor='left', yanchor='bottom')
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

            # Compute fit
            model = CustomRidgeDirect(
                alpha=result['best_alpha'],
                L=result['L'],
                constraint=use_clipping,
                use_bounded=use_bounded_opt
            )
            model.fit(result['X_weighted'], result['y_weighted'])
            y_fit = model.predict(result['X'])

            # Measured EQE
            fig_eqe_compare.add_trace(go.Scatter(
                x=lam,
                y=y/inc_flux,
                mode='lines',
                name=f'{file_name} (measured)',
                line=dict(color=color, width=1, dash='dash'),
                hovertemplate=f'{file_name}<br>λ: %{{x:.1f}} nm<br>EQE: %{{y:.4f}}<extra></extra>'
            ))

            # Fitted EQE
            fig_eqe_compare.add_trace(go.Scatter(
                x=lam,
                y=y_fit/inc_flux,
                mode='lines',
                name=f'{file_name} (fit)',
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
            legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top')
        )

        st.plotly_chart(fig_eqe_compare, use_container_width=True)

        # Download batch results
        st.subheader("Download Batch Results")

        default_name = f"sce_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results_name = st.text_input(
            "Results name (for ZIP file)",
            value=default_name,
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
                f"Used same α for all: {st.session_state.get('use_same_alpha', False)}",
                "",
                "Settings:",
                f"  - Boundaries: x1={x1} nm, x2={x2} nm",
                f"  - White light: {use_white}",
                f"  - Clipping: {use_clipping}",
                f"  - Bounded optimization: {use_bounded_opt}",
                f"  - CV folds: {n_splits}",
                "",
                "Results Summary:",
                "-" * 50,
            ]
            for item in summary_data:
                info_lines.append(f"{item['File']}: α={item['Optimal α']}, MSE={item['MSE']}, Jsc={item['Jsc (mA/cm²)']} mA/cm²")

            zf.writestr("analysis_info.txt", "\n".join(info_lines))

            # Individual SCE files
            for file_name, sce in sce_profiles.items():
                pos = batch_results[file_name]['pos']
                best_alpha = batch_results[file_name]['best_alpha']
                out = np.column_stack((pos, sce))
                buffer = io.StringIO()
                np.savetxt(buffer, out, fmt='%.6f\t%.6f',
                           header=f'pos(nm)\tSCE (alpha={best_alpha:.2e})')
                safe_file_name = "".join(c for c in file_name if c.isalnum() or c in ('_', '-', '.')).strip()
                zf.writestr(f"SCE_{safe_file_name}", buffer.getvalue())

        zip_buffer.seek(0)

        safe_name = "".join(c for c in results_name if c.isalnum() or c in ('_', '-')).strip()
        if not safe_name:
            safe_name = default_name

        st.download_button(
            label="Download Batch Results (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"{safe_name}.zip",
            mime="application/zip",
            use_container_width=True
        )

        st.caption("ZIP includes: analysis_info.txt with summary and individual SCE profiles for each file")

    else:
        # === SINGLE FILE MODE DISPLAY ===
        # Retrieve from session state
        alphas = st.session_state.alphas
        mse = st.session_state.mse
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

        # Display current alpha info
        if weight_factor_stored is not None:
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        else:
            col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Current α", f"{current_alpha:.2e}")
        col_info2.metric("Optimal α", f"{best_alpha:.2e}")

        # Find MSE for current alpha (interpolate if necessary)
        if current_alpha in mse:
            current_mse = mse[current_alpha]
        else:
            # Interpolate MSE
            log_alphas_array = np.log10(alphas)
            mse_array = np.array([mse[a] for a in alphas])
            current_mse = 10**np.interp(log_current_alpha, log_alphas_array, np.log10(mse_array))

        col_info3.metric("Current MSE", f"{current_mse:.2e}",
                        delta=f"{((current_mse/mse[best_alpha] - 1)*100):.1f}%",
                        delta_color="inverse")

        # Show weighted fitting info if it was used
        if weight_factor_stored is not None:
            col_info4.metric("Weight", f"{weight_wl_min_stored}-{weight_wl_max_stored} nm ({weight_factor_stored}×)")

        # Fit model with current alpha (use weighted data for fitting, unweighted for prediction/display)
        model_current = CustomRidgeDirect(alpha=current_alpha, L=L, constraint=use_clipping, use_bounded=use_bounded_opt)
        model_current.fit(X_weighted, y_weighted)
        y_fit_current = model_current.predict(X)  # Predict on unweighted X for display
        sce_current = model_current.coef_

        # Fit model with optimal alpha
        model_optimal = CustomRidgeDirect(alpha=best_alpha, L=L, constraint=use_clipping, use_bounded=use_bounded_opt)
        model_optimal.fit(X_weighted, y_weighted)
        y_fit_optimal = model_optimal.predict(X)  # Predict on unweighted X for display

        # Extract SCE with uncertainty for optimal alpha
        coefs = []
        for tr, val in kf.split(X_weighted):
            m2 = CustomRidgeDirect(alpha=best_alpha, L=L, constraint=use_clipping, use_bounded=use_bounded_opt)
            m2.fit(X_weighted[tr], y_weighted[tr])
            coefs.append(m2.coef_)
        coefs = np.vstack(coefs)
        sce_mean, sce_std = coefs.mean(0), coefs.std(0)

        # Plot CV curve
        col1, col2 = st.columns(2)

        with col1:
            # Create interactive CV plot with Plotly
            mse_values = [mse[a] for a in alphas]

            fig1 = go.Figure()

            # Main curve
            fig1.add_trace(go.Scatter(
                x=alphas,
                y=mse_values,
                mode='lines',
                name='MSE',
                line=dict(color='royalblue', width=2),
                hovertemplate='Alpha: %{x:.2e}<br>MSE: %{y:.2e}<extra></extra>'
            ))

            # Optimal point
            fig1.add_trace(go.Scatter(
                x=[best_alpha],
                y=[mse[best_alpha]],
                mode='markers',
                name='Optimal α',
                marker=dict(color='red', size=12, symbol='star'),
                hovertemplate='Optimal α: %{x:.2e}<br>MSE: %{y:.2e}<extra></extra>'
            ))

            # Current alpha point (only if different from optimal)
            if abs(current_alpha - best_alpha) / best_alpha > 0.01:  # Show if >1% different
                fig1.add_trace(go.Scatter(
                    x=[current_alpha],
                    y=[current_mse],
                    mode='markers',
                    name='Current α',
                    marker=dict(color='orange', size=12, symbol='diamond'),
                    hovertemplate='Current α: %{x:.2e}<br>MSE: %{y:.2e}<extra></extra>'
                ))

            fig1.update_layout(
                title=dict(text='Cross-Validation: MSE vs Alpha', font=dict(size=16, family='Arial Black')),
                xaxis=dict(title='Alpha', type='log', gridcolor='lightgray'),
                yaxis=dict(title='Average MSE', type='log', gridcolor='lightgray'),
                template='plotly_white',
                hovermode='closest',
                height=500,
                showlegend=True,
                legend=dict(x=0.02, y=0.98)
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

            # Optimal fit (light grey background)
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
                legend=dict(x=0.02, y=0.02, xanchor='left', yanchor='bottom')
            )

            st.plotly_chart(fig2, use_container_width=True)

        # Extract SCE with uncertainty
        st.subheader("Extracted spatial collection efficiency (SCE)")

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

            # Show optimal alpha profile in light grey if current is different
            if abs(current_alpha - best_alpha) / best_alpha > 0.01:
                # Optimal upper bound
                fig3.add_trace(go.Scatter(
                    x=pos,
                    y=sce_mean + sce_std,
                    mode='lines',
                    name='Optimal ± σ',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip',
                    opacity=0.3
                ))

                # Optimal mean
                fig3.add_trace(go.Scatter(
                    x=pos,
                    y=sce_mean,
                    mode='lines',
                    name='Optimal α SCE',
                    line=dict(color='lightgray', width=2),
                    fill='tonexty',
                    fillcolor='rgba(211, 211, 211, 0.2)',
                    hovertemplate='Depth: %{x:.1f} nm<br>SCE (opt): %{y:.4f}<extra></extra>',
                    opacity=0.5
                ))

                # Optimal lower bound
                fig3.add_trace(go.Scatter(
                    x=pos,
                    y=sce_mean - sce_std,
                    mode='lines',
                    name='Optimal lower',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(211, 211, 211, 0.2)',
                    showlegend=False,
                    hoverinfo='skip',
                    opacity=0.3
                ))

            # Current alpha profile (solid line)
            fig3.add_trace(go.Scatter(
                x=pos,
                y=sce_current,
                mode='lines',
                name='Current α SCE',
                line=dict(color='darkblue', width=3),
                hovertemplate='Depth: %{x:.1f} nm<br>SCE (current): %{y:.4f}<extra></extra>'
            ))

            fig3.update_layout(
                title=dict(text='Extracted SCE Profile', font=dict(size=18, family='Arial Black')),
                xaxis=dict(title='Depth (nm)', gridcolor='lightgray'),
                yaxis=dict(title='Collection Efficiency (-)', range=[0, 1], gridcolor='lightgray'),
                template='plotly_white',
                hovermode='x unified',
                height=600,
                showlegend=True,
                legend=dict(x=0.02, y=0.02, xanchor='left', yanchor='bottom')
            )

            st.plotly_chart(fig3, use_container_width=True)

        # SCE × Generation Analysis (only if checkbox is selected)
        if show_gen_analysis:
            with col_gen:
                # Calculate total generation profile (integrate over wavelengths)
                total_generation = trapz_func(gen_filtered, lam, axis=1)

                # Calculate collected generation (SCE × total generation)
                collected_generation_current = sce_current * total_generation

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
                    y=sce_current,
                    mode='lines',
                    name='SCE',
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
                    legend=dict(x=0.85, y=0.5, xanchor='right', yanchor='middle')
                )

                st.plotly_chart(fig4, use_container_width=True)

                # Show final Jsc value
                final_jsc = cumulative_jsc_current[-1]
                st.metric("Total Collected Jsc", f"{final_jsc:.2f} mA/cm²")

        # Download results
        st.subheader("Download results")

        # Custom name for results
        default_name = f"sce_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results_name = st.text_input(
            "Results name (for ZIP file)",
            value=default_name,
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
        }

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
            sce_current=sce_current,
            sce_mean=sce_mean,
            sce_std=sce_std,
            current_alpha=current_alpha,
            best_alpha=best_alpha,
            eqe_original_wavelengths=eqe_original_wavelengths,
            eqe_original_values=eqe_original_values,
            gen_filtered=gen_filtered,
            settings_info=settings_info
        )

        # Sanitize filename
        safe_name = "".join(c for c in results_name if c.isalnum() or c in ('_', '-')).strip()
        if not safe_name:
            safe_name = default_name

        st.download_button(
            label="Save All Results (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"{safe_name}.zip",
            mime="application/zip",
            use_container_width=True
        )

        st.caption("ZIP includes: CV results, EQE data, SCE profiles, generation analysis, and analysis_info.txt with all settings")

        st.markdown("**Individual downloads:**")
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            # Download current alpha SCE
            out_current = np.column_stack((pos, sce_current))
            buffer_current = io.StringIO()
            np.savetxt(buffer_current, out_current, fmt='%.6f\t%.6f',
                       header=f'pos(nm)\tSCE (alpha={current_alpha:.2e})')

            st.download_button(
                label=f"Download current α SCE data",
                data=buffer_current.getvalue(),
                file_name=f"SCE_alpha_{current_alpha:.2e}.txt",
                mime="text/plain"
            )

        with col_dl2:
            # Download optimal alpha SCE with uncertainty
            out = np.column_stack((pos, sce_mean, sce_std))
            buffer = io.StringIO()
            np.savetxt(buffer, out, fmt='%.6f\t%.6f\t%.6f',
                       header=f'pos(nm)\tSCE_mean\tSCE_std (optimal alpha={best_alpha:.2e})')

            st.download_button(
                label="Download optimal α SCE data",
                data=buffer.getvalue(),
                file_name="extracted_SCE_optimal.txt",
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
       - Download extracted SCE data with uncertainties
    """)