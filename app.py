import streamlit as st
import numpy as np
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
from pathlib import Path

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
            total_generation = np.trapezoid(generation, wavelengths, axis=1)
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


def prepare_input(pos, gen, eqe, sun_spec, use_white_light, gen_wavelengths=None, wl_min=None, wl_max=None, use_first_derivative=False):
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
        Solar spectrum with columns [wavelength, flux]
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
    if use_white_light:
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

    # Interpolate solar spectrum onto the target wavelengths
    interp_flux = interp1d(sun_spec[:, 0], sun_spec[:, 1], bounds_error=False, fill_value=0)
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
        eqe_selected = st.selectbox("Select EQE file:", [""] + eqe_files, key="eqe_select")
        eqe_file = f"resources/eqe data/{eqe_selected}" if eqe_selected else None
    else:
        eqe_file = st.file_uploader("Upload EQE data", type=['txt', 'csv', 'dat'], key="eqe_upload")

    # Preview button for EQE
    show_eqe_preview = eqe_file and st.button("Preview EQE", key="preview_eqe")

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
    st.subheader("Incident spectrum")
    sun_source = st.radio("Source:", ["Default (Sunspectrum.sp)", "Upload"], key="sun_source", horizontal=True)
    if sun_source == "Default (Sunspectrum.sp)":
        if Path(default_sun_path).exists():
            sun_file = default_sun_path
            st.success("✓ Using default AM1.5G spectrum")
        else:
            st.error("Default spectrum not found!")
            sun_file = None
    else:
        sun_file = st.file_uploader("Upload spectrum", type=['txt', 'csv', 'dat', 'sp'], key="sun_upload")

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

    use_cv_wavelength_range = st.checkbox(
        "Use custom CV wavelength range",
        value=False,
        help="Calculate cross-validation MSE only within a specific wavelength region. "
             "The fit still uses ALL wavelengths, but optimal α is determined by fit quality "
             "in the selected region. Useful when you trust some spectral regions more than others."
    )

    if use_cv_wavelength_range:
        cv_wl_col1, cv_wl_col2 = st.columns(2)
        with cv_wl_col1:
            cv_wl_min = st.number_input(
                "CV λ min (nm)",
                value=400,
                min_value=300,
                max_value=1200,
                key="cv_wl_min",
                help="Minimum wavelength for CV MSE calculation"
            )
        with cv_wl_col2:
            cv_wl_max = st.number_input(
                "CV λ max (nm)",
                value=700,
                min_value=300,
                max_value=1200,
                key="cv_wl_max",
                help="Maximum wavelength for CV MSE calculation"
            )
    else:
        cv_wl_min, cv_wl_max = None, None

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
                smoothing_factor_external=smoothing_factor_manual
            )
        else:
            st.error("Manual SCE fitting module not available. Please ensure manual_sce_fitting.py is in the same directory as app.py")

# Separator before automatic analysis
if st.session_state.get('manual_fitting_open', False):
    st.markdown("---")

# Main content - Automatic Analysis
if run_analysis and eqe_file and gen_file and sun_file:
    with st.spinner("Processing data..."):
        # Load data
        pos, gen, eqe, sun_spec, gen_wavelengths = read_uploaded_data(eqe_file, gen_file, sun_file, x1, x2)

        if pos is None:
            st.stop()

        st.success(f"✅ Data loaded: {len(pos)} depth points, {gen.shape[1]} generation wavelengths, {eqe.shape[0]} EQE wavelengths")

        # Store original EQE data before interpolation
        eqe_original_wavelengths = eqe[:, 0].copy()
        eqe_original_values = eqe[:, 1].copy()

        # Prepare input
        X, y, L, (lam, pos, inc_flux, gen_filtered) = prepare_input(pos, gen, eqe, sun_spec, use_white, gen_wavelengths, wl_min, wl_max, use_first_derivative)

        # Cross-validation for optimal alpha
        st.subheader("Cross-validation: finding optimal regularization")

        # Create CV wavelength mask if custom range is specified
        cv_wl_mask = None
        if cv_wl_min is not None and cv_wl_max is not None:
            cv_wl_mask = (lam >= cv_wl_min) & (lam <= cv_wl_max)
            n_cv_wavelengths = np.sum(cv_wl_mask)
            if n_cv_wavelengths == 0:
                st.error(f"No wavelengths found in CV range [{cv_wl_min}, {cv_wl_max}] nm. "
                        f"Available range: [{lam.min():.0f}, {lam.max():.0f}] nm")
                st.stop()
            st.info(f"📊 CV MSE will be calculated on {n_cv_wavelengths} wavelengths "
                   f"in range [{cv_wl_min}, {cv_wl_max}] nm (full fit uses all {len(lam)} wavelengths)")

        progress_bar = st.progress(0)
        alphas = np.logspace(alpha_min, alpha_max, n_alphas)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        mse = {}

        for idx, a in enumerate(alphas):
            errs = []
            for tr, val in kf.split(X):
                m = CustomRidgeDirect(alpha=a, L=L, constraint=use_clipping, use_bounded=use_bounded_opt)
                m.fit(X[tr], y[tr])

                # Calculate MSE on validation set
                y_val_pred = m.predict(X[val])
                y_val_true = y[val]

                # If CV wavelength range is specified, only compute MSE on those wavelengths
                if cv_wl_mask is not None:
                    # val contains indices into X (wavelengths)
                    # We need to find which validation indices fall within the CV wavelength range
                    val_in_cv_range = cv_wl_mask[val]
                    if np.sum(val_in_cv_range) > 0:
                        errs.append(mean_squared_error(y_val_true[val_in_cv_range], y_val_pred[val_in_cv_range]))
                    # If no validation points in CV range for this fold, skip it
                else:
                    errs.append(mean_squared_error(y_val_true, y_val_pred))

            if errs:  # Only record if we got any errors
                mse[a] = np.mean(errs)
            progress_bar.progress((idx + 1) / len(alphas))

        best_alpha = min(mse, key=mse.get)
        cv_range_str = f" (CV range: {cv_wl_min}-{cv_wl_max} nm)" if cv_wl_mask is not None else ""
        st.success(f"Optimal alpha: **{best_alpha:.2e}** (MSE: {mse[best_alpha]:.2e}){cv_range_str}")

        # Store results in session state for slider updates
        st.session_state.analysis_complete = True
        st.session_state.alphas = alphas
        st.session_state.mse = mse
        st.session_state.best_alpha = best_alpha
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.L = L
        st.session_state.lam = lam
        st.session_state.pos = pos
        st.session_state.inc_flux = inc_flux
        st.session_state.gen_filtered = gen_filtered
        st.session_state.kf = kf
        st.session_state.use_clipping = use_clipping
        st.session_state.use_bounded_opt = use_bounded_opt
        st.session_state.eqe_original_wavelengths = eqe_original_wavelengths
        st.session_state.eqe_original_values = eqe_original_values
        # Store raw data for manual fitting
        st.session_state.pos_raw = pos
        st.session_state.gen_raw = gen
        st.session_state.eqe_raw = eqe
        st.session_state.sun_spec = sun_spec
        st.session_state.gen_wavelengths = gen_wavelengths
        st.session_state.use_white = use_white
        st.session_state.wl_min = wl_min
        st.session_state.wl_max = wl_max
        st.session_state.use_first_derivative = use_first_derivative
        st.session_state.cv_wl_min_used = cv_wl_min
        st.session_state.cv_wl_max_used = cv_wl_max
        st.session_state.cv_wl_mask = cv_wl_mask

# Display results if analysis has been run
if 'analysis_complete' in st.session_state and st.session_state.analysis_complete:
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
    kf = st.session_state.kf
    use_clipping = st.session_state.get('use_clipping', True)
    use_bounded_opt = st.session_state.get('use_bounded_opt', False)
    eqe_original_wavelengths = st.session_state.eqe_original_wavelengths
    eqe_original_values = st.session_state.eqe_original_values
    cv_wl_min_stored = st.session_state.get('cv_wl_min_used', None)
    cv_wl_max_stored = st.session_state.get('cv_wl_max_used', None)
    cv_wl_mask = st.session_state.get('cv_wl_mask', None)

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
    if cv_wl_min_stored is not None and cv_wl_max_stored is not None:
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

    # Show CV wavelength range if it was used
    if cv_wl_min_stored is not None and cv_wl_max_stored is not None:
        col_info4.metric("CV λ range", f"{cv_wl_min_stored}-{cv_wl_max_stored} nm")
    
    # Fit model with current alpha
    model_current = CustomRidgeDirect(alpha=current_alpha, L=L, constraint=use_clipping, use_bounded=use_bounded_opt)
    model_current.fit(X, y)
    y_fit_current = model_current.predict(X)
    sce_current = model_current.coef_

    # Fit model with optimal alpha
    model_optimal = CustomRidgeDirect(alpha=best_alpha, L=L, constraint=use_clipping, use_bounded=use_bounded_opt)
    model_optimal.fit(X, y)
    y_fit_optimal = model_optimal.predict(X)
    
    # Extract SCE with uncertainty for optimal alpha
    coefs = []
    for tr, val in kf.split(X):
        m2 = CustomRidgeDirect(alpha=best_alpha, L=L, constraint=use_clipping, use_bounded=use_bounded_opt)
        m2.fit(X[tr], y[tr])
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
            total_generation = np.trapezoid(gen_filtered, lam, axis=1)

            # Calculate collected generation (SCE × total generation)
            collected_generation_current = sce_current * total_generation

            # Calculate cumulative Jsc (integrate from left to right)
            cumulative_jsc_current = np.zeros_like(pos)
            for i in range(1, len(pos)):
                cumulative_jsc_current[i] = np.trapezoid(collected_generation_current[:i+1], pos[:i+1])

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
    
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        # Download current alpha SCE
        out_current = np.column_stack((pos, sce_current))
        buffer_current = io.StringIO()
        np.savetxt(buffer_current, out_current, fmt='%.6f\t%.6f',
                   header=f'pos(nm)\tSCE (alpha={current_alpha:.2e})')
        
        st.download_button(
            label=f"📥 Download current α SCE data",
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
            label="📥 Download optimal α SCE data",
            data=buffer.getvalue(),
            file_name="extracted_SCE_optimal.txt",
            mime="text/plain"
        )

elif not (eqe_file and gen_file and sun_file):
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