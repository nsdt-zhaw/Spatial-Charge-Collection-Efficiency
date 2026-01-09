"""
Manual SCE Fitting Module
Allows interactive exploration of SCE profiles and their effect on EQE
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.constants import physical_constants as pc
from scipy.interpolate import interp1d


def create_sce_from_segments(pos, segment_values, smoothing_factor=0.0):
    """
    Create a full SCE profile from segment values using linear interpolation with optional smoothing.

    Parameters:
    -----------
    pos : array
        Position array (depth in nm)
    segment_values : array
        SCE values for each segment
    smoothing_factor : float
        Smoothing factor (0 = linear interpolation, higher = more smoothing)

    Returns:
    --------
    sce : array
        Full SCE profile interpolated from segments
    segment_centers : array
        Center positions of each segment
    """
    from scipy.interpolate import UnivariateSpline

    n_segments = len(segment_values)

    # Create segment boundaries and centers
    segment_edges = np.linspace(pos.min(), pos.max(), n_segments + 1)
    segment_centers = (segment_edges[:-1] + segment_edges[1:]) / 2

    if smoothing_factor == 0:
        # Linear interpolation between segment centers
        sce = np.interp(pos, segment_centers, segment_values)
    else:
        # Use spline with smoothing
        # Add boundary points for better edge behavior
        x_points = np.concatenate([[pos.min()], segment_centers, [pos.max()]])
        y_points = np.concatenate([[segment_values[0]], segment_values, [segment_values[-1]]])

        # Create smoothing spline (s parameter controls smoothness)
        # Higher s = more smoothing
        spline = UnivariateSpline(x_points, y_points, s=smoothing_factor, k=3)
        sce = spline(pos)

        # Ensure values stay within [0, 1] after smoothing
        sce = np.clip(sce, 0, 1)

    return sce, segment_centers


def calculate_eqe_from_sce(sce, X, inc_flux):
    """
    Calculate EQE from SCE using the forward model: y = X @ sce

    Parameters:
    -----------
    sce : array
        Collection efficiency at each depth point
    X : array
        Feature matrix (n_wavelengths x n_depths)
    inc_flux : array
        Incident photon flux at each wavelength

    Returns:
    --------
    eqe_fitted : array
        Calculated EQE values
    """
    y_fitted = X @ sce
    eqe_fitted = y_fitted / inc_flux
    return eqe_fitted


def manual_sce_fitting_tab(pos, gen, eqe, sun_spec, use_white_light, gen_wavelengths=None,
                            wl_min=None, wl_max=None, n_segments_default=10, smoothing_factor_external=None):
    """
    Create an interactive manual SCE fitting interface.

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
        Whether to use white light spectrum
    gen_wavelengths : array, optional
        Wavelengths from generation file header
    wl_min : float, optional
        Minimum wavelength to include
    wl_max : float, optional
        Maximum wavelength to include
    n_segments_default : int, optional
        Default number of segments (default: 10)
    smoothing_factor_external : float, optional
        Smoothing factor from external control (if None, create internal slider)
    """

    st.markdown("""
    Manually define the spatial collection efficiency (SCE) profile by adjusting sliders for each segment.
    The app will instantly calculate and display the resulting EQE curve.
    """)

    # Import prepare_input from the main app
    # Since we're in a separate module, we need to recreate the preparation logic
    from scipy.constants import physical_constants as pc
    from scipy.interpolate import interp1d

    # Prepare the data (same as in main app)
    if use_white_light:
        sun_spec = sun_spec.copy()
        sun_spec[:, 1] = 1  # WHITE light

    h = pc['Planck constant'][0]
    c_nm = pc['speed of light in vacuum'][0] * 1e9

    # Get EQE data
    eqe_wavelengths, eqe_vals = eqe[:, 0], eqe[:, 1]

    # If generation wavelengths are provided, interpolate EQE to match
    if gen_wavelengths is not None and len(gen_wavelengths) == gen.shape[1]:
        interp_eqe = interp1d(eqe_wavelengths, eqe_vals, kind='linear',
                             bounds_error=False, fill_value=0)
        lam = gen_wavelengths
        eqe_vals_interp = interp_eqe(lam)
    else:
        lam = eqe_wavelengths
        eqe_vals_interp = eqe_vals

    # Apply wavelength filter if specified
    if wl_min is not None or wl_max is not None:
        wl_mask = np.ones(len(lam), dtype=bool)
        if wl_min is not None:
            wl_mask &= (lam >= wl_min)
        if wl_max is not None:
            wl_mask &= (lam <= wl_max)

        lam = lam[wl_mask]
        eqe_vals_interp = eqe_vals_interp[wl_mask]
        gen = gen[:, wl_mask]

    # Build feature matrix
    dp = np.diff(pos)
    weights = np.concatenate(([0.5*dp[0]], 0.5*(dp[:-1]+dp[1:]), [0.5*dp[-1]]))
    X = (gen / 1e21).T * weights

    # Interpolate solar spectrum
    interp_flux = interp1d(sun_spec[:, 0], sun_spec[:, 1], bounds_error=False, fill_value=0)
    photon_flux = interp_flux(lam) / (h * c_nm / lam) / 1e18
    y_measured = eqe_vals_interp * photon_flux

    # Settings section
    st.markdown("---")

    # Use external n_segments if provided, otherwise it's already set to default
    n_segments = n_segments_default

    # Use external smoothing factor if provided, otherwise create slider
    if smoothing_factor_external is not None:
        smoothing_factor = smoothing_factor_external
    else:
        smoothing_factor = st.slider(
            "Smoothing factor",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            help="0 = linear interpolation between segments, higher values = smoother curves"
        )

    col_settings1, col_settings2 = st.columns([1, 2])

    with col_settings1:
        preset = st.selectbox(
            "Load preset SCE profile",
            ["Custom", "Uniform (0.5)", "Linear (front to back)", "Linear (back to front)",
             "Parabolic (peak center)", "Step (front half)", "Step (back half)"],
            help="Load a predefined SCE pattern as starting point"
        )

    with col_settings2:
        st.markdown("**Quick controls:**")
        col_quick1, col_quick2, col_quick3 = st.columns(3)
        with col_quick1:
            if st.button("Set all to 0.0", use_container_width=True):
                st.session_state['reset_sce'] = 'zero'
        with col_quick2:
            if st.button("Set all to 0.5", use_container_width=True):
                st.session_state['reset_sce'] = 'half'
        with col_quick3:
            if st.button("Set all to 1.0", use_container_width=True):
                st.session_state['reset_sce'] = 'one'

    # Initialize or reset segment values
    segment_positions = np.linspace(pos.min(), pos.max(), n_segments + 1)
    segment_centers = (segment_positions[:-1] + segment_positions[1:]) / 2

    # Apply preset if selected
    if 'last_preset' not in st.session_state or st.session_state['last_preset'] != preset:
        st.session_state['last_preset'] = preset
        if preset == "Uniform (0.5)":
            st.session_state['reset_sce'] = 'half'
        elif preset == "Linear (front to back)":
            st.session_state['preset_values'] = np.linspace(1.0, 0.0, n_segments)
        elif preset == "Linear (back to front)":
            st.session_state['preset_values'] = np.linspace(0.0, 1.0, n_segments)
        elif preset == "Parabolic (peak center)":
            x_norm = np.linspace(-1, 1, n_segments)
            st.session_state['preset_values'] = 1.0 - x_norm**2
        elif preset == "Step (front half)":
            half = n_segments // 2
            st.session_state['preset_values'] = np.concatenate([np.ones(half), np.zeros(n_segments - half)])
        elif preset == "Step (back half)":
            half = n_segments // 2
            st.session_state['preset_values'] = np.concatenate([np.zeros(half), np.ones(n_segments - half)])

    # Apply quick control resets
    if 'reset_sce' in st.session_state:
        if st.session_state['reset_sce'] == 'zero':
            st.session_state['preset_values'] = np.zeros(n_segments)
        elif st.session_state['reset_sce'] == 'half':
            st.session_state['preset_values'] = np.ones(n_segments) * 0.5
        elif st.session_state['reset_sce'] == 'one':
            st.session_state['preset_values'] = np.ones(n_segments)
        del st.session_state['reset_sce']

    # Create sliders for each segment
    st.markdown("---")
    st.subheader("SCE segment values")
    st.markdown(f"Adjust the collection efficiency for each of the {n_segments} segments:")

    # Create columns for sliders (max 4 per row)
    sliders_per_row = 4
    segment_values = []

    for row_start in range(0, n_segments, sliders_per_row):
        cols = st.columns(min(sliders_per_row, n_segments - row_start))

        for col_idx, i in enumerate(range(row_start, min(row_start + sliders_per_row, n_segments))):
            with cols[col_idx]:
                # Get default value
                if 'preset_values' in st.session_state and len(st.session_state['preset_values']) == n_segments:
                    default_val = float(st.session_state['preset_values'][i])
                else:
                    default_val = 0.5

                # Create slider
                val = st.slider(
                    f"Segment {i+1}",
                    min_value=0.0,
                    max_value=1.0,
                    value=default_val,
                    step=0.01,
                    key=f"sce_segment_{n_segments}_{i}",
                    help=f"SCE for segment at depth {segment_centers[i]:.1f} nm"
                )
                segment_values.append(val)

    segment_values = np.array(segment_values)

    # Create full SCE profile from segments
    sce_profile, segment_centers = create_sce_from_segments(pos, segment_values, smoothing_factor)

    # Calculate fitted EQE
    eqe_fitted = calculate_eqe_from_sce(sce_profile, X, photon_flux)

    # Calculate metrics
    mse = np.mean((eqe_vals_interp - eqe_fitted)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(eqe_vals_interp - eqe_fitted))
    r2 = 1 - np.sum((eqe_vals_interp - eqe_fitted)**2) / np.sum((eqe_vals_interp - np.mean(eqe_vals_interp))**2)

    # Display metrics
    st.markdown("---")
    st.subheader("Fit quality metrics")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("MSE", f"{mse:.6f}")
    col_m2.metric("RMSE", f"{rmse:.6f}")
    col_m3.metric("MAE", f"{mae:.6f}")
    col_m4.metric("R²", f"{r2:.4f}")

    # Create plots
    st.markdown("---")
    st.subheader("Results")

    # Create 2-column layout for plots
    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        # SCE Profile Plot
        fig_sce = go.Figure()

        # Add SCE profile
        fig_sce.add_trace(go.Scatter(
            x=pos,
            y=sce_profile,
            mode='lines',
            name='Manual SCE',
            line=dict(color='darkblue', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 139, 0.2)',
            hovertemplate='Depth: %{x:.1f} nm<br>SCE: %{y:.4f}<extra></extra>'
        ))

        # Add segment markers
        fig_sce.add_trace(go.Scatter(
            x=segment_centers,
            y=segment_values,
            mode='markers',
            name='Segment values',
            marker=dict(color='red', size=10, symbol='diamond'),
            hovertemplate='Segment center: %{x:.1f} nm<br>SCE: %{y:.4f}<extra></extra>'
        ))

        fig_sce.update_layout(
            title=dict(text='Manual SCE Profile', font=dict(size=16, family='Arial Black')),
            xaxis=dict(title='Depth (nm)', gridcolor='lightgray'),
            yaxis=dict(title='Collection Efficiency (-)', range=[0, 1], gridcolor='lightgray'),
            template='plotly_white',
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )

        st.plotly_chart(fig_sce, use_container_width=True)

    with col_plot2:
        # EQE Fit Plot
        fig_eqe = go.Figure()

        # Measured EQE
        fig_eqe.add_trace(go.Scatter(
            x=lam,
            y=eqe_vals_interp,
            mode='lines+markers',
            name='Measured EQE',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
            hovertemplate='λ: %{x:.1f} nm<br>EQE (measured): %{y:.4f}<extra></extra>'
        ))

        # Fitted EQE
        fig_eqe.add_trace(go.Scatter(
            x=lam,
            y=eqe_fitted,
            mode='lines',
            name='Fitted EQE (manual SCE)',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='λ: %{x:.1f} nm<br>EQE (fitted): %{y:.4f}<extra></extra>'
        ))

        fig_eqe.update_layout(
            title=dict(text='Measured vs Manual Fitted EQE', font=dict(size=16, family='Arial Black')),
            xaxis=dict(title='Wavelength (nm)', gridcolor='lightgray'),
            yaxis=dict(title='EQE (-)', gridcolor='lightgray'),
            template='plotly_white',
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )

        st.plotly_chart(fig_eqe, use_container_width=True)

    # Optional: Generation profile visualization
    show_generation = st.checkbox(
        "Show generation profile overlay",
        value=False,
        help="Display the total generation profile alongside SCE"
    )

    if show_generation:
        # Calculate total generation
        total_generation = np.trapz(gen, lam, axis=1)

        # Create combined plot
        fig_combined = go.Figure()

        # Add SCE on primary y-axis
        fig_combined.add_trace(go.Scatter(
            x=pos,
            y=sce_profile,
            mode='lines',
            name='SCE',
            line=dict(color='darkblue', width=3),
            yaxis='y',
            hovertemplate='Depth: %{x:.1f} nm<br>SCE: %{y:.4f}<extra></extra>'
        ))

        # Add generation on secondary y-axis
        fig_combined.add_trace(go.Scatter(
            x=pos,
            y=total_generation / 1e21,
            mode='lines',
            name='Total Generation',
            line=dict(color='red', width=2, dash='dash'),
            yaxis='y2',
            hovertemplate='Depth: %{x:.1f} nm<br>Generation: %{y:.2f}×10²¹ cm⁻³s⁻¹<extra></extra>'
        ))

        # Add collected generation (SCE × G)
        collected_gen = sce_profile * total_generation / 1e21
        fig_combined.add_trace(go.Scatter(
            x=pos,
            y=collected_gen,
            mode='lines',
            name='Collected Generation (SCE×G)',
            line=dict(width=0),
            fill='tozeroy',
            fillcolor='rgba(100, 149, 237, 0.3)',
            yaxis='y2',
            hovertemplate='Depth: %{x:.1f} nm<br>Collected: %{y:.2f}×10²¹ cm⁻³s⁻¹<extra></extra>'
        ))

        fig_combined.update_layout(
            title=dict(text='SCE and Generation Profile', font=dict(size=16, family='Arial Black')),
            xaxis=dict(title='Depth (nm)', gridcolor='lightgray'),
            yaxis=dict(
                title='SCE (-)',
                titlefont=dict(color='darkblue'),
                tickfont=dict(color='darkblue'),
                range=[0, 1],
                gridcolor='lightgray'
            ),
            yaxis2=dict(
                title='Generation (×10²¹ cm⁻³s⁻¹)',
                titlefont=dict(color='red'),
                tickfont=dict(color='red'),
                overlaying='y',
                side='right',
                showgrid=False
            ),
            template='plotly_white',
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )

        st.plotly_chart(fig_combined, use_container_width=True)

    # Download section
    st.markdown("---")
    st.subheader("Export results")

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        # Export SCE profile
        import io
        sce_buffer = io.StringIO()
        sce_export = np.column_stack((pos, sce_profile))
        np.savetxt(sce_buffer, sce_export, fmt='%.6f\t%.6f',
                   header=f'pos(nm)\tSCE (manual, {n_segments} segments)')

        st.download_button(
            label="📥 Download manual SCE profile",
            data=sce_buffer.getvalue(),
            file_name="manual_SCE_profile.txt",
            mime="text/plain"
        )

    with col_dl2:
        # Export fitted EQE
        eqe_buffer = io.StringIO()
        eqe_export = np.column_stack((lam, eqe_vals_interp, eqe_fitted))
        np.savetxt(eqe_buffer, eqe_export, fmt='%.6f\t%.6f\t%.6f',
                   header='wavelength(nm)\tEQE_measured\tEQE_fitted')

        st.download_button(
            label="📥 Download EQE comparison",
            data=eqe_buffer.getvalue(),
            file_name="manual_EQE_fit.txt",
            mime="text/plain"
        )
