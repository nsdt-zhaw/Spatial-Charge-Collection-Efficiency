"""
Manual SCE Fitting Module
Upload a custom SCE profile and see the resulting EQE fit
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.constants import physical_constants as pc
from scipy.interpolate import interp1d
import io


def manual_sce_fitting_tab(pos, gen, eqe, sun_spec, use_white_light, gen_wavelengths=None,
                            wl_min=None, wl_max=None, is_photon_flux=False, **kwargs):
    """Upload a custom SCE profile and calculate the resulting EQE."""

    # Prepare the data
    if use_white_light and not is_photon_flux:
        sun_spec = sun_spec.copy()
        sun_spec[:, 1] = 1

    h = pc['Planck constant'][0]
    c_nm = pc['speed of light in vacuum'][0] * 1e9

    eqe_wavelengths, eqe_vals = eqe[:, 0], eqe[:, 1]

    if gen_wavelengths is not None and len(gen_wavelengths) == gen.shape[1]:
        interp_eqe = interp1d(eqe_wavelengths, eqe_vals, kind='linear',
                             bounds_error=False, fill_value=0)
        lam = gen_wavelengths
        eqe_vals_interp = interp_eqe(lam)
    else:
        lam = eqe_wavelengths
        eqe_vals_interp = eqe_vals

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

    interp_flux = interp1d(sun_spec[:, 0], sun_spec[:, 1], bounds_error=False, fill_value=0)
    if is_photon_flux:
        photon_flux = interp_flux(lam) / 1e18
    else:
        photon_flux = interp_flux(lam) / (h * c_nm / lam) / 1e18

    # Upload SCE file
    st.caption(f"Expected position range: {pos.min():.1f} - {pos.max():.1f} nm")
    uploaded_sce = st.file_uploader("Upload SCE profile (position [nm], SCE)", type=['txt', 'csv', 'dat'])

    if uploaded_sce is None:
        return

    try:
        uploaded_sce.seek(0)
        sce_data = np.loadtxt(uploaded_sce, ndmin=2)
        sce_pos = sce_data[:, 0]
        sce_vals = sce_data[:, 1]

        # Interpolate to match position grid
        sce_interp = interp1d(sce_pos, sce_vals, kind='linear',
                              bounds_error=False, fill_value='extrapolate')
        sce_profile = np.clip(sce_interp(pos), 0, 1)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return

    # Calculate fitted EQE
    y_fitted = X @ sce_profile
    eqe_fitted = y_fitted / photon_flux

    # Calculate metrics
    mse = np.mean((eqe_vals_interp - eqe_fitted)**2)
    r2 = 1 - np.sum((eqe_vals_interp - eqe_fitted)**2) / np.sum((eqe_vals_interp - np.mean(eqe_vals_interp))**2)

    # Display metrics
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("MSE", f"{mse:.6f}")
    col_m2.metric("R²", f"{r2:.4f}")

    # Create plots
    col1, col2 = st.columns(2)

    with col1:
        fig_sce = go.Figure()
        fig_sce.add_trace(go.Scatter(x=pos, y=sce_profile, mode='lines', line=dict(color='darkblue', width=2)))
        fig_sce.update_layout(title='SCE Profile', xaxis_title='Depth (nm)', yaxis_title='SCE',
                              yaxis_range=[0, 1], template='plotly_white', height=350)
        st.plotly_chart(fig_sce, use_container_width=True)

    with col2:
        fig_eqe = go.Figure()
        fig_eqe.add_trace(go.Scatter(x=lam, y=eqe_vals_interp, mode='lines', name='Measured', line=dict(color='blue', width=2)))
        fig_eqe.add_trace(go.Scatter(x=lam, y=eqe_fitted, mode='lines', name='From SCE', line=dict(color='red', width=2, dash='dash')))
        fig_eqe.update_layout(title='EQE Comparison', xaxis_title='Wavelength (nm)', yaxis_title='EQE',
                              template='plotly_white', height=350, legend=dict(x=0.02, y=0.98))
        st.plotly_chart(fig_eqe, use_container_width=True)

    # Download EQE comparison
    eqe_buffer = io.StringIO()
    np.savetxt(eqe_buffer, np.column_stack((lam, eqe_vals_interp, eqe_fitted)),
               fmt='%.6f\t%.6f\t%.6f', header='wavelength(nm)\tEQE_measured\tEQE_from_SCE')
    st.download_button("Download EQE comparison", data=eqe_buffer.getvalue(),
                       file_name="EQE_comparison.txt", mime="text/plain")
