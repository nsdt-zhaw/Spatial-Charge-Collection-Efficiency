"""
TMM Generation Profile Creator Component - Enhanced with Layer Reordering
==========================================================================

Optional component for SCE extraction app that allows users to generate
generation profiles from layer structures using Transfer Matrix Method.

Now supports:
- Incoherent layers (e.g., thick glass substrates)
- Layer reordering via up/down arrow buttons
- Layer deletion via delete button
- Persistent active layer tracking
- Drag-and-drop style UI

Usage in app.py:
    from tmm_generator import generation_profile_creator
    
    # In your generation profile section:
    generation_profile_creator()

Requirements:
    pip install tmm plotly

Author: For SCE Extraction Streamlit App
"""

import streamlit as st
import numpy as np
from scipy.interpolate import interp1d
import io
from pathlib import Path


def list_nk_files_in_directory(directory="resources/nk data", extensions=['.txt', '.nk', '.dat']):
    """List all n-k files with given extensions in a directory."""
    path = Path(directory)
    if not path.exists():
        return []
    files = []
    for ext in extensions:
        files.extend([f.name for f in path.glob(f'*{ext}')])
    return sorted(files)


def load_nk_file(file):
    """
    Load n,k data from uploaded file or file path with automatic unit conversion.

    Parameters:
    -----------
    file : UploadedFile or str
        Streamlit uploaded file object or file path (string)

    Returns:
    --------
    function
        Function that takes wavelength (nm) and returns complex n+ik
    """
    # Handle different encodings for local files
    if isinstance(file, str):
        # It's a file path - read the file with proper encoding first
        data = None
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                # Read file with specific encoding
                with open(file, 'r', encoding=encoding) as f:
                    lines = f.readlines()

                # Find the first line that looks like data (starts with a number)
                data_start = 0
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line and not line.startswith('#') and line[0].isdigit():
                        data_start = i
                        break

                # Load data from the data start line
                data = np.loadtxt(file, skiprows=data_start, encoding=encoding)
                break
            except (UnicodeDecodeError, ValueError, OSError) as e:
                continue

        if data is None:
            # If all encodings fail, raise error
            raise ValueError(f"Could not load file {file} with any supported encoding")
    else:
        # It's an uploaded file object
        try:
            # Try loading with header skip
            data = np.loadtxt(file, skiprows=1)
        except:
            # Try without skip
            data = np.loadtxt(file)
    
    # Extract wavelength, n, k
    wavelengths = data[:, 0]
    n_data = data[:, 1]
    k_data = data[:, 2]
    
    # Automatic unit conversion to nm
    if wavelengths.max() < 1e-6:  # Likely in meters
        wavelengths *= 1e9
    elif wavelengths.max() < 1e-3:  # Likely in micrometers
        wavelengths *= 1e3
    # else: already in nanometers
    
    # Create interpolation functions
    n_interp = interp1d(wavelengths, n_data, kind='linear', 
                        bounds_error=False, fill_value='extrapolate')
    k_interp = interp1d(wavelengths, k_data, kind='linear', 
                        bounds_error=False, fill_value='extrapolate')
    
    # Return function that gives n+ik at any wavelength
    return lambda wl: n_interp(wl) + 1j * k_interp(wl)


def load_solar_spectrum(file):
    """
    Load solar spectrum from file and convert to photon flux.
    
    Expected file format:
    - Two columns: wavelength (nm), irradiance (W/m²/nm)
    - Comments start with #
    
    Parameters:
    -----------
    file : UploadedFile or str
        Streamlit uploaded file object or file path
    
    Returns:
    --------
    wavelengths : ndarray
        Wavelengths in nm
    photon_flux : ndarray
        Photon flux in photons/cm²/s/nm
    """
    # Load data, skipping comment lines
    data = np.loadtxt(file, comments='#')
    
    wavelengths = data[:, 0]  # nm
    irradiance = data[:, 1]   # W/m²/nm
    
    # Convert irradiance (W/m²/nm) to photon flux (photons/cm²/s/nm)
    # E_photon = h*c/λ (J)
    # Photon flux = Irradiance / E_photon
    
    h = 6.62607015e-34  # Planck constant (J·s)
    c = 2.99792458e8    # Speed of light (m/s)
    
    # Convert wavelength to meters for energy calculation
    wavelengths_m = wavelengths * 1e-9
    
    # Energy per photon (J)
    E_photon = h * c / wavelengths_m
    
    # Photon flux in photons/m²/s/nm
    photon_flux_m2 = irradiance / E_photon
    
    # Convert to photons/cm²/s/nm
    photon_flux = photon_flux_m2 / 1e4  # 1 m² = 10^4 cm²
    
    return wavelengths, photon_flux


def interpolate_solar_spectrum(spec_wavelengths, spec_flux, target_wavelengths):
    """
    Interpolate solar spectrum to target wavelength range.
    
    Parameters:
    -----------
    spec_wavelengths : ndarray
        Original spectrum wavelengths (nm)
    spec_flux : ndarray
        Original spectrum photon flux (photons/cm²/s/nm)
    target_wavelengths : ndarray
        Target wavelengths for interpolation (nm)
    
    Returns:
    --------
    ndarray
        Interpolated photon flux at target wavelengths
    """
    # Create interpolation function
    interp_func = interp1d(
        spec_wavelengths, 
        spec_flux, 
        kind='linear',
        bounds_error=False,
        fill_value=0  # Zero outside range
    )
    
    return interp_func(target_wavelengths)


def calculate_generation_tmm(materials, thicknesses, coherency_list, active_layer_idx, 
                              wavelengths, solar_flux, n_depths=500):
    """
    Calculate generation profile G(z,λ) using Transfer Matrix Method with support
    for incoherent layers.
    
    Parameters:
    -----------
    materials : list of functions
        Each function takes wavelength (nm) and returns complex n+ik
    thicknesses : list of float
        Layer thicknesses in nm (use np.inf for semi-infinite)
    coherency_list : list of str
        'c' or 'i' for each layer
    active_layer_idx : int
        Index of active layer (0-indexed)
    wavelengths : ndarray
        Array of wavelengths in nm
    solar_flux : ndarray
        Photon flux at each wavelength (photons/cm²/s/nm)
    n_depths : int
        Number of depth points in active layer (default: 500)
    
    Returns:
    --------
    z_positions : ndarray
        Depth positions in active layer (nm)
    generation : ndarray
        Generation rate G(z,λ) in cm⁻³s⁻¹, shape (n_depths, n_wavelengths)
    """
    import tmm
    
    # Get active layer thickness
    active_thickness = thicknesses[active_layer_idx]
    
    # Create depth array in active layer
    z_positions = np.linspace(0, active_thickness, n_depths)
    
    # Helper function to find coherent stack index and layer index within that stack
    def find_coh_stack_indices(layer_idx, c_list):
        """
        Find which coherent stack a layer belongs to and its index within that stack.
        Returns (stack_index, layer_in_stack_index) or (None, None) if layer is incoherent.
        """
        if c_list[layer_idx] == 'i':
            return None, None
        
        # Find the coherent stack this layer belongs to
        stack_idx = 0
        layer_in_stack = 0
        in_current_stack = False
        
        for i in range(len(c_list)):
            if c_list[i] == 'c':
                if not in_current_stack:
                    # Starting a new coherent stack
                    in_current_stack = True
                    layer_in_stack = 0
                
                if i == layer_idx:
                    # Found our layer
                    # Add 1 because TMM uses 0 for front interface, 1 for first layer, etc.
                    return stack_idx, layer_in_stack + 1
                
                layer_in_stack += 1
            else:
                # Incoherent layer - end current stack if we were in one
                if in_current_stack:
                    stack_idx += 1
                    in_current_stack = False
        
        return None, None
    
    # Find where the active layer is in the coherent stack structure
    coh_stack_idx, layer_in_coh_stack = find_coh_stack_indices(active_layer_idx, coherency_list)
    
    # Storage for generation profiles
    generation_2d = []
    
    # Loop over wavelengths
    for wavelength, flux in zip(wavelengths, solar_flux):
        # Get complex refractive indices for all layers at this wavelength
        n_list = [material(wavelength) for material in materials]
        
        # Run TMM calculation with incoherent support
        try:
            # Use inc_tmm (incoherent/mixed) if any layers are incoherent
            if 'i' in coherency_list:
                result = tmm.inc_tmm('s', n_list, thicknesses, coherency_list, 0, wavelength)
            else:
                # All coherent - use standard coherent TMM
                result = tmm.coh_tmm('s', n_list, thicknesses, 0, wavelength)
        except Exception as e:
            # If calculation fails, set generation to zero
            generation_2d.append(np.zeros(n_depths))
            continue
        
        # Get absorption profile in active layer
        absorption_profile = []
        
        for z in z_positions:
            try:
                if 'i' not in coherency_list:
                    # Fully coherent stack - standard method
                    data = tmm.position_resolved(active_layer_idx, z, result)
                    absorption_profile.append(data['absor'])
                elif coh_stack_idx is not None and layer_in_coh_stack is not None:
                    # Active layer is part of a coherent stack within mixed system
                    # Get the coherent TMM result for this stack
                    stack_coh_data = result['coh_tmm_data_list'][coh_stack_idx]
                    
                    # Use position_resolved with the correct layer index within the stack
                    data = tmm.position_resolved(layer_in_coh_stack, z, stack_coh_data)
                    absorption_profile.append(data['absor'])
                else:
                    # Active layer is incoherent - cannot get position-resolved profile
                    absorption_profile.append(0)
                    
            except Exception as e:
                absorption_profile.append(0)
        
        absorption_profile = np.array(absorption_profile)
        
        # Convert absorption to generation rate
        # absorption is in nm⁻¹, convert to cm⁻¹ and multiply by flux
        generation = absorption_profile * 1e7 * flux  # cm⁻³s⁻¹
        
        generation_2d.append(generation)
    
    # Convert to 2D array with shape (n_depths, n_wavelengths)
    generation_2d = np.array(generation_2d).T
    
    return z_positions, generation_2d


def generation_profile_creator():
    """
    Streamlit UI component for creating generation profiles using TMM.
    
    This creates an expander with UI for:
    - Defining layer stack structure
    - Uploading n-k files for each layer
    - Reordering layers with up/down buttons
    - Deleting layers with delete button
    - Marking layers as coherent or incoherent
    - Specifying wavelength range
    - Calculating generation profile
    - Downloading result
    
    Call this function in your app.py where you want the TMM generator to appear.
    """
    
    # Check if required packages are available
    try:
        import tmm
    except ImportError:
        st.warning("⚠️ TMM package not installed. Run: `pip install tmm`")
        return
    
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("⚠️ Plotly package not installed. Run: `pip install plotly`")
        return
    
    with st.expander("Generate G(z) from Layer Structure (TMM)", expanded=False):
        
        st.markdown("""
        Create generation profile from optical simulation using Transfer Matrix Method.
        Upload n-k files (wavelength, n, k) for each layer and specify thicknesses.
        """)
        
        # ====================================================================
        # Initialize layer state in session state with proper tracking
        # ====================================================================
        
        if 'layer_order' not in st.session_state:
            st.session_state.layer_order = []
        
        if 'layer_configs' not in st.session_state:
            st.session_state.layer_configs = {}
        
        # ====================================================================
        # Stack Definition
        # ====================================================================
        
        st.markdown("#### 📚 Define Stack Structure")
        st.markdown("*Air (incident medium) is added automatically as first layer*")
        
        st.info("""
        💡 **Stack Tips:**
        - Define layers from front to back (e.g., Glass → ITO → Perovskite → C60 → Metal)
        - **n-k files**: Select from local files in `resources/nk data` or upload custom files
        - For **opaque** back contacts (Au, Ag, Al): Check "Semi-infinite"
        - For **transparent** devices (ITO back, bifacial): Uncheck "Semi-infinite" and set thickness
        - Active layer = absorber where generation profile is extracted
        - **Thick layers** (>1 μm, like glass): Mark as "Incoherent" to remove interference fringes
        - **Glass thickness examples**: 0.7 mm = 700,000 nm, 1 mm = 1,000,000 nm
        - **Reorder layers** using the ⬆️ and ⬇️ buttons
        - **Delete layers** using the ❌ button
        """, icon="ℹ️")
        
        # Get the current number of layers from session state (for delete updates)
        current_n_layers = len(st.session_state.layer_order) if st.session_state.layer_order else 6
        
        # Number of layers (excluding air which is automatic)
        n_layers = st.number_input(
            "Number of layers (excluding air)",
            min_value=1,
            max_value=10,
            value=current_n_layers,
            help="Example: Glass/ITO/PEDOT/Perovskite/C60/Au = 6 layers. Air is added automatically."
        )
        
        # Initialize or update layer order and configs
        current_layer_ids = set(st.session_state.layer_order)
        
        if len(st.session_state.layer_order) != n_layers:
            # Figure out which layers to add or remove
            new_ids = []
            
            # Keep existing layers up to n_layers
            for idx in st.session_state.layer_order[:n_layers]:
                if idx < n_layers:  # Only keep if within range
                    new_ids.append(idx)
            
            # Add new layer IDs for additional layers
            existing_ids = max(st.session_state.layer_order) if st.session_state.layer_order else -1
            for i in range(len(new_ids), n_layers):
                new_ids.append(existing_ids + i + 1)
            
            st.session_state.layer_order = new_ids
            
            # Clean up configs for removed layers
            removed_ids = current_layer_ids - set(new_ids)
            for removed_id in removed_ids:
                if removed_id in st.session_state.layer_configs:
                    del st.session_state.layer_configs[removed_id]
        
        # Initialize configs for new layers
        for layer_id in st.session_state.layer_order:
            if layer_id not in st.session_state.layer_configs:
                st.session_state.layer_configs[layer_id] = {
                    'name': f'Layer_{layer_id}',
                    'thickness': 100.0,
                    'file': None,
                    'file_source': 'local',  # 'local' or 'upload'
                    'local_file_name': '',   # Name of file in resources/nk data
                    'coherency': 'c'
                }
        
        # Storage for layer information
        layers = [None] * len(st.session_state.layer_order)
        
        # Display layers in order with reordering and deletion controls
        st.markdown("**Layer Configuration** (Reorder with arrows, delete with ❌):")

        # Loop through layers in display order
        for display_idx, layer_id in enumerate(st.session_state.layer_order):
            config = st.session_state.layer_configs[layer_id]
            
            # Create columns: [Up Button] [Layer Config] [Down Button] [Delete Button]
            col_up, col_config, col_down, col_delete = st.columns([0.8, 9.5, 0.8, 0.8])
            
            # ================================================================
            # UP BUTTON
            # ================================================================
            with col_up:
                if display_idx > 0:
                    if st.button("⬆️", key=f"btn_up_{layer_id}", help="Move layer up"):
                        # Swap with previous
                        st.session_state.layer_order[display_idx], st.session_state.layer_order[display_idx - 1] = \
                            st.session_state.layer_order[display_idx - 1], st.session_state.layer_order[display_idx]
                        st.rerun()
                else:
                    st.write("")  # Empty space to keep alignment
            
            # ================================================================
            # LAYER CONFIGURATION
            # ================================================================
            with col_config:
                config_col1, config_col2, config_col3, config_col4 = st.columns([2, 1, 2, 1])

                with config_col1:
                    config['name'] = st.text_input(
                        f"Layer {display_idx + 1} name",
                        value=config['name'],
                        key=f"layer_name_{layer_id}",
                        help="Give this layer a descriptive name"
                    )

                with config_col2:
                    # Allow user to choose semi-infinite or finite for last layer
                    if display_idx == len(st.session_state.layer_order) - 1:
                        # Special handling for last layer in display order
                        is_semi_inf = st.checkbox(
                            "Semi-infinite",
                            value=config['thickness'] == np.inf,
                            key=f"semi_inf_{layer_id}",
                            help="Check if opaque/thick back contact. Uncheck for transparent back electrode."
                        )

                        if is_semi_inf:
                            st.text_input(
                                "Thickness (nm)",
                                value="∞",
                                disabled=True,
                                key=f"thickness_disabled_{layer_id}"
                            )
                            config['thickness'] = np.inf
                        else:
                            config['thickness'] = st.number_input(
                                "Thickness (nm)",
                                min_value=1.0,
                                max_value=None,
                                value=config['thickness'] if config['thickness'] != np.inf else 100.0,
                                key=f"thickness_{layer_id}",
                                help="Finite thickness for transparent back electrode",
                                format="%.1f"
                            )
                    else:
                        config['thickness'] = st.number_input(
                            "Thickness (nm)",
                            min_value=1.0,
                            max_value=None,
                            value=config['thickness'] if config['thickness'] != np.inf else 100.0,
                            key=f"thickness_{layer_id}",
                            help="Layer thickness in nanometers (e.g., glass: 700000 nm = 0.7 mm)",
                            format="%.1f"
                        )

                with config_col3:
                    # File source selection
                    # Get available n-k files
                    nk_files = list_nk_files_in_directory("resources/nk data")

                    # Ensure backward compatibility - if file_source doesn't exist, initialize it
                    if 'file_source' not in config:
                        config['file_source'] = 'local'
                    if 'local_file_name' not in config:
                        config['local_file_name'] = ''

                    file_source = st.radio(
                        "Source:",
                        ["Local file", "Upload"],
                        key=f"file_source_{layer_id}",
                        horizontal=True,
                        index=0 if config['file_source'] == 'local' else 1,
                        label_visibility="collapsed"
                    )

                    config['file_source'] = 'local' if file_source == "Local file" else 'upload'

                    if file_source == "Local file" and nk_files:
                        # Show selectbox with local files
                        # Find current selection index
                        if config['local_file_name'] and config['local_file_name'] in nk_files:
                            default_index = nk_files.index(config['local_file_name']) + 1
                        else:
                            default_index = 0

                        selected_file = st.selectbox(
                            "n-k file",
                            [""] + nk_files,
                            key=f"nk_select_{layer_id}",
                            index=default_index,
                            help="Select n-k file from resources/nk data"
                        )

                        config['local_file_name'] = selected_file

                        if selected_file:
                            config['file'] = f"resources/nk data/{selected_file}"
                        else:
                            config['file'] = None
                    else:
                        # Show file uploader
                        config['file'] = st.file_uploader(
                            "n-k file",
                            type=['txt', 'nk', 'dat'],
                            key=f"nk_file_{layer_id}",
                            help="File with columns: wavelength, n, k"
                        )
                        config['local_file_name'] = ''

                with config_col4:
                    # Coherency selection
                    # Auto-suggest incoherent for thick layers
                    default_incoherent = (config['thickness'] > 1000 and config['thickness'] != np.inf)

                    is_incoherent = st.checkbox(
                        "Incoherent",
                        value=config['coherency'] == 'i' or default_incoherent,
                        key=f"incoherent_{layer_id}",
                        help="Check for thick layers (>1 μm) like glass substrates to remove interference effects"
                    )

                    config['coherency'] = 'i' if is_incoherent else 'c'
            
            # ================================================================
            # DOWN BUTTON
            # ================================================================
            with col_down:
                if display_idx < len(st.session_state.layer_order) - 1:
                    if st.button("⬇️", key=f"btn_down_{layer_id}", help="Move layer down"):
                        # Swap with next
                        st.session_state.layer_order[display_idx], st.session_state.layer_order[display_idx + 1] = \
                            st.session_state.layer_order[display_idx + 1], st.session_state.layer_order[display_idx]
                        st.rerun()
                else:
                    st.write("")  # Empty space to keep alignment
            
            # ================================================================
            # DELETE BUTTON
            # ================================================================
            with col_delete:
                if st.button("❌", key=f"btn_delete_{layer_id}", help="Remove this layer"):
                    # Remove layer
                    st.session_state.layer_order.remove(layer_id)
                    if layer_id in st.session_state.layer_configs:
                        del st.session_state.layer_configs[layer_id]
                    st.rerun()
            
            # Store layer info in ordered list
            layers[display_idx] = config
        
        # ====================================================================
        # Active Layer Selection (Persistent)
        # ====================================================================
        
        st.markdown("#### Active Layer")
        
        # Create display names
        layer_names = [f"Layer {i+1}: {layers[i]['name']}" 
                      for i in range(len(layers))]
        
        active_display_idx = st.selectbox(
            "Which layer is the absorber/active layer?",
            options=list(range(len(layers))),
            format_func=lambda x: layer_names[x],
            help="Generation profile will be extracted from this layer",
            key="active_layer_selector"
        )
        
        # Store the active layer configuration (not the index, which can change)
        active_layer_config = layers[active_display_idx]
        
        # ====================================================================
        # Wavelength Range
        # ====================================================================
        
        st.markdown("#### Wavelength Range")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            wl_min = st.number_input(
                "λ min (nm)",
                min_value=200,
                max_value=1000,
                value=350,
                help="Minimum wavelength"
            )
        
        with col2:
            wl_max = st.number_input(
                "λ max (nm)",
                min_value=300,
                max_value=2000,
                value=800,
                help="Maximum wavelength"
            )
        
        with col3:
            n_wavelengths = st.number_input(
                "Number of points",
                min_value=10,
                max_value=500,
                value=100,
                help="Number of wavelength points"
            )
        
        st.markdown("#### Generation Profile Resolution")
        
        n_depths = st.number_input(
            "Number of depth points in active layer",
            min_value=50,
            max_value=1000,
            value=500,
            step=50,
            help="Higher values give finer spatial resolution in G(z,λ) but larger file size"
        )
        
        # ====================================================================
        # Solar Spectrum
        # ====================================================================
        
        st.markdown("#### Spectrum")

        spectrum_option = st.radio(
            "Incident spectrum",
            ["AM1.5G", "White light", "Upload custom spectrum"],
            help="Choose the incident light spectrum for generation profile calculation"
        )

        solar_spectrum_file = None
        if spectrum_option == "Upload custom spectrum":
            solar_spectrum_file = st.file_uploader(
                "Custom spectrum file (.sp, .txt, .dat)",
                type=['sp', 'txt', 'dat'],
                key="solar_spectrum_file",
                help="Upload a custom spectrum file"
            )

            st.info("""
            📄 **File Format Requirements:**
            - Two columns: `wavelength (nm)` and `irradiance (W/m²/nm)`
            - Tab or space separated
            - Comments starting with `#` are allowed
            - Example:
            ```
            # Wavelength(nm)  Irradiance(W/m²/nm)
            300  0.0
            400  1.5
            500  2.0
            ...
            ```
            """)
        
        # ====================================================================
        # Calculate Button
        # ====================================================================
        
        if st.button("Calculate Generation Profile", type="primary"):
            
            # Validate that all files are uploaded
            if any(layer['file'] is None for layer in layers):
                st.error("❌ Please upload n-k files for all layers!")
                return
            
            with st.spinner("🔄 Calculating generation profile with TMM..."):
                
                try:
                    # Load all material files in display order
                    materials = [lambda w: 1.0 + 0j]  # Air (first layer)
                    
                    for layer in layers:
                        material_func = load_nk_file(layer['file'])
                        materials.append(material_func)
                    
                    # Build thickness list (air + layers in display order)
                    thicknesses = [np.inf] + [layer['thickness'] for layer in layers]
                    
                    # Build coherency list in display order
                    coherency_list = ['i'] + [layer['coherency'] for layer in layers]
                    
                    # Check if device is semi-transparent (last layer is finite) BEFORE adding back air
                    is_semitransparent = thicknesses[-1] != np.inf
                    
                    # Check if we need to add air at the back (if last layer is finite)
                    if is_semitransparent:
                        # Add air as final layer for semi-transparent devices
                        materials.append(lambda w: 1.0 + 0j)
                        thicknesses.append(np.inf)
                        coherency_list.append('i')
                        st.info("📌 Added air layer after back electrode (semi-transparent device mode)")
                    
                    # Ensure last layer is incoherent (required by TMM)
                    coherency_list[-1] = 'i'
                    
                    # Find active layer index in the current order
                    active_idx = layers.index(active_layer_config) + 1  # +1 for front air
                    
                    # Check if active layer is coherent
                    if coherency_list[active_idx] == 'i':
                        st.warning("⚠️ Active layer is marked as incoherent. Generation profiles work best with coherent active layers.")
                    
                    # Display coherency status
                    coherent_layers = [layers[i]['name'] for i in range(len(layers)) 
                                      if layers[i]['coherency'] == 'c']
                    incoherent_layers = [layers[i]['name'] for i in range(len(layers)) 
                                        if layers[i]['coherency'] == 'i']
                    
                    if incoherent_layers:
                        st.info(f"🔄 **Incoherent layers:** {', '.join(incoherent_layers)}")
                    
                    # Create wavelength array
                    wavelengths = np.linspace(wl_min, wl_max, n_wavelengths)

                    # Create or load solar spectrum
                    if spectrum_option == "White light":
                        solar_flux = 1e15 * np.ones_like(wavelengths)
                        st.info("✓ Using flat white light spectrum")

                    elif spectrum_option == "AM1.5G":
                        # Load AM1.5G from resources/Sunspectrum.sp
                        import os
                        resources_path = "resources/Sunspectrum.sp"
                        if os.path.exists(resources_path):
                            try:
                                spec_wl, spec_flux = load_solar_spectrum(resources_path)
                                solar_flux = interpolate_solar_spectrum(spec_wl, spec_flux, wavelengths)
                                st.success(f"✅ Loaded AM1.5G from {resources_path}: {len(spec_wl)} points from {spec_wl.min():.0f}-{spec_wl.max():.0f} nm")
                            except Exception as e:
                                st.error(f"Failed to load AM1.5G spectrum: {e}")
                                st.warning("⚠️ Using Gaussian approximation as fallback")
                                solar_flux = 2e15 * np.exp(-((wavelengths - 500) / 150)**2)
                        else:
                            st.warning(f"⚠️ AM1.5G spectrum file not found at {resources_path}")
                            st.info("Place Sunspectrum.sp in the 'resources' folder, or use White light / Upload custom spectrum")
                            st.warning("Using Gaussian approximation as fallback")
                            solar_flux = 2e15 * np.exp(-((wavelengths - 500) / 150)**2)

                    elif spectrum_option == "Upload custom spectrum":
                        # Load from uploaded file
                        if solar_spectrum_file is not None:
                            try:
                                spec_wl, spec_flux = load_solar_spectrum(solar_spectrum_file)
                                solar_flux = interpolate_solar_spectrum(spec_wl, spec_flux, wavelengths)
                                st.success(f"✅ Loaded custom spectrum: {len(spec_wl)} points from {spec_wl.min():.0f}-{spec_wl.max():.0f} nm")
                            except Exception as e:
                                st.error(f"Failed to load uploaded spectrum: {e}")
                                st.warning("⚠️ Using white light as fallback")
                                solar_flux = 1e15 * np.ones_like(wavelengths)
                        else:
                            st.error("❌ Please upload a spectrum file!")
                            st.warning("⚠️ Using white light as fallback")
                            solar_flux = 1e15 * np.ones_like(wavelengths)

                    else:
                        # Default fallback (shouldn't reach here)
                        solar_flux = 1e15 * np.ones_like(wavelengths)
                    
                    # Show spectrum preview
                    with st.expander("📊 Preview Solar Spectrum"):
                        import plotly.graph_objects as go
                        
                        fig_spectrum = go.Figure()
                        fig_spectrum.add_trace(
                            go.Scatter(
                                x=wavelengths,
                                y=solar_flux / 1e15,
                                mode='lines',
                                line=dict(color='rgb(255, 165, 0)', width=2),
                                fill='tozeroy',
                                fillcolor='rgba(255, 165, 0, 0.3)',
                                name='Photon Flux'
                            )
                        )
                        fig_spectrum.update_layout(
                            title="Solar Spectrum (Photon Flux)",
                            xaxis_title="Wavelength (nm)",
                            yaxis_title="Photon Flux (×10¹⁵ photons/cm²/s/nm)",
                            height=300,
                            showlegend=False,
                            plot_bgcolor='white',
                            margin=dict(l=60, r=40, t=60, b=60)
                        )
                        st.plotly_chart(fig_spectrum, use_container_width=True)
                    
                    # Calculate generation profile
                    z_positions, generation = calculate_generation_tmm(
                        materials, thicknesses, coherency_list, active_idx,
                        wavelengths, solar_flux, n_depths
                    )
                    
                    # Calculate reflection and transmission
                    import tmm
                    
                    # Calculate R and T at a sample wavelength
                    test_wl = wavelengths[len(wavelengths)//2]
                    n_list = [mat(test_wl) for mat in materials]
                    
                    if 'i' in coherency_list:
                        result = tmm.inc_tmm('s', n_list, thicknesses, coherency_list, 0, test_wl)
                    else:
                        result = tmm.coh_tmm('s', n_list, thicknesses, 0, test_wl)
                    
                    avg_transmission = result['T']
                    avg_reflection = result['R']
                    
                    # ========================================================
                    # Create output file
                    # ========================================================

                    # Create header in the format matching app.py expectations
                    header_lines = "Charge generation [cm^-3*s^-1]\n"
                    header_lines += "Column format:\n"
                    header_lines += "x (nm)\t" + "\t".join([f"{int(w)} nm" for w in wavelengths])

                    # Combine position and generation data
                    output_data = np.column_stack([z_positions, generation])

                    # Save to string buffer
                    buffer = io.StringIO()
                    np.savetxt(
                        buffer,
                        output_data,
                        fmt='%.6e',
                        delimiter='\t',
                        header=header_lines,
                        comments='# '
                    )
                    
                    file_content = buffer.getvalue()
                    
                    # ========================================================
                    # Success message and preview
                    # ========================================================
                    
                    st.success("✅ Generation profile calculated successfully!")
                    
                    # Interactive Plotly visualization
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots
                    
                    # Calculate total generation (integrated over wavelength)
                    total_generation = np.trapezoid(generation, wavelengths, axis=1)
                    
                    # Calculate absorptance in active layer for each wavelength
                    absorptance_active = []
                    absorptance_all_layers = []  # Store absorption for all layers
                    
                    for lam in wavelengths:
                        n_list = [mat(lam) for mat in materials]
                        try:
                            if 'i' in coherency_list:
                                result = tmm.inc_tmm('s', n_list, thicknesses, coherency_list, 0, lam)
                                # Get absorption in active layer only
                                layer_absorp = tmm.inc_absorp_in_each_layer(result)
                            else:
                                result = tmm.coh_tmm('s', n_list, thicknesses, 0, lam)
                                layer_absorp = tmm.absorp_in_each_layer(result)
                            
                            absorptance_active.append(layer_absorp[active_idx])
                            absorptance_all_layers.append(layer_absorp)
                        except:
                            absorptance_active.append(0)
                            absorptance_all_layers.append([0] * len(materials))
                    
                    absorptance_active = np.array(absorptance_active)
                    absorptance_all_layers = np.array(absorptance_all_layers)
                    
                    # ========================================================
                    # Create 2x2 subplot layout with individual controls
                    # ========================================================
                    
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=(
                            f'Total Generation in {active_layer_config["name"]}',
                            'Wavelength-Dependent Generation G(z,λ)',
                            f'Absorptance in {active_layer_config["name"]} vs Wavelength',
                            'Absorption in All Layers vs Wavelength'
                        ),
                        specs=[
                            [{"type": "scatter"}, {"type": "heatmap"}],
                            [{"type": "scatter"}, {"type": "scatter"}]
                        ],
                        vertical_spacing=0.12,
                        horizontal_spacing=0.10,
                        row_heights=[0.5, 0.5]
                    )
                    
                    # ========================================================
                    # PLOT 1 (Top Left): Total Generation Profile G(z)
                    # ========================================================
                    
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
                        title_text="Depth in Active Layer (nm)",
                        row=1, col=1,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.2)'
                    )
                    
                    fig.update_yaxes(
                        title_text="Total Generation (×10²¹ cm⁻³s⁻¹)",
                        row=1, col=1,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.2)'
                    )
                    
                    # ========================================================
                    # PLOT 2 (Top Right): Wavelength-Dependent Generation G(z,λ) Heatmap
                    # ========================================================
                    
                    fig.add_trace(
                        go.Heatmap(
                            x=wavelengths,
                            y=z_positions,
                            z=generation / 1e21,
                            colorscale='Viridis',
                            colorbar=dict(
                                title=dict(
                                    text="G(z,λ)<br>(×10²¹ cm⁻³s⁻¹)",
                                    side="right"
                                ),
                                x=1.02,
                                len=0.45,
                                y=0.75
                            ),
                            hovertemplate='<b>Wavelength:</b> %{x:.0f} nm<br>' +
                                        '<b>Depth:</b> %{y:.1f} nm<br>' +
                                        '<b>Generation:</b> %{z:.2f}×10²¹ cm⁻³s⁻¹<br>' +
                                        '<extra></extra>',
                            name='G(z,λ)',
                            showscale=True
                        ),
                        row=1, col=2
                    )
                    
                    fig.update_xaxes(
                        title_text="Wavelength (nm)",
                        row=1, col=2,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.2)'
                    )
                    
                    fig.update_yaxes(
                        title_text="Depth (nm)",
                        row=1, col=2,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.2)'
                    )
                    
                    # ========================================================
                    # PLOT 3 (Bottom Left): Active Layer Absorptance vs Wavelength
                    # ========================================================
                    
                    fig.add_trace(
                        go.Scatter(
                            x=wavelengths,
                            y=absorptance_active * 100,
                            mode='lines',
                            line=dict(color='rgb(50, 120, 180)', width=3),
                            fill='tozeroy',
                            fillcolor='rgba(50, 120, 180, 0.3)',
                            name='Absorptance',
                            hovertemplate='<b>Wavelength:</b> %{x:.0f} nm<br>' +
                                        '<b>Absorptance:</b> %{y:.1f}%<br>' +
                                        '<extra></extra>',
                            showlegend=False
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_xaxes(
                        title_text="Wavelength (nm)",
                        row=2, col=1,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.2)'
                    )
                    
                    fig.update_yaxes(
                        title_text="Absorptance (%)",
                        row=2, col=1,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.2)',
                        range=[0, 100]
                    )
                    
                    # ========================================================
                    # PLOT 4 (Bottom Right): Absorption in All Layers
                    # ========================================================
                    
                    # Define a better color palette (more distinct colors)
                    colors = [
                        'rgb(228, 26, 28)',    # Red
                        'rgb(55, 126, 184)',   # Blue
                        'rgb(77, 175, 74)',    # Green
                        'rgb(152, 78, 163)',   # Purple
                        'rgb(255, 127, 0)',    # Orange
                        'rgb(166, 86, 40)',    # Brown
                        'rgb(247, 129, 191)',  # Pink
                        'rgb(153, 153, 153)',  # Gray
                        'rgb(255, 255, 51)',   # Yellow
                        'rgb(0, 255, 255)',    # Cyan
                        'rgb(128, 0, 128)',    # Dark Purple
                        'rgb(0, 128, 128)'     # Teal
                    ]
                    
                    # Calculate total R + T for verification
                    total_R = []
                    total_T = []
                    total_A_sum = []
                    
                    for lam in wavelengths:
                        n_list = [mat(lam) for mat in materials]
                        try:
                            if 'i' in coherency_list:
                                result_rt = tmm.inc_tmm('s', n_list, thicknesses, coherency_list, 0, lam)
                            else:
                                result_rt = tmm.coh_tmm('s', n_list, thicknesses, 0, lam)
                            
                            total_R.append(result_rt['R'])
                            total_T.append(result_rt['T'])
                            
                            # Sum absorption in all layers (should equal 1 - R - T)
                            if 'i' in coherency_list:
                                layer_absorp = tmm.inc_absorp_in_each_layer(result_rt)
                            else:
                                layer_absorp = tmm.absorp_in_each_layer(result_rt)
                            total_A_sum.append(sum(layer_absorp))
                        except:
                            total_R.append(0)
                            total_T.append(0)
                            total_A_sum.append(0)
                    
                    total_R = np.array(total_R)
                    total_T = np.array(total_T)
                    total_A_sum = np.array(total_A_sum)
                    
                    # Plot absorption for each layer (skip air layers only)
                    layer_trace_count = 0
                    for i in range(len(materials)):
                        # Determine layer name
                        if i == 0:
                            layer_name = "Air (front)"
                            # Skip - air doesn't absorb
                            continue
                        elif i == len(materials) - 1 and thicknesses[-1] == np.inf:
                            # This is the back semi-infinite layer
                            if is_semitransparent:
                                # Semi-transparent device: back layer is air
                                layer_name = "Air (back)"
                                # Skip - air doesn't absorb
                                continue
                            else:
                                # Opaque device: back layer is the last material layer (skip, it's semi-infinite)
                                continue
                        elif i - 1 < len(layers):
                            layer_name = layers[i - 1]['name']
                        else:
                            layer_name = f"Layer {i}"
                        
                        # Add trace for this layer (no filtering by absorption amount)
                        fig.add_trace(
                            go.Scatter(
                                x=wavelengths,
                                y=absorptance_all_layers[:, i] * 100,
                                mode='lines',
                                line=dict(color=colors[layer_trace_count % len(colors)], width=2),
                                name=layer_name,
                                hovertemplate='<b>Wavelength:</b> %{x:.0f} nm<br>' +
                                            '<b>Absorptance:</b> %{y:.2f}%<br>' +
                                            f'<b>Layer:</b> {layer_name}<br>' +
                                            '<extra></extra>',
                                legendgroup=f'layer_{i}',  # Unique legend group for each layer
                                showlegend=True
                            ),
                            row=2, col=2
                        )
                        layer_trace_count += 1
                    
                    # Add Reflection and Transmission traces with distinct legend groups
                    fig.add_trace(
                        go.Scatter(
                            x=wavelengths,
                            y=total_R * 100,
                            mode='lines',
                            line=dict(color='rgb(0, 0, 0)', width=2.5, dash='dash'),
                            name='Reflection',
                            hovertemplate='<b>Wavelength:</b> %{x:.0f} nm<br>' +
                                        '<b>Reflectance:</b> %{y:.2f}%<br>' +
                                        '<extra></extra>',
                            legendgroup='reflection',  # Unique legend group
                            showlegend=True
                        ),
                        row=2, col=2
                    )
                    
                    if is_semitransparent:
                        fig.add_trace(
                            go.Scatter(
                                x=wavelengths,
                                y=total_T * 100,
                                mode='lines',
                                line=dict(color='rgb(100, 100, 100)', width=2.5, dash='dot'),
                                name='Transmission',
                                hovertemplate='<b>Wavelength:</b> %{x:.0f} nm<br>' +
                                            '<b>Transmittance:</b> %{y:.2f}%<br>' +
                                            '<extra></extra>',
                                legendgroup='transmission',  # Unique legend group
                                showlegend=True
                            ),
                            row=2, col=2
                        )
                    
                    fig.update_xaxes(
                        title_text="Wavelength (nm)",
                        row=2, col=2,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.2)'
                    )
                    
                    fig.update_yaxes(
                        title_text="Absorptance (%)",
                        row=2, col=2,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.2)'
                    )
                    
                    # Update overall layout
                    fig.update_layout(
                        height=900,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(size=11),
                        title=dict(
                            text="<b>Generation Profile Analysis</b>",
                            x=0.5,
                            xanchor='center',
                            font=dict(size=16)
                        ),
                        legend=dict(
                            yanchor="top",
                            y=0.48,
                            xanchor="right",
                            x=0.99,
                            bgcolor='rgba(255, 255, 255, 0.9)',
                            bordercolor='rgba(0, 0, 0, 0.3)',
                            borderwidth=1,
                            font=dict(size=9),
                            itemclick='toggle',      # Enable click to hide/show
                            itemdoubleclick='toggleothers'  # Double-click to isolate
                        ),
                        margin=dict(l=70, r=120, t=80, b=70)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ========================================================
                    # Download button
                    # ========================================================
                    
                    st.download_button(
                        label="📥 Download Generation Profile",
                        data=file_content,
                        file_name="generation_profile_TMM.txt",
                        mime="text/plain",
                        help="Download this file and use it in the SCE extraction analysis"
                    )
                    
                    # ========================================================
                    # Statistics
                    # ========================================================
                    
                    # Calculate approximate generation current
                    total_gen_cm2 = np.trapezoid(total_generation, z_positions * 1e-7)  # cm⁻²s⁻¹
                    J_gen = 1.60218e-19 * total_gen_cm2 * 1e3  # mA/cm²
                    
                    # Find peak absorptance wavelength
                    peak_abs_idx = np.argmax(absorptance_active)
                    peak_abs_wl = wavelengths[peak_abs_idx]
                    peak_abs_val = absorptance_active[peak_abs_idx]
                    
                    # Display metrics
                    if is_semitransparent:
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        col1.metric(
                            "Max Generation",
                            f"{generation.max():.2e} cm⁻³s⁻¹"
                        )
                        
                        col2.metric(
                            "Predicted J_gen",
                            f"{J_gen:.2f} mA/cm²",
                            help="Maximum current assuming perfect collection (SCE=1)"
                        )
                        
                        col3.metric(
                            "Peak Absorptance",
                            f"{peak_abs_val*100:.1f}%",
                            help=f"Maximum at {peak_abs_wl:.0f} nm"
                        )
                        
                        # col4.metric(
                        #     "Transmission",
                        #     f"{avg_transmission*100:.1f}%",
                        #     help="Light transmitted through device (semi-transparent mode)"
                        # )
                        
                        # col5.metric(
                        #     "Reflection",
                        #     f"{avg_reflection*100:.1f}%",
                        #     help="Light reflected from front surface"
                        # )
                    else:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        col1.metric(
                            "Max Generation",
                            f"{generation.max():.2e} cm⁻³s⁻¹"
                        )
                        
                        col2.metric(
                            "Avg Generation",
                            f"{generation.mean():.2e} cm⁻³s⁻¹"
                        )
                        
                        col3.metric(
                            "Predicted J_gen",
                            f"{J_gen:.2f} mA/cm²",
                            help="Maximum current assuming perfect collection (SCE=1)"
                        )
                        
                        col4.metric(
                            "Peak Absorptance",
                            f"{peak_abs_val*100:.1f}%",
                            help=f"Maximum at {peak_abs_wl:.0f} nm in active layer"
                        )
                    
                    st.info(
                        "💡 **Tip**: Download the file above and use it as the "
                        "generation profile input for SCE extraction analysis."
                    )
                    
                except Exception as e:
                    st.error(f"❌ Calculation failed: {str(e)}")
                    
                    # Show detailed error for debugging
                    with st.expander("🔍 Error Details"):
                        import traceback
                        st.code(traceback.format_exc())


if __name__ == "__main__":
    """
    Standalone demo of the TMM generator component with layer reordering and deletion.
    Run with: streamlit run tmm_generator_final.py
    """
    
    st.set_page_config(page_title="TMM Generator Demo", layout="wide")
    
    st.title("TMM Generation Profile Creator - With Layer Reordering & Deletion")
    
    st.markdown("""
    This is an enhanced demo with **layer reordering and deletion** support.
    
    ### New Features:
    - ⬆️ **Move Up**: Click to move a layer toward the front of the device
    - ⬇️ **Move Down**: Click to move a layer toward the back of the device
    - ❌ **Delete**: Click to remove a layer
    - ✅ **Persistent Active Layer**: Active layer selection remains correct when adding/removing/reordering layers
    - 🔄 **Real-time Updates**: Changes apply immediately
    """)
    
    generation_profile_creator()