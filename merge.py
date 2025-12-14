import streamlit as st
import pyvista as pv
import pandas as pd
import numpy as np
import rioxarray
from scipy.interpolate import RegularGridInterpolator
import math
import tempfile
import os
import streamlit.components.v1 as components
import time

# ==========================================
# 0. CONFIGURATION & SETUP
# ==========================================
st.set_page_config(layout="wide", page_title="Drill Hole Visualizer Pro")

# PyVista configuration for headless/web environments
pv.set_plot_theme("document")

# Default Style Constants from drill.py
DEFAULT_TUBE_RADIUS = 12.0
DEFAULT_COLLAR_RADIUS = 40.0
DEFAULT_V_EXAG = 5

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def save_uploaded_file(uploaded_file):
    """Helper to save uploaded file to a temp path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def safe_remove_file(path):
    """Robust file remover that tolerates Windows file locking."""
    if not path or not os.path.exists(path):
        return
    try:
        os.remove(path)
    except Exception:
        try:
            time.sleep(1.0)
            os.remove(path)
        except Exception:
            pass

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def get_coords_at_depth(row, depth, v_exag, center_x, center_y, z_datum=0):
    """
    Calculates 3D coordinates based on azimuth/dip.
    Adapted from drill.py logic.
    """
    az_rad = math.radians(row['Azimuth'])
    dip_rad = math.radians(row['Dip'])
    
    # Horizontal is 1:1
    dx = depth * math.cos(dip_rad) * math.sin(az_rad)
    dy = depth * math.cos(dip_rad) * math.cos(az_rad)
    
    # Apply centering
    x = (row['X'] - center_x) + dx
    y = (row['Y'] - center_y) + dy
    
    # Vertical is scaled by Exaggeration
    vertical_drop = depth * math.sin(dip_rad)
    
    # Current Z (relative to datum) + vertical component * Exaggeration
    z = row['Z_rel'] + (vertical_drop * v_exag)
    
    return np.array([x, y, z])

def get_grade_color(grade):
    """
    Hardcoded color logic from drill.py
    """
    if grade >= 0.50: return "red"
    elif grade >= 0.30: return "darkorange"
    elif grade >= 0.15: return "yellow"
    elif grade >= 0.05: return "green"
    else: return "lightgray"

# ==========================================
# 2. DEFAULT DATA (From drill.py)
# ==========================================
# We inject the drill.py data here to serve as the default state
default_collars_dict = {
    'HoleID': ['DBW-25-001', 'DBW-25-002', 'DBW-25-004R', 'DBW-25-005', 'DBW-25-006', 'DBW-25-009', 'DBW-25-010', 'DBW-25-003', 'DBW-25-007'], 
    'X': [440375, 440770, 440924, 440435, 440670, 440450, 440300, 440515, 440400], 
    'Y': [8627810, 8628758, 8628758, 8627812, 8627812, 8627310, 8627310, 8627810, 8627310], 
    'Azimuth': [90, 270, 270, 90, 270, 270, 90, 90, 270], 
    'Dip': [-70, -50, -60, -60, -60, -60, -50, -60, -60], 
    'Depth': [386.00, 218.50, 223.90, 294.20, 200.65, 260.00, 170.10, 166.5, 210.0]
}

default_assays_dict = {
    'HoleID': ['DBW-25-010', 'DBW-25-009', 'DBW-25-009', 'DBW-25-010', 'DBW-25-010', 'DBW-25-009', 'DBW-25-006', 'DBW-25-006', 'DBW-25-006', 'DBW-25-003', 'DBW-25-003', 'DBW-25-007', 'DBW-25-007', 'DBW-25-007'], 
    'From': [117, 155, 119.75, 45, 101, 178, 142, 159, 167, 19.2, 56.5, 4.0, 52.0, 100.3], 
    'To': [139, 170, 125, 54, 110, 186, 153, 163, 171, 24.4, 82.4, 20.0, 75.0, 140.0],
    'Grade': [1.00, 0.98, 0.79, 0.31, 0.32, 0.36, 0.25, 0.18, 0.20, 0.29, 0.48, 0.31, 0.33, 0.51] 
}

# ==========================================
# 3. SIDEBAR UI
# ==========================================
st.sidebar.header("1. Visualization Config")
V_EXAG = st.sidebar.slider("Vertical Exaggeration", 1, 10, DEFAULT_V_EXAG)
BUFFER_SIZE = st.sidebar.number_input("Buffer Size (m)", value=1000)

st.sidebar.markdown("---")
st.sidebar.header("2. Terrain Upload")
st.sidebar.info("If no DEM is uploaded, a flat plane will be used.")
dem_file = st.sidebar.file_uploader("Upload DEM (.tif)", type=["tif", "tiff"])
sat_file = st.sidebar.file_uploader("Upload Satellite (.tif)", type=["tif", "tiff"])

# ==========================================
# 4. MAIN UI & DATA INPUT
# ==========================================
st.title("3D Drill Hole Visualizer")
st.markdown("Combines interactive data editing with high-fidelity geospatial rendering.")

tab_collars, tab_assays, tab_info = st.tabs(["1. Drill Collars", "2. Assay Intervals", "3. Legend & Info"])

# --- TAB 1: COLLARS ---
with tab_collars:
    col1, col2 = st.columns([1, 1])
    with col1:
        collar_file = st.file_uploader("Upload Collars (CSV/Excel)", type=["csv", "xlsx", "xls"], key="collar_up")
    with col2:
        st.download_button("⬇️ Download Template", data=convert_df_to_csv(pd.DataFrame(columns=['HoleID', 'X', 'Y', 'Azimuth', 'Dip', 'Depth'])), file_name="collar_template.csv", mime="text/csv")
    
    # Load default drill.py data if no file, otherwise load file
    if collar_file:
        try:
            if collar_file.name.endswith('.csv'): df_collars_in = pd.read_csv(collar_file)
            else: df_collars_in = pd.read_excel(collar_file)
        except: df_collars_in = pd.DataFrame(default_collars_dict)
    else:
        df_collars_in = pd.DataFrame(default_collars_dict)
        
    edited_collars = st.data_editor(df_collars_in, num_rows="dynamic", key="collar_editor", use_container_width=True)

# --- TAB 2: ASSAYS ---
with tab_assays:
    col1, col2 = st.columns([1, 1])
    with col1:
        assay_file = st.file_uploader("Upload Assays (CSV/Excel)", type=["csv", "xlsx", "xls"], key="assay_up")
    with col2:
        st.download_button("⬇️ Download Template", data=convert_df_to_csv(pd.DataFrame(columns=['HoleID', 'From', 'To', 'Grade'])), file_name="assay_template.csv", mime="text/csv")

    if assay_file:
        try:
            if assay_file.name.endswith('.csv'): df_assays_in = pd.read_csv(assay_file)
            else: df_assays_in = pd.read_excel(assay_file)
        except: df_assays_in = pd.DataFrame(default_assays_dict)
    else:
        df_assays_in = pd.DataFrame(default_assays_dict)
        
    edited_assays = st.data_editor(df_assays_in, num_rows="dynamic", key="assay_editor", use_container_width=True)

# --- TAB 3: INFO ---
with tab_info:
    st.markdown("#### Grade Color Logic")
    st.markdown("""
    - <span style='color:red'>■</span> **Red**: Grade >= 0.50
    - <span style='color:darkorange'>■</span> **Orange**: Grade >= 0.30
    - <span style='color:yellow'>■</span> **Yellow**: Grade >= 0.15
    - <span style='color:green'>■</span> **Green**: Grade >= 0.05
    - <span style='color:gray'>■</span> **Gray**: Low Grade / Trace
    """, unsafe_allow_html=True)
    st.info("Render uses Tube Radius: 12.0 | Collar Radius: 40.0")

# ==========================================
# 5. GENERATION LOGIC
# ==========================================
st.markdown("### Step 3: Generate Model")
generate_btn = st.button("Generate 3D Model", type="primary")

if generate_btn:
    if edited_collars.empty:
        st.error("Collar data is required.")
        st.stop()
        
    with st.spinner("Processing terrain, snapping holes, and rendering..."):
        
        # 1. PREP DATA
        df_collars = edited_collars.copy()
        df_assays = edited_assays.copy()
        
        # Ensure numerics
        for col in ['X', 'Y', 'Azimuth', 'Dip', 'Depth']:
            if col in df_collars.columns: df_collars[col] = pd.to_numeric(df_collars[col], errors='coerce')
        if not df_assays.empty:
            for col in ['From', 'To', 'Grade']:
                if col in df_assays.columns: df_assays[col] = pd.to_numeric(df_assays[col], errors='coerce')
        
        # Center Calculation
        CENTER_X = df_collars['X'].mean()
        CENTER_Y = df_collars['Y'].mean()
        
        # Resource cleanup variables
        dem_path, sat_path, html_path = None, None, None
        dem_ds, sat_ds = None, None
        
        terrain_mesh, terrain_tex = None, None
        z_interpolator = None
        avg_elev = 0.0

        try:
            # 2. PROCESS MAPS (If Uploaded)
            if dem_file and sat_file:
                dem_path = save_uploaded_file(dem_file)
                sat_path = save_uploaded_file(sat_file)

                # Load DEM
                dem_ds = rioxarray.open_rasterio(dem_path, masked=True).squeeze()
                
                # Clip to Drill Area + Buffer
                min_x, max_x = df_collars['X'].min() - BUFFER_SIZE, df_collars['X'].max() + BUFFER_SIZE
                min_y, max_y = df_collars['Y'].min() - BUFFER_SIZE, df_collars['Y'].max() + BUFFER_SIZE
                
                dem_clip = dem_ds.rio.clip_box(minx=min_x, miny=min_y, maxx=max_x, maxy=max_y)
                
                # Load Sat
                sat_ds = rioxarray.open_rasterio(sat_path, masked=True)
                sat_clip = sat_ds.rio.clip_box(minx=min_x, miny=min_y, maxx=max_x, maxy=max_y)
                
                # Texture
                sat_data = sat_clip.values.transpose(1, 2, 0)
                if sat_data.dtype != np.uint8:
                    sat_data = ((sat_data - np.nanmin(sat_data)) / (np.nanmax(sat_data) - np.nanmin(sat_data)) * 255).astype(np.uint8)
                terrain_tex = pv.Texture(sat_data)
                
                # Create Grid
                x = dem_clip.x.values - CENTER_X
                y = dem_clip.y.values - CENTER_Y
                z = dem_clip.values
                
                # Datum logic from drill.py
                avg_elev = np.nanmean(z)
                z = np.nan_to_num(z, nan=avg_elev)
                
                # Interpolator for snapping
                z_interpolator = RegularGridInterpolator(
                    (dem_clip.y.values, dem_clip.x.values), dem_clip.values, 
                    bounds_error=False, fill_value=avg_elev
                )
                
                grid = pv.RectilinearGrid(x, y, [0])
                grid["Elevation"] = z.flatten()
                # Warp
                grid["Relative_Z"] = (grid["Elevation"] - avg_elev) * V_EXAG
                terrain_mesh = grid.warp_by_scalar("Relative_Z")
                terrain_mesh.texture_map_to_plane(use_bounds=True, inplace=True)
                
                dem_ds.close()
                sat_ds.close()

            # 3. SNAP COLLARS
            df_collars['Z_rel'] = 0.0
            for i, row in df_collars.iterrows():
                real_z = avg_elev # Default
                if z_interpolator:
                    try:
                        val = z_interpolator((row['Y'], row['X']))
                        if not np.isnan(val): real_z = val
                    except: pass
                
                # Calculate Z relative to datum, then exaggerate
                rel_z = (real_z - avg_elev) * V_EXAG
                # Add offset so collar sits slightly above ground
                df_collars.at[i, 'Z_rel'] = float(rel_z) + 2.0

            # 4. PLOTTING
            plotter = pv.Plotter(window_size=[800, 600])
            plotter.set_background('white')
            
            # A. Terrain
            if terrain_mesh:
                plotter.add_mesh(terrain_mesh, texture=terrain_tex, opacity=1.0)
            else:
                # Fallback Plane if no maps
                plane = pv.Plane(center=(0,0,0), i_size=2000, j_size=2000)
                plotter.add_mesh(plane, color="gray", opacity=0.5, show_edges=True)
            
            # B. Traces & Collars (Style from drill.py)
            for _, row in df_collars.iterrows():
                start = get_coords_at_depth(row, 0, V_EXAG, CENTER_X, CENTER_Y, avg_elev)
                end = get_coords_at_depth(row, row['Depth'], V_EXAG, CENTER_X, CENTER_Y, avg_elev)
                
                # Thick Collar
                plotter.add_mesh(pv.Sphere(radius=DEFAULT_COLLAR_RADIUS, center=start), color="black")
                # Thick Trace
                plotter.add_mesh(pv.Line(start, end).tube(radius=DEFAULT_TUBE_RADIUS), color="lightgrey", opacity=0.5)
                # Label
                plotter.add_point_labels([start + [0,0,120]], [str(row['HoleID'])], font_size=16, text_color="black", always_visible=True, shape_opacity=0.5)

            # C. Assays (Style from drill.py)
            if not df_assays.empty and 'Grade' in df_assays.columns:
                for _, row in df_assays.iterrows():
                    collar = df_collars[df_collars['HoleID'] == row['HoleID']]
                    if not collar.empty:
                        c_row = collar.iloc[0]
                        s = get_coords_at_depth(c_row, row['From'], V_EXAG, CENTER_X, CENTER_Y, avg_elev)
                        e = get_coords_at_depth(c_row, row['To'], V_EXAG, CENTER_X, CENTER_Y, avg_elev)
                        
                        color = get_grade_color(row['Grade'])
                        # Tube slightly thicker than trace (drill.py logic: TUBE_RADIUS * 1.2)
                        plotter.add_mesh(pv.Line(s, e).tube(radius=DEFAULT_TUBE_RADIUS * 1.2), color=color)

            # D. Axes
            plotter.show_grid(xtitle="East", ytitle="North", ztitle="Depth (Scaled)")
            
            # 5. EXPORT
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_html:
                plotter.export_html(tmp_html.name)
                html_path = tmp_html.name

            # 6. DISPLAY & DOWNLOAD
            st.success("Model Generated Successfully!")
            
            # Read HTML content for download button
            with open(html_path, 'r') as f:
                html_content = f.read()

            col_disp, col_dl = st.columns([3, 1])
            with col_disp:
                st.components.v1.html(html_content, height=600, scrolling=False)
            with col_dl:
                st.markdown("### Download Model")
                st.write("Save this 3D model as a standalone HTML file. You can send this to clients or host it on the web.")
                st.download_button(
                    label="💾 Download 3D Model (.html)",
                    data=html_content,
                    file_name="drill_model_3d.html",
                    mime="text/html"
                )

        except Exception as e:
            st.error(f"An error occurred during generation: {e}")
            st.exception(e)

        finally:
            # Cleanup
            safe_remove_file(dem_path)
            safe_remove_file(sat_path)
            safe_remove_file(html_path)