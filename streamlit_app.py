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
from concurrent.futures import ThreadPoolExecutor # NEW: Required for threading fix
import nest_asyncio

# Apply the asyncio patch just in case, though threading does the heavy lifting
nest_asyncio.apply()

# ==========================================
# 0. CONFIGURATION & SETUP
# ==========================================
st.set_page_config(layout="wide", page_title="Drill Hole Visualizer Pro")
pv.set_plot_theme("document")

# Default Style Constants
DEFAULT_TUBE_RADIUS = 12.0
DEFAULT_COLLAR_RADIUS = 40.0
DEFAULT_V_EXAG = 5

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def safe_remove_file(path):
    if not path or not os.path.exists(path): return
    try: os.remove(path)
    except: pass

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- NEW: Threaded Export Wrapper ---
# This runs the export in a separate thread to prevent the "RuntimeError" loop crash
def export_html_threaded(plotter, filename):
    with ThreadPoolExecutor() as executor:
        future = executor.submit(plotter.export_html, filename)
        return future.result()

def get_coords_at_depth(row, depth, v_exag, center_x, center_y, z_datum=0):
    az_rad = math.radians(row['Azimuth'])
    dip_rad = math.radians(row['Dip'])
    dx = depth * math.cos(dip_rad) * math.sin(az_rad)
    dy = depth * math.cos(dip_rad) * math.cos(az_rad)
    x = (row['X'] - center_x) + dx
    y = (row['Y'] - center_y) + dy
    vertical_drop = depth * math.sin(dip_rad)
    z = row['Z_rel'] + (vertical_drop * v_exag)
    return np.array([x, y, z])

def get_grade_color(grade):
    if grade >= 0.50: return "red"
    elif grade >= 0.30: return "darkorange"
    elif grade >= 0.15: return "yellow"
    elif grade >= 0.05: return "green"
    else: return "lightgray"

# ==========================================
# 2. DEFAULT DATA
# ==========================================
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
st.sidebar.header("1. Config")
V_EXAG = st.sidebar.slider("Vertical Exaggeration", 1, 10, DEFAULT_V_EXAG)
BUFFER_SIZE = st.sidebar.number_input("Buffer Size (m)", value=1000)
# Default to UTM Zone 35S (EPSG:32735)
TARGET_CRS = st.sidebar.text_input("Project CRS (EPSG Code)", value="EPSG:32735")

st.sidebar.markdown("---")
st.sidebar.header("2. Terrain Upload")
dem_file = st.sidebar.file_uploader("Upload DEM (.tif)", type=["tif", "tiff"])
sat_file = st.sidebar.file_uploader("Upload Satellite (.tif)", type=["tif", "tiff"])

# ==========================================
# 4. MAIN UI & DATA INPUT
# ==========================================
st.title("3D Drill Hole Visualizer")

tab_collars, tab_assays, tab_info = st.tabs(["1. Drill Collars", "2. Assay Intervals", "3. Legend & Info"])

with tab_collars:
    col1, col2 = st.columns([1, 1])
    with col1:
        collar_file = st.file_uploader("Upload Collars (CSV/Excel)", type=["csv", "xlsx", "xls"], key="collar_up")
    with col2:
        st.download_button("⬇️ Template", data=convert_df_to_csv(pd.DataFrame(columns=['HoleID', 'X', 'Y', 'Azimuth', 'Dip', 'Depth'])), file_name="collar_template.csv", mime="text/csv")
    
    if collar_file:
        try:
            if collar_file.name.endswith('.csv'): df_collars_in = pd.read_csv(collar_file)
            else: df_collars_in = pd.read_excel(collar_file)
        except: df_collars_in = pd.DataFrame(default_collars_dict)
    else:
        df_collars_in = pd.DataFrame(default_collars_dict)
    edited_collars = st.data_editor(df_collars_in, num_rows="dynamic", key="collar_editor", use_container_width=True)

with tab_assays:
    col1, col2 = st.columns([1, 1])
    with col1:
        assay_file = st.file_uploader("Upload Assays (CSV/Excel)", type=["csv", "xlsx", "xls"], key="assay_up")
    with col2:
        st.download_button("⬇️ Template", data=convert_df_to_csv(pd.DataFrame(columns=['HoleID', 'From', 'To', 'Grade'])), file_name="assay_template.csv", mime="text/csv")

    if assay_file:
        try:
            if assay_file.name.endswith('.csv'): df_assays_in = pd.read_csv(assay_file)
            else: df_assays_in = pd.read_excel(assay_file)
        except: df_assays_in = pd.DataFrame(default_assays_dict)
    else:
        df_assays_in = pd.DataFrame(default_assays_dict)
    edited_assays = st.data_editor(df_assays_in, num_rows="dynamic", key="assay_editor", use_container_width=True)

with tab_info:
    st.markdown("#### Grade Legend")
    st.markdown("- <span style='color:red'>■</span> **Red**: >= 0.50 | <span style='color:darkorange'>■</span> **Orange**: >= 0.30 | <span style='color:yellow'>■</span> **Yellow**: >= 0.15", unsafe_allow_html=True)

# ==========================================
# 5. GENERATION LOGIC
# ==========================================
if st.button("Generate 3D Model", type="primary"):
    if edited_collars.empty:
        st.error("Collar data is required.")
        st.stop()
        
    with st.spinner("Processing... (Aligning Maps & Generating 3D Mesh)"):
        # Prep Data
        df_collars = edited_collars.copy()
        df_assays = edited_assays.copy()
        for col in ['X', 'Y', 'Azimuth', 'Dip', 'Depth']:
            if col in df_collars.columns: df_collars[col] = pd.to_numeric(df_collars[col], errors='coerce')
        if not df_assays.empty:
            for col in ['From', 'To', 'Grade']:
                if col in df_assays.columns: df_assays[col] = pd.to_numeric(df_assays[col], errors='coerce')
        
        CENTER_X = df_collars['X'].mean()
        CENTER_Y = df_collars['Y'].mean()
        
        # Resources
        dem_path, sat_path, html_path = None, None, None
        terrain_mesh, terrain_tex = None, None
        z_interpolator = None
        avg_elev = 0.0

        try:
            if dem_file and sat_file:
                dem_path = save_uploaded_file(dem_file)
                sat_path = save_uploaded_file(sat_file)

                # --- 1. LOAD DEM ---
                dem_ds = rioxarray.open_rasterio(dem_path, masked=True).squeeze()
                
                # --- FIX: FORCE WGS84 & REPROJECT ---
                if dem_ds.rio.crs is None:
                     dem_ds.rio.write_crs("EPSG:4326", inplace=True)

                if str(dem_ds.rio.crs) != TARGET_CRS:
                    try:
                        dem_ds = dem_ds.rio.reproject(TARGET_CRS)
                    except Exception as e:
                        st.warning(f"Reprojection warning: {e}. Using original CRS.")

                # --- SAFE CLIPPING ---
                min_x, max_x = df_collars['X'].min() - BUFFER_SIZE, df_collars['X'].max() + BUFFER_SIZE
                min_y, max_y = df_collars['Y'].min() - BUFFER_SIZE, df_collars['Y'].max() + BUFFER_SIZE
                
                try:
                    dem_clip = dem_ds.rio.clip_box(minx=min_x, miny=min_y, maxx=max_x, maxy=max_y)
                except Exception as clip_err:
                    st.warning(f"Clipping warning (might be no overlap?): {clip_err}")
                    dem_clip = dem_ds

                # --- 2. SATELLITE PROCESSING ---
                sat_ds = rioxarray.open_rasterio(sat_path, masked=True)
                if sat_ds.rio.crs is None:
                    sat_ds.rio.write_crs("EPSG:4326", inplace=True)

                if str(sat_ds.rio.crs) != TARGET_CRS:
                    try: sat_ds = sat_ds.rio.reproject(TARGET_CRS)
                    except: pass
                
                try: sat_clip = sat_ds.rio.clip_box(minx=min_x, miny=min_y, maxx=max_x, maxy=max_y)
                except: sat_clip = sat_ds

                # Texture Processing
                sat_data = sat_clip.values.transpose(1, 2, 0)
                if sat_data.dtype != np.uint8:
                    sat_data = ((sat_data - np.nanmin(sat_data)) / (np.nanmax(sat_data) - np.nanmin(sat_data)) * 255).astype(np.uint8)
                terrain_tex = pv.Texture(sat_data)
                
                # Mesh Gen
                x = dem_clip.x.values - CENTER_X
                y = dem_clip.y.values - CENTER_Y
                z = dem_clip.values
                avg_elev = np.nanmean(z)
                z = np.nan_to_num(z, nan=avg_elev)
                
                z_interpolator = RegularGridInterpolator(
                    (dem_clip.y.values, dem_clip.x.values), dem_clip.values, 
                    bounds_error=False, fill_value=avg_elev
                )
                
                grid = pv.RectilinearGrid(x, y, [0])
                grid["Elevation"] = z.flatten()
                grid["Relative_Z"] = (grid["Elevation"] - avg_elev) * V_EXAG
                terrain_mesh = grid.warp_by_scalar("Relative_Z")
                terrain_mesh.texture_map_to_plane(use_bounds=True, inplace=True)
                
                dem_ds.close()
                sat_ds.close()

            # PLOTTING
            plotter = pv.Plotter(window_size=[800, 600])
            plotter.set_background('white')
            
            if terrain_mesh:
                plotter.add_mesh(terrain_mesh, texture=terrain_tex, opacity=1.0)
            else:
                plane = pv.Plane(center=(0,0,0), i_size=2000, j_size=2000)
                plotter.add_mesh(plane, color="gray", opacity=0.5, show_edges=True)
            
            for _, row in df_collars.iterrows():
                z_local = avg_elev
                if z_interpolator:
                    try: z_local = z_interpolator((row['Y'], row['X']))
                    except: pass
                
                row['Z_rel'] = float((z_local - avg_elev) * V_EXAG) + 2.0
                
                start = get_coords_at_depth(row, 0, V_EXAG, CENTER_X, CENTER_Y)
                end = get_coords_at_depth(row, row['Depth'], V_EXAG, CENTER_X, CENTER_Y)
                
                plotter.add_mesh(pv.Sphere(radius=DEFAULT_COLLAR_RADIUS, center=start), color="black")
                plotter.add_mesh(pv.Line(start, end).tube(radius=DEFAULT_TUBE_RADIUS), color="lightgrey", opacity=0.5)
                plotter.add_point_labels([start + [0,0,120]], [str(row['HoleID'])], font_size=16, text_color="black", always_visible=True, shape_opacity=0.5)

                if not df_assays.empty and 'Grade' in df_assays.columns:
                    hole_assays = df_assays[df_assays['HoleID'] == row['HoleID']]
                    for _, a_row in hole_assays.iterrows():
                        s = get_coords_at_depth(row, a_row['From'], V_EXAG, CENTER_X, CENTER_Y)
                        e = get_coords_at_depth(row, a_row['To'], V_EXAG, CENTER_X, CENTER_Y)
                        plotter.add_mesh(pv.Line(s, e).tube(radius=DEFAULT_TUBE_RADIUS * 1.2), color=get_grade_color(a_row['Grade']))

            plotter.show_grid(xtitle="East", ytitle="North", ztitle="Depth")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_html:
                # --- FIX: Run Export in Thread to Avoid Loop Conflict ---
                export_html_threaded(plotter, tmp_html.name)
                html_path = tmp_html.name

            with open(html_path, 'r') as f: html_content = f.read()
            st.components.v1.html(html_content, height=600, scrolling=False)
            st.download_button("💾 Download Model", html_content, "drill_model.html", "text/html")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)
        finally:
            safe_remove_file(dem_path)
            safe_remove_file(sat_path)
            safe_remove_file(html_path)
