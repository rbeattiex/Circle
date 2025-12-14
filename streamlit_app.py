import streamlit as st
import pandas as pd
import tempfile
import os
import sys
import subprocess
import streamlit.components.v1 as components

# ==========================================
# 0. CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="Drill Hole Visualizer Pro")

# Default Constants
DEFAULT_V_EXAG = 5
DEFAULT_BUFFER = 1000

# ==========================================
# 1. THE WORKER SCRIPT (Runs in separate process)
# ==========================================
# This script handles all 3D logic in isolation to prevent Async/Loop crashes.
RENDER_SCRIPT = """
import sys
import os
import pandas as pd
import numpy as np
import pyvista as pv
import rioxarray
from scipy.interpolate import RegularGridInterpolator
import math

# Force off-screen rendering
pv.set_plot_theme("document")

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

def run_render(dem_path, sat_path, collars_path, assays_path, html_out_path, v_exag, buffer_sz, target_crs):
    # 1. Load Data
    df_collars = pd.read_csv(collars_path)
    df_assays = pd.DataFrame()
    if assays_path and assays_path != 'None':
        df_assays = pd.read_csv(assays_path)
    
    # 2. Geometry Prep
    CENTER_X = df_collars['X'].mean()
    CENTER_Y = df_collars['Y'].mean()
    
    terrain_mesh = None
    terrain_tex = None
    z_interpolator = None
    avg_elev = 0.0

    # 3. Terrain Processing
    if dem_path != 'None' and sat_path != 'None':
        try:
            # DEM
            dem_ds = rioxarray.open_rasterio(dem_path, masked=True).squeeze()
            if dem_ds.rio.crs is None: dem_ds.rio.write_crs("EPSG:4326", inplace=True)
            if str(dem_ds.rio.crs) != target_crs:
                try: dem_ds = dem_ds.rio.reproject(target_crs)
                except: pass

            # Clip DEM
            min_x, max_x = df_collars['X'].min() - buffer_sz, df_collars['X'].max() + buffer_sz
            min_y, max_y = df_collars['Y'].min() - buffer_sz, df_collars['Y'].max() + buffer_sz
            
            try: dem_clip = dem_ds.rio.clip_box(minx=min_x, miny=min_y, maxx=max_x, maxy=max_y)
            except: dem_clip = dem_ds

            # SAT
            sat_ds = rioxarray.open_rasterio(sat_path, masked=True)
            if sat_ds.rio.crs is None: sat_ds.rio.write_crs("EPSG:4326", inplace=True)
            if str(sat_ds.rio.crs) != target_crs:
                try: sat_ds = sat_ds.rio.reproject(target_crs)
                except: pass
            
            try: sat_clip = sat_ds.rio.clip_box(minx=min_x, miny=min_y, maxx=max_x, maxy=max_y)
            except: sat_clip = sat_ds

            # Texture
            sat_data = sat_clip.values.transpose(1, 2, 0)
            if sat_data.dtype != np.uint8:
                sat_data = ((sat_data - np.nanmin(sat_data)) / (np.nanmax(sat_data) - np.nanmin(sat_data)) * 255).astype(np.uint8)
            terrain_tex = pv.Texture(sat_data)

            # Mesh
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
            grid["Relative_Z"] = (grid["Elevation"] - avg_elev) * v_exag
            terrain_mesh = grid.warp_by_scalar("Relative_Z")
            terrain_mesh.texture_map_to_plane(use_bounds=True, inplace=True)
            
            dem_ds.close()
            sat_ds.close()

        except Exception as e:
            print(f"Terrain Error: {e}")
            pass

    # 4. Plotting
    plotter = pv.Plotter(window_size=[800, 600], off_screen=True)
    plotter.set_background('white')

    if terrain_mesh:
        plotter.add_mesh(terrain_mesh, texture=terrain_tex, opacity=1.0)
    else:
        plane = pv.Plane(center=(0,0,0), i_size=2000, j_size=2000)
        plotter.add_mesh(plane, color="gray", opacity=0.5, show_edges=True)

    # Traces
    for _, row in df_collars.iterrows():
        z_local = avg_elev
        if z_interpolator:
            try: z_local = z_interpolator((row['Y'], row['X']))
            except: pass
        
        row['Z_rel'] = float((z_local - avg_elev) * v_exag) + 2.0
        
        start = get_coords_at_depth(row, 0, v_exag, CENTER_X, CENTER_Y)
        end = get_coords_at_depth(row, row['Depth'], v_exag, CENTER_X, CENTER_Y)
        
        plotter.add_mesh(pv.Sphere(radius=40, center=start), color="black")
        plotter.add_mesh(pv.Line(start, end).tube(radius=12), color="lightgrey", opacity=0.5)
        plotter.add_point_labels([start + [0,0,120]], [str(row['HoleID'])], font_size=16, text_color="black", always_visible=True, shape_opacity=0.5)

        # Assays
        if not df_assays.empty and 'Grade' in df_assays.columns:
            hole_assays = df_assays[df_assays['HoleID'] == row['HoleID']]
            for _, a_row in hole_assays.iterrows():
                s = get_coords_at_depth(row, a_row['From'], v_exag, CENTER_X, CENTER_Y)
                e = get_coords_at_depth(row, a_row['To'], v_exag, CENTER_X, CENTER_Y)
                plotter.add_mesh(pv.Line(s, e).tube(radius=14), color=get_grade_color(a_row['Grade']))

    plotter.show_grid(xtitle="East", ytitle="North", ztitle="Depth")
    plotter.export_html(html_out_path)

if __name__ == "__main__":
    # Args: dem, sat, collars, assays, out_html, vexag, buf, crs
    run_render(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]), float(sys.argv[7]), sys.argv[8])
"""

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e: return None

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# ==========================================
# 3. DEFAULT DATA
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
# 4. MAIN UI
# ==========================================
st.sidebar.header("1. Config")
V_EXAG = st.sidebar.slider("Vertical Exaggeration", 1, 10, DEFAULT_V_EXAG)
BUFFER_SIZE = st.sidebar.number_input("Buffer Size (m)", value=1000)
TARGET_CRS = st.sidebar.text_input("Project CRS (EPSG Code)", value="EPSG:32735")

st.sidebar.markdown("---")
st.sidebar.header("2. Terrain Upload")
dem_file = st.sidebar.file_uploader("Upload DEM (.tif)", type=["tif", "tiff"])
sat_file = st.sidebar.file_uploader("Upload Satellite (.tif)", type=["tif", "tiff"])

st.title("3D Drill Hole Visualizer (Subprocess Mode)")

tab_collars, tab_assays, tab_info = st.tabs(["1. Drill Collars", "2. Assay Intervals", "3. Legend"])

with tab_collars:
    collar_file = st.file_uploader("Upload Collars", type=["csv", "xlsx"], key="c_up")
    df_collars_in = pd.read_csv(collar_file) if collar_file else pd.DataFrame(default_collars_dict)
    edited_collars = st.data_editor(df_collars_in, num_rows="dynamic", key="c_ed", use_container_width=True)

with tab_assays:
    assay_file = st.file_uploader("Upload Assays", type=["csv", "xlsx"], key="a_up")
    df_assays_in = pd.read_csv(assay_file) if assay_file else pd.DataFrame(default_assays_dict)
    edited_assays = st.data_editor(df_assays_in, num_rows="dynamic", key="a_ed", use_container_width=True)

with tab_info:
    st.markdown("- <span style='color:red'>■</span> **Red**: >= 0.50 | <span style='color:darkorange'>■</span> **Orange**: >= 0.30", unsafe_allow_html=True)

# ==========================================
# 5. EXECUTION LOGIC
# ==========================================
if st.button("Generate 3D Model", type="primary"):
    if edited_collars.empty:
        st.error("Collar data required.")
        st.stop()
        
    with st.spinner("Starting isolated render process..."):
        # 1. Save all inputs to temp files
        t_dir = tempfile.gettempdir()
        
        # Save Tables
        p_collars = os.path.join(t_dir, "temp_collars.csv")
        p_assays = os.path.join(t_dir, "temp_assays.csv")
        p_html = os.path.join(t_dir, "output_model.html")
        p_script = os.path.join(t_dir, "renderer.py")
        
        edited_collars.to_csv(p_collars, index=False)
        edited_assays.to_csv(p_assays, index=False)
        
        # Save Maps
        p_dem = "None"
        p_sat = "None"
        if dem_file: p_dem = save_uploaded_file(dem_file)
        if sat_file: p_sat = save_uploaded_file(sat_file)
        
        # 2. Write the Worker Script
        with open(p_script, "w") as f:
            f.write(RENDER_SCRIPT)
            
        # 3. Execute Worker Script
        # Command: python renderer.py [dem] [sat] [collars] [assays] [out] [vexag] [buf] [crs]
        cmd = [
            sys.executable, p_script,
            p_dem, p_sat, p_collars, p_assays, p_html,
            str(V_EXAG), str(BUFFER_SIZE), TARGET_CRS
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(p_html):
                with open(p_html, "r") as f: html_content = f.read()
                st.components.v1.html(html_content, height=600)
                st.download_button("Download HTML", html_content, "model.html")
            else:
                st.error("Render failed in subprocess.")
                st.code(result.stderr) # Show the error from the worker
                
        except Exception as e:
            st.error(f"System Error: {e}")
