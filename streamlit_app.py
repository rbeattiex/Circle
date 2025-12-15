import streamlit as st
import pandas as pd
import tempfile
import os
import sys
import subprocess
import json
import streamlit.components.v1 as components

# ==========================================
# 0. CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="Drill Hole Visualizer Pro")

# Default Constants
DEFAULT_V_EXAG = 5
DEFAULT_WIDTH_EXAG = 1.0 
DEFAULT_MIN_SCALE = 1.5 # New default for mineralization pop
DEFAULT_BUFFER = 1000

# ==========================================
# 1. THE WORKER SCRIPT (Runs in separate process)
# ==========================================
RENDER_SCRIPT = """
import sys
import os
import pandas as pd
import numpy as np
import pyvista as pv
import rioxarray
from scipy.interpolate import RegularGridInterpolator
import math
import json

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

def get_dynamic_color(grade, rules):
    # Rules must be sorted descending by cutoff
    for rule in rules:
        if grade >= rule['cutoff']:
            return rule['color']
    return "lightgray" # Default for low grade

def run_render(dem_path, sat_path, collars_path, assays_path, html_out_path, v_exag, buffer_sz, target_crs, colors_json_path, show_grid_str, width_exag, min_scale):
    # Parse Arguments
    show_grid = (show_grid_str == "True")
    
    # Base Sizes (Base radius in meters)
    BASE_COLLAR_R = 40.0
    BASE_TRACE_R = 12.0
    
    # Apply General Multiplier
    eff_collar_r = BASE_COLLAR_R * width_exag
    eff_trace_r = BASE_TRACE_R * width_exag
    
    # Apply Mineralization Specific Multiplier (relative to trace)
    eff_assay_r = eff_trace_r * min_scale

    # 1. Load Data
    try:
        df_collars = pd.read_csv(collars_path)
    except pd.errors.EmptyDataError:
        print("Error: Collar CSV is empty")
        return

    df_assays = pd.DataFrame()
    if assays_path and assays_path != 'None':
        try:
            df_assays = pd.read_csv(assays_path)
        except:
            pass
    
    # Load Color Rules
    color_rules = []
    if os.path.exists(colors_json_path):
        with open(colors_json_path, 'r') as f:
            color_rules = json.load(f)
    color_rules.sort(key=lambda x: x['cutoff'], reverse=True)
    
    # 2. Geometry Prep
    if df_collars.empty:
        print("Error: No collar data found")
        return

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
        if pd.isna(row['X']) or pd.isna(row['Y']) or pd.isna(row['Depth']): continue

        z_local = avg_elev
        if z_interpolator:
            try: z_local = z_interpolator((row['Y'], row['X']))
            except: pass
        
        row['Z_rel'] = float((z_local - avg_elev) * v_exag) + 2.0
        
        start = get_coords_at_depth(row, 0, v_exag, CENTER_X, CENTER_Y)
        end = get_coords_at_depth(row, row['Depth'], v_exag, CENTER_X, CENTER_Y)
        
        plotter.add_mesh(pv.Sphere(radius=eff_collar_r, center=start), color="black")
        plotter.add_mesh(pv.Line(start, end).tube(radius=eff_trace_r), color="lightgrey", opacity=0.5)
        plotter.add_point_labels([start + [0,0,120]], [str(row['HoleID'])], font_size=16, text_color="black", always_visible=True, shape_opacity=0.5)

        # Assays
        if not df_assays.empty and 'Grade' in df_assays.columns:
            hole_assays = df_assays[df_assays['HoleID'] == row['HoleID']]
            for _, a_row in hole_assays.iterrows():
                if pd.isna(a_row['From']) or pd.isna(a_row['To']): continue
                
                s = get_coords_at_depth(row, a_row['From'], v_exag, CENTER_X, CENTER_Y)
                e = get_coords_at_depth(row, a_row['To'], v_exag, CENTER_X, CENTER_Y)
                
                c_val = get_dynamic_color(a_row['Grade'], color_rules)
                plotter.add_mesh(pv.Line(s, e).tube(radius=eff_assay_r), color=c_val)
    
    # 5. Add Legend
    legend_entries = []
    for rule in color_rules:
        legend_entries.append((rule['label'], rule['color']))
    
    if legend_entries:
        plotter.add_legend(labels=legend_entries, bcolor='white', size=[0.2, 0.2])

    # 6. Grid
    if show_grid:
        plotter.show_grid(xtitle="East", ytitle="North", ztitle="Depth")

    plotter.export_html(html_out_path)

if __name__ == "__main__":
    # Args: dem, sat, collars, assays, out_html, vexag, buf, crs, colors_json, show_grid, width_exag, min_scale
    run_render(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]), float(sys.argv[7]), sys.argv[8], sys.argv[9], sys.argv[10], float(sys.argv[11]), float(sys.argv[12]))
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

# ==========================================
# 3. EMPTY TEMPLATES
# ==========================================
EMPTY_COLLARS = pd.DataFrame(columns=['HoleID', 'X', 'Y', 'Azimuth', 'Dip', 'Depth'])
EMPTY_ASSAYS = pd.DataFrame(columns=['HoleID', 'From', 'To', 'Grade'])

# ==========================================
# 4. MAIN UI
# ==========================================
st.sidebar.header("1. Config")
V_EXAG = st.sidebar.slider("Vertical Exaggeration", 1, 10, DEFAULT_V_EXAG)
WIDTH_EXAG = st.sidebar.slider("General Hole Width", 0.1, 5.0, DEFAULT_WIDTH_EXAG, 0.1)
MIN_SCALE = st.sidebar.slider("Mineralization Scale (Multiplier)", 1.0, 5.0, DEFAULT_MIN_SCALE, 0.1, help="Makes colored assay intervals thicker than the drill trace.")
BUFFER_SIZE = st.sidebar.number_input("Buffer Size (m)", value=1000)
TARGET_CRS = st.sidebar.text_input("Project CRS (EPSG Code)", value="EPSG:32735")
SHOW_GRID = st.sidebar.checkbox("Show Grid & Axes", value=True)

st.sidebar.markdown("---")
st.sidebar.header("2. Terrain Upload")
dem_file = st.sidebar.file_uploader("Upload DEM (.tif)", type=["tif", "tiff"])
sat_file = st.sidebar.file_uploader("Upload Satellite (.tif)", type=["tif", "tiff"])

st.title("3D Drill Hole Visualizer")
st.caption("Upload your data or enter it manually below.")

tab_collars, tab_assays, tab_colors = st.tabs(["1. Drill Collars", "2. Assay Intervals", "3. Grade Colours"])

with tab_collars:
    collar_file = st.file_uploader("Upload Collars", type=["csv", "xlsx"], key="c_up")
    
    if collar_file:
        if collar_file.name.endswith('.csv'): df_collars_in = pd.read_csv(collar_file)
        else: df_collars_in = pd.read_excel(collar_file)
        c_key = f"c_ed_{collar_file.name}"
    else:
        df_collars_in = EMPTY_COLLARS
        c_key = "c_ed_empty"
        st.info("👋 **No file?** Enter collar data manually below. Required columns: `HoleID`, `X`, `Y`, `Azimuth`, `Dip`, `Depth`")

    edited_collars = st.data_editor(df_collars_in, num_rows="dynamic", key=c_key, use_container_width=True)

with tab_assays:
    assay_file = st.file_uploader("Upload Assays", type=["csv", "xlsx"], key="a_up")
    
    if assay_file:
        if assay_file.name.endswith('.csv'): df_assays_in = pd.read_csv(assay_file)
        else: df_assays_in = pd.read_excel(assay_file)
        a_key = f"a_ed_{assay_file.name}"
    else:
        df_assays_in = EMPTY_ASSAYS
        a_key = "a_ed_empty"
        st.info("👋 **No file?** Enter assay data manually below. Required columns: `HoleID`, `From`, `To`, `Grade`")

    edited_assays = st.data_editor(df_assays_in, num_rows="dynamic", key=a_key, use_container_width=True)

# --- TAB 3: DYNAMIC COLORS ---
with tab_colors:
    st.markdown("#### Grade Thresholds")
    
    if 'grade_rules' not in st.session_state:
        st.session_state.grade_rules = [
            {'cutoff': 0.50, 'color': '#FF0000', 'label': 'High Grade'},
            {'cutoff': 0.30, 'color': '#FFA500', 'label': 'Med-High'},
            {'cutoff': 0.15, 'color': '#FFFF00', 'label': 'Med Grade'},
            {'cutoff': 0.05, 'color': '#008000', 'label': 'Low Grade'}
        ]

    rules_to_remove = []
    
    h1, h2, h3, h4 = st.columns([2, 1, 3, 1])
    h1.markdown("**Grade >=**")
    h2.markdown("**Color**")
    h3.markdown("**Label**")
    
    for i, rule in enumerate(st.session_state.grade_rules):
        c1, c2, c3, c4 = st.columns([2, 1, 3, 1])
        with c1: 
            # Note: We assign logic immediately so values persist
            rule['cutoff'] = st.number_input(f"Cutoff {i}", value=float(rule['cutoff']), key=f"cut_{i}", label_visibility="collapsed")
        with c2: 
            rule['color'] = st.color_picker(f"Color {i}", value=rule['color'], key=f"col_{i}", label_visibility="collapsed")
        with c3: 
            rule['label'] = st.text_input(f"Label {i}", value=rule['label'], key=f"lbl_{i}", label_visibility="collapsed")
        with c4: 
            if st.button("X", key=f"rem_{i}", help="Remove Rule"):
                rules_to_remove.append(i)

    if rules_to_remove:
        for i in reversed(rules_to_remove):
            st.session_state.grade_rules.pop(i)
        st.rerun()

    c_add, c_sort = st.columns([1, 4])
    with c_add:
        if st.button("➕ Add Rule"):
            st.session_state.grade_rules.append({'cutoff': 0.0, 'color': '#808080', 'label': 'New Rule'})
            st.rerun()
    with c_sort:
        # MOVED BUTTON HERE: Processing runs top-to-bottom. 
        # By placing this after the loop, we ensure the latest numbers are captured before sorting.
        if st.button("⬇️ Sort by Grade (High to Low)"):
            st.session_state.grade_rules.sort(key=lambda x: x['cutoff'], reverse=True)
            st.rerun()

# ==========================================
# 5. EXECUTION LOGIC
# ==========================================
if st.button("Generate 3D Model", type="primary"):
    if edited_collars.empty:
        st.error("⚠️ **Missing Data:** Please add at least one Drill Collar before generating the model.")
        st.stop()
        
    with st.spinner("Starting isolated render process..."):
        t_dir = tempfile.gettempdir()
        
        p_collars = os.path.join(t_dir, "temp_collars.csv")
        p_assays = os.path.join(t_dir, "temp_assays.csv")
        p_colors = os.path.join(t_dir, "temp_colors.json")
        p_html = os.path.join(t_dir, "output_model.html")
        p_script = os.path.join(t_dir, "renderer.py")
        
        edited_collars.to_csv(p_collars, index=False)
        edited_assays.to_csv(p_assays, index=False)
        
        with open(p_colors, 'w') as f:
            json.dump(st.session_state.grade_rules, f)
        
        p_dem = "None"
        p_sat = "None"
        if dem_file: p_dem = save_uploaded_file(dem_file)
        if sat_file: p_sat = save_uploaded_file(sat_file)
        
        with open(p_script, "w") as f:
            f.write(RENDER_SCRIPT)
            
        # Updated Command to include MIN_SCALE
        cmd = [
            sys.executable, p_script,
            p_dem, p_sat, p_collars, p_assays, p_html,
            str(V_EXAG), str(BUFFER_SIZE), TARGET_CRS, p_colors,
            str(SHOW_GRID), str(WIDTH_EXAG), str(MIN_SCALE)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(p_html):
                with open(p_html, "r") as f: html_content = f.read()
                st.components.v1.html(html_content, height=600)
                st.download_button("Download HTML", html_content, "model.html")
            else:
                st.error("Render failed in subprocess.")
                st.code(result.stderr)
                
        except Exception as e:
            st.error(f"System Error: {e}")
