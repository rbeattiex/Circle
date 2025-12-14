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

# Set page layout to wide
st.set_page_config(layout="wide", page_title="Drill Hole Visualizer")

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def save_uploaded_file(uploaded_file):
    """Helper to save uploaded file to a temp path."""
    try:
        # Delete=False is CRITICAL for Windows
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def safe_remove_file(path):
    """
    Robust file remover that tolerates Windows file locking.
    """
    if not path or not os.path.exists(path):
        return
    
    try:
        os.remove(path)
    except Exception:
        # If Windows locks the file, we try waiting 1 second
        try:
            time.sleep(1.0)
            os.remove(path)
        except Exception:
            # If it's STILL locked, we just leave it alone.
            # This prevents the app from crashing.
            print(f"Windows locked file {path}. Skipping deletion.")
            pass

def load_data_file(uploaded_file):
    """Helper to load CSV or Excel files."""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def get_coords_at_depth(row, depth, v_exag, center_x, center_y):
    """Calculates 3D coordinates."""
    az_rad = math.radians(row['Azimuth'])
    dip_rad = math.radians(row['Dip'])
    
    dx = depth * math.cos(dip_rad) * math.sin(az_rad)
    dy = depth * math.cos(dip_rad) * math.cos(az_rad)
    
    x = (row['X'] - center_x) + dx
    y = (row['Y'] - center_y) + dy
    
    vertical_drop = depth * math.sin(dip_rad)
    z = row['Z_scaled'] + (vertical_drop * v_exag)
    
    return np.array([x, y, z])

def get_color_for_grade(grade, rules):
    sorted_rules = sorted(rules, key=lambda x: x['cutoff'], reverse=True)
    for rule in sorted_rules:
        if grade >= rule['cutoff']:
            return rule['color']
    return "lightgray"

# ==========================================
# 2. SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.header("1. Project Config")
V_EXAG = st.sidebar.slider("Vertical Exaggeration", 1, 10, 5)
BUFFER_SIZE = st.sidebar.number_input("Buffer Size (m)", value=1000)

st.sidebar.header("2. Upload Maps")
dem_file = st.sidebar.file_uploader("Upload DEM (.tif)", type=["tif", "tiff"])
sat_file = st.sidebar.file_uploader("Upload Satellite (.tif)", type=["tif", "tiff"])

# ==========================================
# 3. DATA INPUT UI
# ==========================================
st.title("3D Drill Hole Visualizer")
st.markdown("### Step 1: Upload Data")

# Create tabs
tab_collars, tab_assays, tab_colors = st.tabs(["1. Drill Collars", "2. Assay Intervals", "3. Grade Colors"])

# --- TAB 1: COLLARS ---
with tab_collars:
    st.markdown("#### Upload Collars")
    st.info("Required Columns: `HoleID`, `X`, `Y`, `Azimuth`, `Dip`, `Depth`")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        # 1. FILE UPLOADER
        collar_file = st.file_uploader("Drag & Drop CSV/Excel here", type=["csv", "xlsx", "xls"], key="collar_up")
    with col2:
        # 2. TEMPLATE DOWNLOAD
        template_collars = pd.DataFrame(columns=['HoleID', 'X', 'Y', 'Azimuth', 'Dip', 'Depth'])
        st.download_button(
            label="⬇️ Download Collar Template",
            data=convert_df_to_csv(template_collars),
            file_name="collar_template.csv",
            mime="text/csv",
        )
    
    # 3. DATA EDITOR
    default_collars = pd.DataFrame({
        'HoleID': ['DH-001'], 'X': [440375.0], 'Y': [8627810.0],
        'Azimuth': [90.0], 'Dip': [-60.0], 'Depth': [200.0]
    })

    if collar_file:
        loaded_df = load_data_file(collar_file)
        if loaded_df is not None:
            default_collars = loaded_df
            
    edited_collars = st.data_editor(default_collars, num_rows="dynamic", key="collar_editor", use_container_width=True)

# --- TAB 2: ASSAYS ---
with tab_assays:
    st.markdown("#### Upload Assays")
    st.info("Required Columns: `HoleID`, `From`, `To`, `Grade`")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        # 1. FILE UPLOADER
        assay_file = st.file_uploader("Drag & Drop CSV/Excel here", type=["csv", "xlsx", "xls"], key="assay_up")
    with col2:
        # 2. TEMPLATE DOWNLOAD
        template_assays = pd.DataFrame(columns=['HoleID', 'From', 'To', 'Grade'])
        st.download_button(
            label="⬇️ Download Assay Template",
            data=convert_df_to_csv(template_assays),
            file_name="assay_template.csv",
            mime="text/csv",
        )

    # 3. DATA EDITOR
    default_assays = pd.DataFrame({
        'HoleID': ['DH-001', 'DH-001'], 'From': [10.0, 50.0],
        'To': [20.0, 60.0], 'Grade': [0.5, 1.2]
    })

    if assay_file:
        loaded_df = load_data_file(assay_file)
        if loaded_df is not None:
            default_assays = loaded_df

    edited_assays = st.data_editor(default_assays, num_rows="dynamic", key="assay_editor", use_container_width=True)

# --- TAB 3: COLORS ---
with tab_colors:
    st.markdown("#### Grade Thresholds")
    if 'color_rules' not in st.session_state:
        st.session_state.color_rules = [
            {'cutoff': 1.0, 'color': '#FF0000', 'label': 'High Grade (>1.0)'},
            {'cutoff': 0.5, 'color': '#FFA500', 'label': 'Med Grade (>0.5)'},
            {'cutoff': 0.2, 'color': '#FFFF00', 'label': 'Low Grade (>0.2)'}
        ]
    
    rules_to_remove = []
    for i, rule in enumerate(st.session_state.color_rules):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1: rule['cutoff'] = st.number_input(f"Grade >=", value=rule['cutoff'], key=f"cut_{i}")
        with c2: rule['color'] = st.color_picker(f"Color", value=rule['color'], key=f"col_{i}")
        with c3: rule['label'] = st.text_input("Label", value=rule['label'], key=f"lbl_{i}")
        with c4: 
            if st.button("Remove", key=f"rem_{i}"): rules_to_remove.append(i)
    
    if rules_to_remove:
        for i in reversed(rules_to_remove): st.session_state.color_rules.pop(i)
        st.rerun()

    if st.button("Add New Grade Threshold"):
        st.session_state.color_rules.append({'cutoff': 0.0, 'color': '#FFFFFF', 'label': 'New'})
        st.rerun()

# ==========================================
# 4. PROCESSING
# ==========================================
st.markdown("### Step 2: Generate Model")
if st.button("Generate 3D Model", type="primary"):
    
    # --- VALIDATION ---
    if edited_collars.empty:
        st.error("Please add at least one drill collar.")
        st.stop()

    with st.spinner("Processing terrain and generating 3D model..."):
        
        # Prepare Data
        df_collars = edited_collars.copy()
        df_assays = edited_assays.copy()
        
        # Numeric cleanup
        req_cols = ['X', 'Y', 'Azimuth', 'Dip', 'Depth']
        for col in req_cols:
            if col in df_collars.columns:
                df_collars[col] = pd.to_numeric(df_collars[col], errors='coerce')
        
        if not df_assays.empty:
            for col in ['From', 'To', 'Grade']:
                if col in df_assays.columns:
                    df_assays[col] = pd.to_numeric(df_assays[col], errors='coerce')
        
        df_collars.dropna(subset=['X', 'Y', 'Depth'], inplace=True)
        
        if df_collars.empty:
            st.error("No valid collar data found. Check your columns and data types.")
            st.stop()

        CENTER_X = df_collars['X'].mean()
        CENTER_Y = df_collars['Y'].mean()

        # Variables for cleanup
        dem_path, sat_path = None, None
        html_path = None
        dem_ds, sat_ds = None, None # File handles

        terrain_mesh, terrain_tex = None, None
        z_interpolator, avg_elev = None, 0

        try:
            # --- MAP PROCESSING ---
            if dem_file and sat_file:
                dem_path = save_uploaded_file(dem_file)
                sat_path = save_uploaded_file(sat_file)

                # Open with rioxarray
                dem_ds = rioxarray.open_rasterio(dem_path, masked=True)
                dem_squeezed = dem_ds.squeeze()
                
                min_x, max_x = df_collars['X'].min() - BUFFER_SIZE, df_collars['X'].max() + BUFFER_SIZE
                min_y, max_y = df_collars['Y'].min() - BUFFER_SIZE, df_collars['Y'].max() + BUFFER_SIZE
                
                dem_clip = dem_squeezed.rio.clip_box(minx=min_x, miny=min_y, maxx=max_x, maxy=max_y)
                
                sat_ds = rioxarray.open_rasterio(sat_path, masked=True)
                sat_clip = sat_ds.rio.clip_box(minx=min_x, miny=min_y, maxx=max_x, maxy=max_y)

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
                grid["Relative_Z"] = (grid["Elevation"] - avg_elev) * V_EXAG
                terrain_mesh = grid.warp_by_scalar("Relative_Z")
                terrain_mesh.texture_map_to_plane(use_bounds=True, inplace=True)
                
                # Close handles explicitly
                dem_ds.close()
                sat_ds.close()
                dem_ds, sat_ds = None, None

            # --- DRILL TRACE GENERATION ---
            df_collars['Z_scaled'] = 0.0
            for i, row in df_collars.iterrows():
                real_z = z_interpolator((row['Y'], row['X'])) if z_interpolator else 0
                if np.isnan(real_z): real_z = avg_elev
                df_collars.at[i, 'Z_scaled'] = float((real_z - avg_elev) * V_EXAG) + 1.0

            # --- PLOTTING ---
            plotter = pv.Plotter(window_size=[800, 600])
            plotter.set_background('white')

            if terrain_mesh:
                plotter.add_mesh(terrain_mesh, texture=terrain_tex, opacity=1.0)
            else:
                plane = pv.Plane(center=(0,0,0), i_size=2000, j_size=2000)
                plotter.add_mesh(plane, color="gray", opacity=0.5, show_edges=True)

            for _, row in df_collars.iterrows():
                start = get_coords_at_depth(row, 0, V_EXAG, CENTER_X, CENTER_Y)
                end = get_coords_at_depth(row, row['Depth'], V_EXAG, CENTER_X, CENTER_Y)
                plotter.add_mesh(pv.Sphere(radius=5, center=start), color="black")
                plotter.add_mesh(pv.Line(start, end).tube(radius=1.0), color="lightgrey", opacity=0.5)
                plotter.add_point_labels([start + [0,0,20]], [str(row['HoleID'])], font_size=14, text_color="black", always_visible=True)

            if not df_assays.empty and 'Grade' in df_assays.columns:
                for _, row in df_assays.iterrows():
                    collar = df_collars[df_collars['HoleID'] == row['HoleID']]
                    if not collar.empty:
                        c_row = collar.iloc[0]
                        s = get_coords_at_depth(c_row, row['From'], V_EXAG, CENTER_X, CENTER_Y)
                        e = get_coords_at_depth(c_row, row['To'], V_EXAG, CENTER_X, CENTER_Y)
                        plotter.add_mesh(pv.Line(s, e).tube(radius=2.5), color=get_color_for_grade(row['Grade'], st.session_state.color_rules))

            plotter.show_grid(xtitle="East", ytitle="North", ztitle="Depth")
            
            # --- EXPORT & DISPLAY ---
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_html:
                plotter.export_html(tmp_html.name)
                html_path = tmp_html.name

            with open(html_path, 'r') as f:
                html_content = f.read()
                st.components.v1.html(html_content, height=600, scrolling=False)

        except Exception as e:
            st.error(f"An error occurred: {e}")

        finally:
            # --- SAFE CLEANUP ---
            if dem_ds: dem_ds.close()
            if sat_ds: sat_ds.close()
            
            # Use the robust remover instead of os.remove
            safe_remove_file(dem_path)
            safe_remove_file(sat_path)
            safe_remove_file(html_path)