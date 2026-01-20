"""
Marine Mammal Proteome Explorer - Streamlit Version
User Flow: Select Animals → Watch Animation → Explore Proteomes
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import base64
import time

# Page configuration
st.set_page_config(
    page_title="Marine Mammal Proteome Explorer",
    page_icon="🐬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ocean-inspired color palette
OCEAN_COLORS = {
    'deep_ocean': '#0a192f',
    'midnight': '#112240',
    'surface': '#1d3557',
    'wave': '#457b9d',
    'seafoam': '#a8dadc',
    'foam': '#f1faee',
    'coral': '#e63946',
    'kelp': '#2d6a4f',
    'bioluminescent': '#64ffda',
    'sunlight': '#ffd166',
}

# Species configuration
SPECIES_DATA = {
    'Sea Lion': {
        'csv': 'umap_output/sealion_umap.csv',
        'image': 'animal_pics/sealion.png',
        'color': '#e63946',
        'glow': 'rgba(230, 57, 70, 0.4)',
        'display_name': 'California Sea Lion',
        'key': 'sealion'
    },
    'Bottlenose Dolphin': {
        'csv': 'umap_output/bottlenose_umap.csv',
        'image': 'animal_pics/bottlenose.png',
        'color': '#64ffda',
        'glow': 'rgba(100, 255, 218, 0.4)',
        'display_name': 'Bottlenose Dolphin',
        'key': 'bottlenose'
    },
    'Gray Whale': {
        'csv': 'umap_output/graywhale_umap.csv',
        'image': 'animal_pics/graywhale.png',
        'color': '#ffd166',
        'glow': 'rgba(255, 209, 102, 0.4)',
        'display_name': 'Gray Whale',
        'key': 'graywhale'
    },
    'Orca': {
        'csv': 'umap_output/orca_umap.csv',
        'image': 'animal_pics/orca.png',
        'color': '#a8dadc',
        'glow': 'rgba(168, 218, 220, 0.4)',
        'display_name': 'Orca (Killer Whale)',
        'key': 'orca'
    },
    'Harbor Seal': {
        'csv': 'umap_output/harborseal_umap.csv',
        'image': 'animal_pics/harbor_seal.png',
        'color': '#06d6a0',
        'glow': 'rgba(6, 214, 160, 0.4)',
        'display_name': 'Harbor Seal',
        'key': 'harborseal'
    }
}

SAMPLE_SIZE = 3000
PRERENDERED_DIR = Path('prerendered_animations')

# Custom CSS for ocean theme
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Main background */
    .stApp {
        background: linear-gradient(180deg, #0a192f 0%, #112240 50%, #1d3557 100%);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #112240;
    }
    ::-webkit-scrollbar-thumb {
        background: #457b9d;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #64ffda;
    }

    /* Glass card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 25px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(100, 255, 218, 0.3);
        transform: translateY(-2px);
    }

    /* Title styling */
    .main-title {
        background: linear-gradient(90deg, #a8dadc, #64ffda, #a8dadc);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 3s linear infinite;
        font-family: 'Inter', sans-serif;
        font-weight: 300;
        font-size: 48px;
        text-align: center;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }

    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }

    .subtitle {
        text-align: center;
        color: #64ffda;
        font-size: 13px;
        margin-bottom: 8px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    .description {
        text-align: center;
        color: #a8dadc;
        font-size: 16px;
        margin-bottom: 40px;
        font-family: 'Inter', sans-serif;
        font-weight: 300;
    }

    /* Species card */
    .species-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        min-height: 200px;
    }

    .species-card.selected {
        background: rgba(255, 255, 255, 0.08);
        transform: translateY(-6px) scale(1.02);
    }

    .species-card img {
        width: 100px;
        height: 100px;
        object-fit: contain;
        margin-bottom: 15px;
    }

    .species-name {
        color: #f1faee;
        font-size: 14px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        margin-bottom: 10px;
    }

    /* Button styling */
    .stButton > button {
        background: rgba(100, 255, 218, 0.1);
        color: #64ffda;
        border: 1px solid #64ffda;
        border-radius: 30px;
        padding: 12px 40px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        letter-spacing: 2px;
        text-transform: uppercase;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        background: #64ffda;
        color: #0a192f;
        box-shadow: 0 0 30px rgba(100, 255, 218, 0.4);
    }

    .stButton > button:disabled {
        background: rgba(100, 100, 100, 0.1);
        color: #457b9d;
        border: 1px solid rgba(100, 100, 100, 0.2);
        cursor: not-allowed;
    }

    /* Selected species display */
    .selected-display {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        max-width: 600px;
        margin: 0 auto;
    }

    .section-title {
        color: #64ffda;
        font-size: 12px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 15px;
    }

    /* Animation container */
    .animation-container {
        text-align: center;
        padding: 40px;
    }

    .animation-container img {
        max-width: 100%;
        max-height: 70vh;
        border-radius: 16px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4), 0 0 40px rgba(100, 255, 218, 0.1);
        border: 1px solid rgba(100, 255, 218, 0.2);
    }

    /* Explorer controls */
    .control-panel {
        background: rgba(10, 25, 47, 0.9);
        backdrop-filter: blur(15px);
        border-radius: 16px;
        border: 1px solid rgba(100, 255, 218, 0.15);
        padding: 25px;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
    }

    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #64ffda, #a8dadc);
    }

    .stSlider > div > div > div > div {
        background: #64ffda;
        box-shadow: 0 0 10px rgba(100, 255, 218, 0.5);
    }

    /* Data table styling */
    .dataframe {
        background: rgba(10, 25, 47, 0.6) !important;
        border-radius: 8px;
    }

    .dataframe th {
        background: rgba(100, 255, 218, 0.1) !important;
        color: #64ffda !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 10px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }

    .dataframe td {
        background: rgba(10, 25, 47, 0.4) !important;
        color: #f1faee !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
    }

    /* Checkbox styling */
    .stCheckbox > label {
        color: #a8dadc !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #64ffda !important;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #64ffda !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    [data-testid="stMetricLabel"] {
        color: #a8dadc !important;
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(10, 25, 47, 0.8) !important;
        border: 1px solid rgba(100, 255, 218, 0.2) !important;
        color: #f1faee !important;
    }

    /* Multiselect styling */
    .stMultiSelect > div > div {
        background: rgba(10, 25, 47, 0.8) !important;
        border: 1px solid rgba(100, 255, 218, 0.2) !important;
    }

    .stMultiSelect span {
        background: rgba(100, 255, 218, 0.2) !important;
        color: #64ffda !important;
    }

    /* Info box */
    .info-box {
        background: rgba(100, 255, 218, 0.1);
        border: 1px solid rgba(100, 255, 218, 0.3);
        border-radius: 12px;
        padding: 15px;
        color: #a8dadc;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
    }

    /* Legend item */
    .legend-item {
        display: flex;
        align-items: center;
        margin: 8px 0;
        color: #f1faee;
        font-family: 'Inter', sans-serif;
        font-size: 13px;
    }

    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'screen' not in st.session_state:
    st.session_state.screen = 'selection'
if 'selected_species' not in st.session_state:
    st.session_state.selected_species = []
if 'loaded_data' not in st.session_state:
    st.session_state.loaded_data = None


def encode_image_to_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    try:
        with open(image_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode()
        ext = Path(image_path).suffix.lower()
        mime_type = {'gif': 'image/gif', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}.get(ext, 'image/png')
        return f"data:{mime_type};base64,{encoded}"
    except:
        return None


def get_animation_path(species_list):
    """Get path to pre-rendered animation"""
    species_keys = sorted([SPECIES_DATA[species]['key'] for species in species_list])
    filename = '_'.join(species_keys) + '.gif'
    prerendered_path = PRERENDERED_DIR / filename
    if prerendered_path.exists():
        return str(prerendered_path), True
    return str(prerendered_path), False


def process_image(fn, num_points):
    """Extract sampled coordinates and colors from animal silhouette."""
    image = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image file: {fn}")

    if image.ndim == 2:
        image_original = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_gray = image.copy()
        alpha = None
    elif image.shape[2] == 4:
        bgr = image[..., :3]
        alpha = image[..., 3]
        image_original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        image_original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        alpha = None

    if alpha is not None and np.std(alpha) > 1:
        mask = (alpha > 20).astype(np.uint8) * 255
    else:
        _, mask = cv2.threshold(image_gray, 40, 255, cv2.THRESH_BINARY)
        if np.mean(image_gray[mask == 255]) < np.mean(image_gray[mask == 0]):
            mask = cv2.bitwise_not(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found")
    largest_contour = max(contours, key=cv2.contourArea)

    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, [largest_contour], -1, 255, cv2.FILLED)

    white_coords = np.column_stack(np.where(filled_mask == 255))
    if white_coords.shape[0] < num_points:
        raise ValueError(f"Not enough pixels ({white_coords.shape[0]} < {num_points})")

    np.random.seed(0)
    sampled_coords = white_coords[np.random.choice(len(white_coords), num_points, replace=False)]
    sampled_colors = image_original[sampled_coords[:, 0], sampled_coords[:, 1]]
    sampled_colors_normalized = sampled_colors / 255.0

    return sampled_coords, sampled_colors_normalized


def place_animals_smart(animals_data, plot_bounds):
    """Position animals in grid layout."""
    n_animals = len(animals_data)
    placed = []

    if n_animals <= 2:
        rows, cols = 1, 2
    elif n_animals <= 4:
        rows, cols = 2, 2
    else:
        rows = int(np.ceil(np.sqrt(n_animals)))
        cols = int(np.ceil(n_animals / rows))

    plot_x_min, plot_x_max, plot_y_min, plot_y_max = plot_bounds
    cell_width = (plot_x_max - plot_x_min) / cols
    cell_height = (plot_y_max - plot_y_min) / rows

    padding = 0.1
    usable_width = cell_width * (1 - 2 * padding)
    usable_height = cell_height * (1 - 2 * padding)

    for idx, animal in enumerate(animals_data):
        row = idx // cols
        col = idx % cols

        cell_center_x = plot_x_min + (col + 0.5) * cell_width
        cell_center_y = plot_y_max - (row + 0.5) * cell_height

        coords = animal["coords"]
        orig_width = coords[:, 1].max() - coords[:, 1].min()
        orig_height = coords[:, 0].max() - coords[:, 0].min()
        aspect_ratio = orig_width / orig_height

        if aspect_ratio > (usable_width / usable_height):
            scale = usable_width / orig_width
        else:
            scale = usable_height / orig_height

        scaled_coords_x = (coords[:, 1] - coords[:, 1].min()) * scale
        scaled_coords_y = (coords[:, 0] - coords[:, 0].min()) * scale

        final_x = scaled_coords_x - scaled_coords_x.mean() + cell_center_x
        final_y = -(scaled_coords_y - scaled_coords_y.mean()) + cell_center_y

        placed.append({
            "name": animal["name"],
            "x": final_x,
            "y": final_y,
            "colors": animal["colors"],
            "df": animal["df"],
            "color": animal["color"]
        })

    return placed


@st.cache_data(show_spinner=False)
def load_data_for_species(species_list):
    """Load proteome and animal data for selected species."""
    PLOT_X_MIN, PLOT_X_MAX = -300, 300
    PLOT_Y_MIN, PLOT_Y_MAX = -300, 300

    animals_data = []

    for species in species_list:
        info = SPECIES_DATA[species]
        csv_path = Path(info['csv'])
        img_path = Path(info['image'])

        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        essential_cols = ['Entry', 'Protein names', 'Gene Names', 'Organism',
                         'UMAP 1', 'UMAP 2', 'Cluster Label', 'Length']
        available_cols = [col for col in essential_cols if col in df.columns]
        df = df[available_cols]

        if len(df) > SAMPLE_SIZE:
            df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

        if img_path.exists():
            try:
                coords, colors = process_image(str(img_path), len(df))
            except Exception:
                continue
        else:
            continue

        animals_data.append({
            "name": species,
            "coords": coords,
            "colors": colors,
            "df": df,
            "color": info['color']
        })

    if not animals_data:
        raise ValueError("No valid animal data could be loaded.")

    plot_bounds = (PLOT_X_MIN, PLOT_X_MAX, PLOT_Y_MIN, PLOT_Y_MAX)
    placed_animals = place_animals_smart(animals_data, plot_bounds)

    # Scale UMAP uniformly
    all_dfs = [p["df"] for p in placed_animals]
    umap_x_min = min(df['UMAP 1'].min() for df in all_dfs)
    umap_x_max = max(df['UMAP 1'].max() for df in all_dfs)
    umap_y_min = min(df['UMAP 2'].min() for df in all_dfs)
    umap_y_max = max(df['UMAP 2'].max() for df in all_dfs)

    umap_center_x = (umap_x_min + umap_x_max) / 2
    umap_center_y = (umap_y_min + umap_y_max) / 2
    umap_max_range = max(umap_x_max - umap_x_min, umap_y_max - umap_y_min)

    scale_factor = (PLOT_X_MAX - PLOT_X_MIN) * 0.8 / umap_max_range

    for p in placed_animals:
        p["df"]['UMAP 1 Scaled'] = (p["df"]['UMAP 1'] - umap_center_x) * scale_factor
        p["df"]['UMAP 2 Scaled'] = (p["df"]['UMAP 2'] - umap_center_y) * scale_factor

    return placed_animals


def create_umap_plot(loaded_data, point_size, opacity):
    """Create the interactive UMAP plot"""
    fig = go.Figure()

    for species_data in loaded_data:
        df = species_data['df']

        hover_text = []
        for _, row in df.iterrows():
            text = f"<b>{species_data['name']}</b><br>"
            text += f"Entry: {row['Entry']}<br>"
            if 'Protein names' in row:
                text += f"Protein: {str(row['Protein names'])[:60]}...<br>"
            if 'Cluster Label' in row:
                text += f"Cluster: {row['Cluster Label']}"
            hover_text.append(text)

        fig.add_trace(go.Scattergl(
            x=df['UMAP 1 Scaled'],
            y=df['UMAP 2 Scaled'],
            mode='markers',
            name=species_data['name'],
            marker=dict(size=point_size, color=species_data['color'],
                       opacity=opacity, line=dict(width=0)),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            customdata=df['Entry'].tolist()
        ))

    margin = 50
    fig.update_layout(
        title=None,
        xaxis=dict(
            range=[-300 - margin, 300 + margin],
            showgrid=False,
            zeroline=False,
            visible=False
        ),
        yaxis=dict(
            range=[-300 - margin, 300 + margin],
            showgrid=False,
            zeroline=False,
            visible=False,
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor='rgba(10, 25, 47, 0.95)',
        paper_bgcolor='rgba(10, 25, 47, 0)',
        hovermode='closest',
        dragmode='lasso',
        clickmode='event+select',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.08,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(10, 25, 47, 0.8)",
            bordercolor='rgba(100, 255, 218, 0.2)',
            borderwidth=1,
            font=dict(size=12, family='Inter, sans-serif', color='#a8dadc')
        ),
        margin=dict(l=20, r=20, t=20, b=50),
        autosize=True,
        hoverlabel=dict(
            bgcolor='rgba(10, 25, 47, 0.95)',
            bordercolor='rgba(100, 255, 218, 0.3)',
            font=dict(family='Inter, sans-serif', size=12, color='#f1faee')
        ),
        height=600
    )

    return fig


def selection_screen():
    """Render the species selection screen"""
    st.markdown('<h1 class="main-title">Marine Mammal Proteome Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Comparative Proteomics Visualization System</p>', unsafe_allow_html=True)
    st.markdown('<p class="description">Select species for multi-dimensional protein space analysis</p>', unsafe_allow_html=True)

    # Species selection grid
    cols = st.columns(5)

    for idx, (species, info) in enumerate(SPECIES_DATA.items()):
        with cols[idx]:
            img_path = Path(info['image'])
            is_selected = species in st.session_state.selected_species

            # Card container
            border_color = info['color'] if is_selected else 'rgba(255, 255, 255, 0.1)'
            bg_color = 'rgba(255, 255, 255, 0.08)' if is_selected else 'rgba(255, 255, 255, 0.03)'

            st.markdown(f"""
            <div style="
                background: {bg_color};
                backdrop-filter: blur(10px);
                border: 2px solid {border_color};
                border-radius: 16px;
                padding: 20px;
                text-align: center;
                min-height: 220px;
                {'box-shadow: 0 8px 32px ' + info['glow'] + ';' if is_selected else ''}
            ">
            """, unsafe_allow_html=True)

            if img_path.exists():
                st.image(str(img_path), width=100)

            st.markdown(f'<p style="color: #f1faee; font-size: 14px; font-family: Inter, sans-serif; font-weight: 500;">{info["display_name"]}</p>', unsafe_allow_html=True)

            btn_label = "Selected ✓" if is_selected else "Select"
            if st.button(btn_label, key=f"btn_{species}", use_container_width=True):
                if species in st.session_state.selected_species:
                    st.session_state.selected_species.remove(species)
                else:
                    st.session_state.selected_species.append(species)
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    # Selected species display
    st.markdown("<br>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<p class="section-title">Selected Organisms</p>', unsafe_allow_html=True)

        if st.session_state.selected_species:
            selected_html = ""
            for species in st.session_state.selected_species:
                color = SPECIES_DATA[species]['color']
                name = SPECIES_DATA[species]['display_name']
                selected_html += f'<span style="color: {color}; font-weight: 500; margin-right: 20px; font-size: 15px; font-family: Inter, sans-serif;">● {name}</span>'
            st.markdown(f'<div style="text-align: center; margin-bottom: 20px;">{selected_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color: #a8dadc; text-align: center; font-size: 15px;">No organisms selected</p>', unsafe_allow_html=True)

        # Analyze button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔬 Analyze Proteomes", disabled=len(st.session_state.selected_species) == 0, use_container_width=True):
                st.session_state.screen = 'animation'
                st.rerun()


def animation_screen():
    """Render the animation screen"""
    st.markdown('<h1 class="main-title">Proteome Space Transformation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">Watch as species silhouettes morph into their UMAP embeddings</p>', unsafe_allow_html=True)

    # Check for animation
    animation_path, exists = get_animation_path(st.session_state.selected_species)

    # Load data with spinner
    with st.spinner("Loading proteome data..."):
        try:
            st.session_state.loaded_data = load_data_for_species(tuple(st.session_state.selected_species))
        except Exception as e:
            st.error(f"Error loading data: {e}")
            if st.button("← Back to Selection"):
                st.session_state.screen = 'selection'
                st.rerun()
            return

    # Display animation if available
    if exists:
        with open(animation_path, 'rb') as f:
            gif_data = f.read()

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(gif_data, use_container_width=True)

        # Auto-transition after showing animation
        time.sleep(3.5)
        st.session_state.screen = 'explorer'
        st.rerun()
    else:
        # No animation, go directly to explorer
        st.info("Animation not available for this combination. Proceeding to explorer...")
        time.sleep(1)
        st.session_state.screen = 'explorer'
        st.rerun()


def explorer_screen():
    """Render the interactive explorer screen"""
    st.markdown('<h1 class="main-title">Proteome Space Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">Interactive UMAP embedding - hover and click to explore proteins</p>', unsafe_allow_html=True)

    # Ensure data is loaded
    if st.session_state.loaded_data is None:
        with st.spinner("Loading data..."):
            st.session_state.loaded_data = load_data_for_species(tuple(st.session_state.selected_species))

    # Layout: main plot + controls sidebar
    col_main, col_sidebar = st.columns([4, 1])

    with col_sidebar:
        st.markdown('<p class="section-title">Controls</p>', unsafe_allow_html=True)

        if st.button("← Back to Selection", use_container_width=True):
            st.session_state.screen = 'selection'
            st.session_state.loaded_data = None
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Visualization controls
        st.markdown('<p class="section-title">Visualization</p>', unsafe_allow_html=True)
        point_size = st.slider("Point Size", 1, 10, 3)
        opacity = st.slider("Opacity", 0.1, 1.0, 0.6, 0.1)

        st.markdown("<br>", unsafe_allow_html=True)

        # Legend
        st.markdown('<p class="section-title">Species Legend</p>', unsafe_allow_html=True)
        for species_data in st.session_state.loaded_data:
            species = species_data['name']
            color = species_data['color']
            display_name = SPECIES_DATA[species]['display_name']
            st.markdown(f"""
            <div class="legend-item">
                <div class="legend-dot" style="background: {color};"></div>
                <span>{display_name}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Stats
        st.markdown('<p class="section-title">Statistics</p>', unsafe_allow_html=True)
        total_proteins = sum(len(d['df']) for d in st.session_state.loaded_data)
        st.metric("Total Proteins", f"{total_proteins:,}")
        st.metric("Species", len(st.session_state.loaded_data))

    with col_main:
        # Create and display the plot
        fig = create_umap_plot(st.session_state.loaded_data, point_size, opacity)

        # Use plotly_chart with selection enabled
        selected_points = st.plotly_chart(
            fig,
            use_container_width=True,
            key="umap_plot",
            on_select="rerun",
            selection_mode=["points", "lasso", "box"]
        )

    # Protein selection table
    st.markdown('<p class="section-title">Selected Proteins</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Click points or use lasso/box select to view protein details</div>', unsafe_allow_html=True)

    # Handle selection
    if selected_points and selected_points.selection and selected_points.selection.points:
        selected_data = []

        for point in selected_points.selection.points:
            curve_idx = point.get('curve_number', 0)
            point_idx = point.get('point_index', 0)

            if curve_idx < len(st.session_state.loaded_data):
                species_data = st.session_state.loaded_data[curve_idx]
                df = species_data['df']

                if point_idx < len(df):
                    row = df.iloc[point_idx]
                    selected_data.append({
                        'Entry': row.get('Entry', 'N/A'),
                        'Organism': SPECIES_DATA[species_data['name']]['display_name'],
                        'Protein Name': str(row.get('Protein names', 'N/A'))[:70],
                        'Gene': str(row.get('Gene Names', 'N/A')),
                        'Cluster': str(row.get('Cluster Label', 'N/A')),
                        'Length': row.get('Length', 'N/A'),
                        'UMAP 1': f"{row.get('UMAP 1', 0):.3f}",
                        'UMAP 2': f"{row.get('UMAP 2', 0):.3f}"
                    })

        if selected_data:
            st.success(f"{len(selected_data)} protein{'s' if len(selected_data) != 1 else ''} selected")
            st.dataframe(
                pd.DataFrame(selected_data),
                use_container_width=True,
                hide_index=True
            )

            # Export button
            csv = pd.DataFrame(selected_data).to_csv(index=False)
            st.download_button(
                "📥 Download Selection as CSV",
                csv,
                "selected_proteins.csv",
                "text/csv",
                use_container_width=False
            )
    else:
        st.info("No proteins selected. Click or drag to select proteins from the plot.")


# Main app routing
def main():
    if st.session_state.screen == 'selection':
        selection_screen()
    elif st.session_state.screen == 'animation':
        animation_screen()
    elif st.session_state.screen == 'explorer':
        explorer_screen()


if __name__ == '__main__':
    main()
