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

# Species configuration
SPECIES_DATA = {
    'Sea Lion': {
        'csv': 'umap_output/sealion_umap.csv',
        'image': 'animal_pics/sealion.png',
        'color': '#e63946',
        'display_name': 'California Sea Lion',
        'key': 'sealion'
    },
    'Bottlenose Dolphin': {
        'csv': 'umap_output/bottlenose_umap.csv',
        'image': 'animal_pics/bottlenose.png',
        'color': '#64ffda',
        'display_name': 'Bottlenose Dolphin',
        'key': 'bottlenose'
    },
    'Gray Whale': {
        'csv': 'umap_output/graywhale_umap.csv',
        'image': 'animal_pics/graywhale.png',
        'color': '#ffd166',
        'display_name': 'Gray Whale',
        'key': 'graywhale'
    },
    'Orca': {
        'csv': 'umap_output/orca_umap.csv',
        'image': 'animal_pics/orca.png',
        'color': '#a8dadc',
        'display_name': 'Orca (Killer Whale)',
        'key': 'orca'
    },
    'Harbor Seal': {
        'csv': 'umap_output/harborseal_umap.csv',
        'image': 'animal_pics/harbor_seal.png',
        'color': '#06d6a0',
        'display_name': 'Harbor Seal',
        'key': 'harborseal'
    }
}

SAMPLE_SIZE = 3000
PRERENDERED_DIR = Path('prerendered_animations')

# Custom CSS for ocean theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {
        background: linear-gradient(180deg, #0a192f 0%, #112240 50%, #1d3557 100%);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

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
        font-size: 42px;
        text-align: center;
        letter-spacing: 2px;
        margin-bottom: 5px;
    }

    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }

    .subtitle {
        text-align: center;
        color: #64ffda;
        font-size: 12px;
        margin-bottom: 5px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    .description {
        text-align: center;
        color: #a8dadc;
        font-size: 15px;
        margin-bottom: 30px;
        font-family: 'Inter', sans-serif;
        font-weight: 300;
    }

    .section-title {
        color: #64ffda;
        font-size: 12px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 15px;
    }

    /* Button styling */
    .stButton > button {
        background: rgba(100, 255, 218, 0.1) !important;
        color: #64ffda !important;
        border: 1px solid #64ffda !important;
        border-radius: 25px !important;
        padding: 8px 20px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        background: #64ffda !important;
        color: #0a192f !important;
    }

    .stButton > button:disabled {
        background: rgba(100, 100, 100, 0.1) !important;
        color: #457b9d !important;
        border: 1px solid rgba(100, 100, 100, 0.2) !important;
    }

    /* Slider styling */
    .stSlider > div > div > div > div {
        background: #64ffda !important;
    }

    /* Checkbox styling */
    .stCheckbox > label {
        color: #f1faee !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stCheckbox > label > span {
        color: #f1faee !important;
    }

    /* Column container for cards */
    [data-testid="column"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }

    [data-testid="column"]:hover {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(100, 255, 218, 0.3);
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #64ffda !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    [data-testid="stMetricLabel"] {
        color: #a8dadc !important;
    }

    /* Dataframe styling */
    .stDataFrame {
        background: rgba(10, 25, 47, 0.6) !important;
    }

    /* Info/success/error boxes */
    .stAlert {
        background: rgba(100, 255, 218, 0.1) !important;
        border: 1px solid rgba(100, 255, 218, 0.3) !important;
        color: #a8dadc !important;
    }

    /* Scrollbar */
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'screen' not in st.session_state:
    st.session_state.screen = 'selection'
if 'selected_species' not in st.session_state:
    st.session_state.selected_species = set()
if 'loaded_data' not in st.session_state:
    st.session_state.loaded_data = None


def get_animation_path(species_list):
    """Get path to pre-rendered animation"""
    species_keys = sorted([SPECIES_DATA[species]['key'] for species in species_list])
    filename = '_'.join(species_keys) + '.gif'
    prerendered_path = PRERENDERED_DIR / filename
    if prerendered_path.exists():
        return str(prerendered_path), True
    return str(prerendered_path), False


def process_image(fn, num_points):
    """Extract sampled coordinates from animal silhouette."""
    image = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image file: {fn}")

    if image.ndim == 2:
        image_gray = image.copy()
        alpha = None
    elif image.shape[2] == 4:
        bgr = image[..., :3]
        alpha = image[..., 3]
        image_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
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

    return sampled_coords


@st.cache_data(show_spinner=False)
def load_data_for_species(species_tuple):
    """Load proteome data for selected species."""
    species_list = list(species_tuple)
    PLOT_X_MIN, PLOT_X_MAX = -300, 300
    PLOT_Y_MIN, PLOT_Y_MAX = -300, 300

    loaded_data = []

    for species in species_list:
        info = SPECIES_DATA[species]
        csv_path = Path(info['csv'])

        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        essential_cols = ['Entry', 'Protein names', 'Gene Names', 'Organism',
                         'UMAP 1', 'UMAP 2', 'Cluster Label', 'Length']
        available_cols = [col for col in essential_cols if col in df.columns]
        df = df[available_cols]

        if len(df) > SAMPLE_SIZE:
            df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

        loaded_data.append({
            "name": species,
            "df": df,
            "color": info['color']
        })

    if not loaded_data:
        raise ValueError("No valid data could be loaded.")

    # Scale UMAP uniformly
    all_dfs = [d["df"] for d in loaded_data]
    umap_x_min = min(df['UMAP 1'].min() for df in all_dfs)
    umap_x_max = max(df['UMAP 1'].max() for df in all_dfs)
    umap_y_min = min(df['UMAP 2'].min() for df in all_dfs)
    umap_y_max = max(df['UMAP 2'].max() for df in all_dfs)

    umap_center_x = (umap_x_min + umap_x_max) / 2
    umap_center_y = (umap_y_min + umap_y_max) / 2
    umap_max_range = max(umap_x_max - umap_x_min, umap_y_max - umap_y_min)

    scale_factor = (PLOT_X_MAX - PLOT_X_MIN) * 0.8 / umap_max_range

    for d in loaded_data:
        d["df"]['UMAP 1 Scaled'] = (d["df"]['UMAP 1'] - umap_center_x) * scale_factor
        d["df"]['UMAP 2 Scaled'] = (d["df"]['UMAP 2'] - umap_center_y) * scale_factor

    return loaded_data


def create_umap_plot(loaded_data, point_size, opacity):
    """Create the interactive UMAP plot"""
    fig = go.Figure()

    for species_data in loaded_data:
        df = species_data['df']

        hover_text = []
        for _, row in df.iterrows():
            text = f"<b>{species_data['name']}</b><br>"
            text += f"Entry: {row['Entry']}<br>"
            if 'Protein names' in row and pd.notna(row['Protein names']):
                pname = str(row['Protein names'])[:60]
                text += f"Protein: {pname}...<br>"
            if 'Cluster Label' in row and pd.notna(row['Cluster Label']):
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
            y=-0.12,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(10, 25, 47, 0.8)",
            bordercolor='rgba(100, 255, 218, 0.2)',
            borderwidth=1,
            font=dict(size=12, family='Inter, sans-serif', color='#a8dadc')
        ),
        margin=dict(l=20, r=20, t=20, b=60),
        autosize=True,
        hoverlabel=dict(
            bgcolor='rgba(10, 25, 47, 0.95)',
            bordercolor='rgba(100, 255, 218, 0.3)',
            font=dict(family='Inter, sans-serif', size=12, color='#f1faee')
        ),
        height=550
    )

    return fig


def selection_screen():
    """Render the species selection screen"""
    st.markdown('<h1 class="main-title">Marine Mammal Proteome Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Comparative Proteomics Visualization System</p>', unsafe_allow_html=True)
    st.markdown('<p class="description">Select species for multi-dimensional protein space analysis</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Species selection with checkboxes in columns
    cols = st.columns(5)

    for idx, (species, info) in enumerate(SPECIES_DATA.items()):
        with cols[idx]:
            # Display image
            img_path = Path(info['image'])
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.write("🐋")

            # Species name
            st.markdown(f"<p style='text-align: center; color: {info['color']}; font-weight: 600; font-size: 13px;'>{info['display_name']}</p>", unsafe_allow_html=True)

            # Checkbox for selection
            is_selected = st.checkbox(
                "Select",
                key=f"check_{species}",
                value=species in st.session_state.selected_species
            )

            if is_selected:
                st.session_state.selected_species.add(species)
            elif species in st.session_state.selected_species:
                st.session_state.selected_species.discard(species)

    st.markdown("---")

    # Selected species display
    st.markdown('<p class="section-title">Selected Organisms</p>', unsafe_allow_html=True)

    if st.session_state.selected_species:
        selected_text = " • ".join([
            f"<span style='color: {SPECIES_DATA[s]['color']};'>{SPECIES_DATA[s]['display_name']}</span>"
            for s in st.session_state.selected_species
        ])
        st.markdown(f"<p style='text-align: center; font-size: 15px;'>{selected_text}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='text-align: center; color: #a8dadc;'>No organisms selected</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Analyze button centered
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔬 Analyze Proteomes", disabled=len(st.session_state.selected_species) == 0, use_container_width=True):
            st.session_state.screen = 'animation'
            st.rerun()


def animation_screen():
    """Render the animation screen"""
    st.markdown('<h1 class="main-title">Proteome Space Transformation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">Morphing species silhouettes into UMAP protein embeddings</p>', unsafe_allow_html=True)

    species_list = list(st.session_state.selected_species)

    # Check for animation
    animation_path, exists = get_animation_path(species_list)

    # Show loading while preparing
    with st.spinner("Loading proteome data..."):
        try:
            st.session_state.loaded_data = load_data_for_species(tuple(sorted(species_list)))
        except Exception as e:
            st.error(f"Error loading data: {e}")
            if st.button("← Back to Selection"):
                st.session_state.screen = 'selection'
                st.rerun()
            return

    # Display animation if available
    if exists:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(animation_path, use_container_width=True)
            st.markdown("<p style='text-align: center; color: #64ffda; font-size: 12px;'>Animation complete! Transitioning to explorer...</p>", unsafe_allow_html=True)

        # Add a button to proceed manually or auto-transition
        time.sleep(3)
        st.session_state.screen = 'explorer'
        st.rerun()
    else:
        st.info("Animation not available for this combination. Loading explorer...")
        time.sleep(1)
        st.session_state.screen = 'explorer'
        st.rerun()


def explorer_screen():
    """Render the interactive explorer screen"""
    st.markdown('<h1 class="main-title">Proteome Space Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">Interactive UMAP embedding - hover over points to explore proteins</p>', unsafe_allow_html=True)

    # Ensure data is loaded
    if st.session_state.loaded_data is None:
        species_list = list(st.session_state.selected_species)
        with st.spinner("Loading data..."):
            st.session_state.loaded_data = load_data_for_species(tuple(sorted(species_list)))

    # Sidebar for controls
    with st.sidebar:
        st.markdown('<p class="section-title">Controls</p>', unsafe_allow_html=True)

        if st.button("← Back to Selection", use_container_width=True):
            st.session_state.screen = 'selection'
            st.session_state.loaded_data = None
            st.rerun()

        st.markdown("---")

        st.markdown('<p class="section-title">Visualization</p>', unsafe_allow_html=True)
        point_size = st.slider("Point Size", 1, 10, 4)
        opacity = st.slider("Opacity", 0.1, 1.0, 0.7, 0.1)

        st.markdown("---")

        st.markdown('<p class="section-title">Legend</p>', unsafe_allow_html=True)
        for species_data in st.session_state.loaded_data:
            species = species_data['name']
            color = species_data['color']
            display_name = SPECIES_DATA[species]['display_name']
            st.markdown(f"<p style='color: {color}; margin: 5px 0;'>● {display_name}</p>", unsafe_allow_html=True)

        st.markdown("---")

        st.markdown('<p class="section-title">Statistics</p>', unsafe_allow_html=True)
        total_proteins = sum(len(d['df']) for d in st.session_state.loaded_data)
        st.metric("Total Proteins", f"{total_proteins:,}")
        st.metric("Species", len(st.session_state.loaded_data))

    # Main content - UMAP plot
    fig = create_umap_plot(st.session_state.loaded_data, point_size, opacity)

    selected_points = st.plotly_chart(
        fig,
        use_container_width=True,
        key="umap_plot",
        on_select="rerun",
        selection_mode=["points", "lasso", "box"]
    )

    # Protein selection section
    st.markdown("---")
    st.markdown('<p class="section-title">Selected Proteins</p>', unsafe_allow_html=True)
    st.caption("Use lasso or box select on the plot to view protein details")

    # Handle selection
    if selected_points and selected_points.selection and len(selected_points.selection.points) > 0:
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
                        'Protein Name': str(row.get('Protein names', 'N/A'))[:70] if pd.notna(row.get('Protein names')) else 'N/A',
                        'Gene': str(row.get('Gene Names', 'N/A')) if pd.notna(row.get('Gene Names')) else 'N/A',
                        'Cluster': str(row.get('Cluster Label', 'N/A')) if pd.notna(row.get('Cluster Label')) else 'N/A',
                        'Length': row.get('Length', 'N/A'),
                    })

        if selected_data:
            st.success(f"✓ {len(selected_data)} protein{'s' if len(selected_data) != 1 else ''} selected")

            df_display = pd.DataFrame(selected_data)
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # Export button
            csv = df_display.to_csv(index=False)
            st.download_button(
                "📥 Download as CSV",
                csv,
                "selected_proteins.csv",
                "text/csv"
            )
    else:
        st.info("No proteins selected. Click or drag on the plot to select proteins.")


# Main app
def main():
    if st.session_state.screen == 'selection':
        selection_screen()
    elif st.session_state.screen == 'animation':
        animation_screen()
    elif st.session_state.screen == 'explorer':
        explorer_screen()


if __name__ == '__main__':
    main()
