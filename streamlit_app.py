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

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {
        background: linear-gradient(180deg, #0a192f 0%, #112240 50%, #1d3557 100%);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

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

    /* Animal card styling */
    .animal-card {
        background: rgba(255, 255, 255, 0.03);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        height: 280px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .animal-card:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(100, 255, 218, 0.4);
        transform: translateY(-4px);
    }

    .animal-card.selected {
        background: rgba(255, 255, 255, 0.08);
        transform: translateY(-6px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }

    .animal-card img {
        width: 120px;
        height: 120px;
        object-fit: contain;
        margin-bottom: 15px;
    }

    .animal-name {
        font-size: 14px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        margin-bottom: 10px;
    }

    /* Button styling */
    .stButton > button {
        background: rgba(100, 255, 218, 0.1) !important;
        color: #64ffda !important;
        border: 1px solid #64ffda !important;
        border-radius: 25px !important;
        padding: 10px 30px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
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

    /* Animation container */
    .animation-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 40px;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 20px;
        margin: 20px 0;
    }

    /* Slider styling */
    .stSlider > div > div > div > div {
        background: #64ffda !important;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #64ffda !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    [data-testid="stMetricLabel"] {
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'screen' not in st.session_state:
    st.session_state.screen = 'selection'
if 'selected_species' not in st.session_state:
    st.session_state.selected_species = []
if 'loaded_data' not in st.session_state:
    st.session_state.loaded_data = None


def get_image_base64(image_path):
    """Convert image to base64"""
    try:
        with open(image_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{data}"
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


@st.cache_data(show_spinner=False)
def load_data_for_species(species_tuple):
    """Load proteome data for selected species."""
    species_list = list(species_tuple)
    PLOT_X_MIN, PLOT_X_MAX = -300, 300

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
        xaxis=dict(range=[-350, 350], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-350, 350], showgrid=False, zeroline=False, visible=False,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor='rgba(10, 25, 47, 0.95)',
        paper_bgcolor='rgba(10, 25, 47, 0)',
        hovermode='closest',
        dragmode='lasso',
        clickmode='event+select',
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5,
            bgcolor="rgba(10, 25, 47, 0.8)",
            bordercolor='rgba(100, 255, 218, 0.2)', borderwidth=1,
            font=dict(size=12, family='Inter, sans-serif', color='#a8dadc')
        ),
        margin=dict(l=20, r=20, t=20, b=60),
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

    # Create 5 equal columns for the animal cards
    cols = st.columns(5, gap="medium")

    species_list = list(SPECIES_DATA.keys())

    for idx, species in enumerate(species_list):
        info = SPECIES_DATA[species]
        is_selected = species in st.session_state.selected_species

        with cols[idx]:
            # Create card container
            border_color = info['color'] if is_selected else 'rgba(255, 255, 255, 0.1)'
            bg_color = 'rgba(255, 255, 255, 0.08)' if is_selected else 'rgba(255, 255, 255, 0.03)'
            shadow = f'0 10px 30px {info["color"]}40' if is_selected else 'none'
            transform = 'translateY(-6px)' if is_selected else 'translateY(0)'

            # Get image as base64
            img_path = Path(info['image'])
            img_base64 = get_image_base64(str(img_path)) if img_path.exists() else None

            img_html = f'<img src="{img_base64}" style="width: 120px; height: 120px; object-fit: contain;">' if img_base64 else '🐋'

            st.markdown(f"""
                <div style="
                    background: {bg_color};
                    border: 2px solid {border_color};
                    border-radius: 16px;
                    padding: 25px 15px;
                    text-align: center;
                    box-shadow: {shadow};
                    transform: {transform};
                    transition: all 0.3s ease;
                    min-height: 250px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                ">
                    {img_html}
                    <p style="color: {info['color']}; font-size: 13px; font-weight: 600;
                       font-family: 'Inter', sans-serif; margin: 15px 0 5px 0;">
                        {info['display_name']}
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # Toggle button
            btn_label = "✓ Selected" if is_selected else "Select"
            if st.button(btn_label, key=f"btn_{species}", use_container_width=True):
                if species in st.session_state.selected_species:
                    st.session_state.selected_species.remove(species)
                else:
                    st.session_state.selected_species.append(species)
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Selected organisms summary
    st.markdown('<p class="section-title" style="text-align: center;">Selected Organisms</p>', unsafe_allow_html=True)

    if st.session_state.selected_species:
        selected_html = " &nbsp;•&nbsp; ".join([
            f'<span style="color: {SPECIES_DATA[s]["color"]}; font-weight: 500;">{SPECIES_DATA[s]["display_name"]}</span>'
            for s in st.session_state.selected_species
        ])
        st.markdown(f'<p style="text-align: center; font-size: 16px; color: #f1faee;">{selected_html}</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="text-align: center; color: #a8dadc;">No organisms selected</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔬 Analyze Proteomes", disabled=len(st.session_state.selected_species) == 0, use_container_width=True):
            st.session_state.screen = 'animation'
            st.rerun()


def animation_screen():
    """Render the animation screen - separate page"""
    st.markdown('<h1 class="main-title">Proteome Space Transformation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">Watch as species silhouettes morph into their UMAP protein embeddings</p>', unsafe_allow_html=True)

    species_list = st.session_state.selected_species

    # Check for animation
    animation_path, exists = get_animation_path(species_list)

    if exists:
        # Display animation prominently
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            # Read and display GIF
            with open(animation_path, 'rb') as f:
                gif_data = f.read()
                gif_base64 = base64.b64encode(gif_data).decode()

            # Use HTML to ensure GIF animates properly
            st.markdown(f"""
                <div style="
                    background: rgba(0, 0, 0, 0.4);
                    border-radius: 20px;
                    padding: 30px;
                    text-align: center;
                    border: 1px solid rgba(100, 255, 218, 0.2);
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
                ">
                    <img src="data:image/gif;base64,{gif_base64}"
                         style="max-width: 100%; max-height: 60vh; border-radius: 12px;">
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        # Button to continue
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Continue to Explorer →", use_container_width=True):
                # Load data before transitioning
                with st.spinner("Loading proteome data..."):
                    st.session_state.loaded_data = load_data_for_species(tuple(sorted(species_list)))
                st.session_state.screen = 'explorer'
                st.rerun()

        # Back button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("← Back to Selection", use_container_width=True):
                st.session_state.screen = 'selection'
                st.rerun()
    else:
        # No animation available
        st.info(f"Animation not available for this species combination.")
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Continue to Explorer →", use_container_width=True):
                with st.spinner("Loading proteome data..."):
                    st.session_state.loaded_data = load_data_for_species(tuple(sorted(species_list)))
                st.session_state.screen = 'explorer'
                st.rerun()

            if st.button("← Back to Selection", use_container_width=True):
                st.session_state.screen = 'selection'
                st.rerun()


def explorer_screen():
    """Render the interactive explorer screen"""
    st.markdown('<h1 class="main-title">Proteome Space Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">Interactive UMAP embedding - hover over points to explore proteins</p>', unsafe_allow_html=True)

    # Ensure data is loaded
    if st.session_state.loaded_data is None:
        species_list = st.session_state.selected_species
        with st.spinner("Loading data..."):
            st.session_state.loaded_data = load_data_for_species(tuple(sorted(species_list)))

    # Sidebar controls
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

    # Main plot
    fig = create_umap_plot(st.session_state.loaded_data, point_size, opacity)

    selected_points = st.plotly_chart(
        fig,
        use_container_width=True,
        key="umap_plot",
        on_select="rerun",
        selection_mode=["points", "lasso", "box"]
    )

    # Protein selection
    st.markdown("---")
    st.markdown('<p class="section-title">Selected Proteins</p>', unsafe_allow_html=True)
    st.caption("Use lasso or box select on the plot to view protein details")

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

            csv = df_display.to_csv(index=False)
            st.download_button("📥 Download as CSV", csv, "selected_proteins.csv", "text/csv")
    else:
        st.info("No proteins selected. Click or drag on the plot to select proteins.")


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
