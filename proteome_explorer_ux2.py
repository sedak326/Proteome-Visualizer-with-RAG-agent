"""
Marine Mammal Proteome Explorer - Ocean Theme Edition
User Flow: Select Animals (with icons) → Watch Animation → Explore Proteomes
Deep sea ocean theme with bioluminescent accents
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import base64
import subprocess
import sys

# Resolve all paths relative to this script, not the working directory
BASE_DIR = Path(__file__).resolve().parent

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

# Species configuration - colors match animation for visual consistency
SPECIES_DATA = {
    'Sea Lion': {
        'csv': str(BASE_DIR / 'umap_output/California Sealion_umap.csv'),
        'image': str(BASE_DIR / 'animal_pics/sealion.png'),
        'color': '#FF0000',
        'glow': 'rgba(255, 0, 0, 0.4)',
        'display_name': 'California Sea Lion',
        'key': 'sealion'
    },
    'Bottlenose Dolphin': {
        'csv': str(BASE_DIR / 'umap_output/Bottlenose Dolphin_umap.csv'),
        'image': str(BASE_DIR / 'animal_pics/bottlenose.png'),
        'color': '#0000FF',
        'glow': 'rgba(0, 0, 255, 0.4)',
        'display_name': 'Bottlenose Dolphin',
        'key': 'bottlenose'
    },
    'Gray Whale': {
        'csv': str(BASE_DIR / 'umap_output/Graywhale_umap.csv'),
        'image': str(BASE_DIR / 'animal_pics/graywhale.png'),
        'color': '#00CC00',
        'glow': 'rgba(0, 204, 0, 0.4)',
        'display_name': 'Gray Whale',
        'key': 'graywhale'
    },
    'Orca': {
        'csv': str(BASE_DIR / 'umap_output/Killer whale_umap.csv'),
        'image': str(BASE_DIR / 'animal_pics/orca.png'),
        'color': '#8000FF',
        'glow': 'rgba(128, 0, 255, 0.4)',
        'display_name': 'Orca (Killer Whale)',
        'key': 'orca'
    },
    'Harbor Seal': {
        'csv': str(BASE_DIR / 'umap_output/Harborseal_umap.csv'),
        'image': str(BASE_DIR / 'animal_pics/harbor_seal.png'),
        'color': '#FF8000',
        'glow': 'rgba(255, 128, 0, 0.4)',
        'display_name': 'Harbor Seal',
        'key': 'harborseal'
    },
    'King Cobra': {
        'csv': str(BASE_DIR / 'umap_output/King Cobra_umap.csv'),
        'image': str(BASE_DIR / 'animal_pics/cobra.png'),
        'color': '#00CCCC',
        'glow': 'rgba(0, 204, 204, 0.4)',
        'display_name': 'King Cobra',
        'key': 'cobra'
    }
}

SAMPLE_SIZE = 3000
PRERENDERED_DIR = BASE_DIR / 'prerendered_animations'

# Annotated CSV files per species (for protein classification/domain info)
ANNOTATION_FILES = {
    'bottlenose': [str(BASE_DIR / 'annotated/bottlenose_part1_annotated.csv'), str(BASE_DIR / 'annotated/bottlenose_part2_annotated.csv')],
    'graywhale': [str(BASE_DIR / 'annotated/graywhale_part1_annotated.csv'), str(BASE_DIR / 'annotated/graywhale_part2_annotated.csv')],
    'harborseal': [str(BASE_DIR / 'annotated/harborseal_part1_annotated.csv'), str(BASE_DIR / 'annotated/harborseal_part2_annotated.csv')],
    'orca': [str(BASE_DIR / 'annotated/orca_part1_annotated.csv'), str(BASE_DIR / 'annotated/orca_part2_annotated.csv')],
    'sealion': [str(BASE_DIR / 'annotated/sealion_part1_annotated.csv'), str(BASE_DIR / 'annotated/sealion_part2_annotated.csv')],
}

def load_annotations():
    """Load annotation data from all annotated CSVs into a global lookup dict."""
    cols = ['ident', 'clipDescription', 'hmm_labels', 'hmm_descriptions',
            'annotationConfidence', 'percentIdentity', 'queryCoverage',
            'order', 'family', 'genus']
    value_cols = [c for c in cols if c != 'ident']
    frames = []
    for file_list in ANNOTATION_FILES.values():
        for csv_path in file_list:
            p = Path(csv_path)
            if not p.exists():
                print(f"Warning: annotation file not found: {csv_path}")
                continue
            try:
                frames.append(pd.read_csv(p, usecols=cols))
            except Exception as e:
                print(f"Warning: failed to load {csv_path}: {e}")
    if not frames:
        print("Warning: no annotation files loaded")
        return {}
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=['ident'])
    df['ident'] = df['ident'].astype(str)
    df[value_cols] = df[value_cols].fillna('')
    for c in value_cols:
        df[c] = df[c].astype(str)
    df = df.set_index('ident')
    lookup = df[value_cols].to_dict('index')
    print(f"Loaded {len(lookup)} annotation entries")
    return lookup

# Load annotations at startup
ANNOTATIONS = load_annotations()

# ============= DATA LOADING =============

def encode_image_to_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    try:
        with open(image_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode()
        ext = Path(image_path).suffix.lower()
        if ext == '.gif':
            mime_type = 'image/gif'
        elif ext == '.png':
            mime_type = 'image/png'
        elif ext in ['.jpg', '.jpeg']:
            mime_type = 'image/jpeg'
        else:
            mime_type = 'image/png'
        return f"data:{mime_type};base64,{encoded}"
    except:
        return None

def get_animation_path(species_list):
    """Check if pre-rendered animation exists for species combination."""
    species_keys = sorted([SPECIES_DATA[species]['key'] for species in species_list])
    filename = '_'.join(species_keys) + '.gif'
    prerendered_path = PRERENDERED_DIR / filename
    if prerendered_path.exists():
        print(f"Using pre-rendered animation: {prerendered_path}")
        return str(prerendered_path), True
    print(f"Pre-rendered animation not found, will generate: {filename}")
    return str(prerendered_path), False

def generate_animation_on_demand(species_list):
    """Generate animation on-demand if not pre-rendered."""
    species_keys = [SPECIES_DATA[species]['key'] for species in species_list]
    animation_path, is_prerendered = get_animation_path(species_list)
    if is_prerendered:
        return animation_path
    PRERENDERED_DIR.mkdir(exist_ok=True, parents=True)
    cmd = [sys.executable, str(BASE_DIR / 'animal_morph_fixed.py')] + species_keys + ['--output', animation_path]
    print(f"Generating animation: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and Path(animation_path).exists():
            print(f"Successfully generated: {animation_path}")
            return animation_path
        else:
            print(f"Failed to generate animation: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print("Animation generation timed out (>120s)")
        return None
    except Exception as e:
        print(f"Error generating animation: {str(e)}")
        return None

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
            print(f"ERROR: CSV file not found for {species}: {csv_path}")
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
            except Exception as e:
                print(f"ERROR: Failed to process image for {species}: {e}")
                continue
        else:
            print(f"ERROR: Image file not found for {species}: {img_path}")
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

# Load animal icons
print("Loading animal icons...")
animal_icons = {}
for species, info in SPECIES_DATA.items():
    img_path = Path(info['image'])
    if img_path.exists():
        animal_icons[species] = encode_image_to_base64(str(img_path))
print(f"Loaded {len(animal_icons)} animal icons")

# ============= DASH APP =============

app = dash.Dash(__name__)

# Serve prerendered animation files over HTTP (avoids 8MB+ base64 through WebSocket)
from flask import send_from_directory

@app.server.route('/animations/<path:filename>')
def serve_animation(filename):
    return send_from_directory(str(PRERENDERED_DIR), filename)

# Ocean theme CSS with waves, particles, and glassmorphism
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <style>
            * { box-sizing: border-box; }
            body {
                margin: 0;
                padding: 0;
                background: linear-gradient(180deg, #0a192f 0%, #112240 50%, #1d3557 100%);
                min-height: 100vh;
            }
            .ocean-bg {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
                z-index: -1;
                background: linear-gradient(180deg, #0a192f 0%, #112240 40%, #1d3557 100%);
            }
            .wave {
                position: absolute;
                bottom: 0;
                left: 0;
                width: 200%;
                height: 100px;
                background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 320'%3E%3Cpath fill='%23457b9d' fill-opacity='0.15' d='M0,192L48,197.3C96,203,192,213,288,229.3C384,245,480,267,576,250.7C672,235,768,181,864,181.3C960,181,1056,235,1152,234.7C1248,235,1344,181,1392,154.7L1440,128L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z'%3E%3C/path%3E%3C/svg%3E") repeat-x;
                animation: wave-animation 25s linear infinite;
            }
            .wave:nth-child(2) { bottom: 10px; opacity: 0.5; animation: wave-animation 20s linear reverse infinite; }
            .wave:nth-child(3) { bottom: 20px; opacity: 0.3; animation: wave-animation 30s linear infinite; }
            @keyframes wave-animation {
                0% { transform: translateX(0); }
                100% { transform: translateX(-50%); }
            }
            .particles {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
                z-index: -1;
                pointer-events: none;
            }
            .particle {
                position: absolute;
                width: 6px;
                height: 6px;
                background: rgba(100, 255, 218, 0.3);
                border-radius: 50%;
                animation: float-up 15s infinite;
            }
            .particle:nth-child(1) { left: 10%; animation-delay: 0s; animation-duration: 20s; }
            .particle:nth-child(2) { left: 20%; animation-delay: 2s; animation-duration: 18s; }
            .particle:nth-child(3) { left: 30%; animation-delay: 4s; animation-duration: 22s; }
            .particle:nth-child(4) { left: 40%; animation-delay: 1s; animation-duration: 16s; }
            .particle:nth-child(5) { left: 50%; animation-delay: 3s; animation-duration: 24s; }
            .particle:nth-child(6) { left: 60%; animation-delay: 5s; animation-duration: 19s; }
            .particle:nth-child(7) { left: 70%; animation-delay: 2s; animation-duration: 21s; }
            .particle:nth-child(8) { left: 80%; animation-delay: 4s; animation-duration: 17s; }
            .particle:nth-child(9) { left: 90%; animation-delay: 1s; animation-duration: 23s; }
            @keyframes float-up {
                0% { transform: translateY(100vh) scale(0); opacity: 0; }
                10% { opacity: 1; }
                90% { opacity: 1; }
                100% { transform: translateY(-100px) scale(1); opacity: 0; }
            }
            .marine-loader {
                width: 80px;
                height: 80px;
                position: relative;
                margin: 0 auto 25px auto;
            }
            .marine-loader::before, .marine-loader::after {
                content: '';
                position: absolute;
                border-radius: 50%;
                animation: ripple 2s cubic-bezier(0, 0.2, 0.8, 1) infinite;
            }
            .marine-loader::before { width: 100%; height: 100%; border: 3px solid #64ffda; animation-delay: -0.5s; }
            .marine-loader::after { width: 100%; height: 100%; border: 3px solid #a8dadc; }
            @keyframes ripple {
                0% { transform: scale(0.1); opacity: 1; }
                100% { transform: scale(1); opacity: 0; }
            }
            .glass-card {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            }
            .glass-card:hover {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(100, 255, 218, 0.3);
                box-shadow: 0 8px 32px rgba(100, 255, 218, 0.15);
                transform: translateY(-4px);
            }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
            @keyframes fadeOut { from { opacity: 1; } to { opacity: 0; } }
            .fade-in { animation: fadeIn 0.8s ease-out; }
            .fade-out { animation: fadeOut 0.6s ease-in-out; }
            #animation-screen, #explorer-screen, #selection-screen { transition: opacity 0.8s ease-in-out; }
            ::-webkit-scrollbar { width: 8px; height: 8px; }
            ::-webkit-scrollbar-track { background: #112240; }
            ::-webkit-scrollbar-thumb { background: #457b9d; border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: #64ffda; }
            .ocean-btn {
                position: relative;
                overflow: hidden;
                transition: all 0.3s ease;
            }
            .ocean-btn::before {
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 0;
                height: 0;
                background: rgba(100, 255, 218, 0.2);
                border-radius: 50%;
                transform: translate(-50%, -50%);
                transition: width 0.6s ease, height 0.6s ease;
            }
            .ocean-btn:hover::before { width: 300px; height: 300px; }
            .shimmer-text {
                background: linear-gradient(90deg, #a8dadc, #64ffda, #a8dadc);
                background-size: 200% auto;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                animation: shimmer 3s linear infinite;
            }
            @keyframes shimmer {
                0% { background-position: 0% center; }
                100% { background-position: 200% center; }
            }
            .rc-slider-track { background: linear-gradient(90deg, #64ffda, #a8dadc) !important; }
            .rc-slider-handle { border-color: #64ffda !important; background-color: #64ffda !important; box-shadow: 0 0 10px rgba(100, 255, 218, 0.5) !important; }
            .rc-slider-rail { background-color: rgba(100, 255, 218, 0.2) !important; }
            .modebar {
                background: rgba(10, 25, 47, 0.85) !important;
                border-radius: 8px !important;
                padding: 4px 8px !important;
            }
            .modebar-btn { color: #a8dadc !important; }
            .modebar-btn:hover { color: #64ffda !important; }
            .modebar-btn.active { color: #64ffda !important; }
        </style>
    </head>
    <body>
        <div class="ocean-bg">
            <div class="wave"></div>
            <div class="wave"></div>
            <div class="wave"></div>
        </div>
        <div class="particles">
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
        </div>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    dcc.Store(id='loaded-data', data=None),
    dcc.Store(id='selected-species-store', data=[]),

    # Screen 1: Species Selection
    html.Div([
        html.Div([
            html.Div([
                html.H1("Marine Mammal Proteome Explorer",
                       className='shimmer-text',
                       style={'textAlign': 'center', 'marginBottom': 8,
                             'fontFamily': '"Inter", sans-serif', 'fontWeight': '300',
                             'fontSize': 48, 'letterSpacing': '2px'}),
                html.P("Comparative Proteomics Visualization System",
                       style={'textAlign': 'center', 'color': '#64ffda', 'fontSize': 13,
                             'marginBottom': 8, 'fontFamily': '"JetBrains Mono", monospace',
                             'letterSpacing': '3px', 'textTransform': 'uppercase'}),
                html.P("Select species for multi-dimensional protein space analysis",
                       style={'textAlign': 'center', 'color': '#a8dadc', 'fontSize': 16,
                             'marginBottom': 60, 'fontFamily': '"Inter", sans-serif',
                             'fontWeight': '300'}),
            ], style={'marginBottom': 20}),

            html.Div([
                html.Div([
                    html.Div([
                        html.Img(src=animal_icons.get(species, ''),
                                style={'width': '120px', 'height': '120px', 'objectFit': 'contain',
                                      'marginBottom': 15,
                                      'transition': 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)'}),
                        html.H3(info['display_name'],
                               style={'color': '#f1faee', 'marginBottom': 12, 'fontSize': 15,
                                     'fontFamily': '"Inter", sans-serif', 'fontWeight': '500'}),
                        html.Button('Select', id=f'btn-{species}', n_clicks=0,
                                   style={'padding': '10px 28px', 'fontSize': 11,
                                         'backgroundColor': 'transparent',
                                         'color': '#a8dadc',
                                         'border': '1px solid rgba(168, 218, 220, 0.3)',
                                         'borderRadius': 25,
                                         'cursor': 'pointer', 'transition': 'all 0.3s ease',
                                         'fontWeight': '500', 'fontFamily': '"JetBrains Mono", monospace',
                                         'letterSpacing': '2px', 'textTransform': 'uppercase'})
                    ], className='glass-card',
                       style={'padding': 25, 'textAlign': 'center',
                             'background': 'rgba(255, 255, 255, 0.03)',
                             'backdropFilter': 'blur(10px)',
                             'border': '1px solid rgba(255, 255, 255, 0.1)',
                             'borderRadius': 16},
                       id=f'card-{species}')
                ], style={'width': '18%', 'display': 'inline-block', 'margin': '1%',
                         'verticalAlign': 'top'})
                for species, info in SPECIES_DATA.items()
            ], style={'textAlign': 'center', 'marginBottom': 60}),

            html.Div([
                html.Div([
                    html.H4("Selected Organisms",
                           style={'color': '#64ffda', 'marginBottom': 15,
                                 'fontSize': 12, 'fontFamily': '"JetBrains Mono", monospace',
                                 'letterSpacing': '2px', 'textTransform': 'uppercase'}),
                    html.Div(id='selected-species-display',
                            children="No organisms selected",
                            style={'color': '#a8dadc', 'fontSize': 15, 'marginBottom': 30,
                                  'fontFamily': '"Inter", sans-serif', 'minHeight': 30}),
                    html.Button('Analyze Proteomes', id='view-button', n_clicks=0,
                               className='ocean-btn',
                               style={'padding': '16px 48px', 'fontSize': 13,
                                     'backgroundColor': 'rgba(100, 255, 218, 0.1)',
                                     'color': '#64ffda',
                                     'border': '1px solid #64ffda',
                                     'borderRadius': 30,
                                     'cursor': 'pointer', 'fontWeight': '500',
                                     'transition': 'all 0.3s ease',
                                     'fontFamily': '"JetBrains Mono", monospace',
                                     'letterSpacing': '2px',
                                     'textTransform': 'uppercase'},
                               disabled=True)
                ], className='glass-card',
                   style={'padding': 40, 'textAlign': 'center',
                         'background': 'rgba(255, 255, 255, 0.02)',
                         'backdropFilter': 'blur(10px)',
                         'border': '1px solid rgba(255, 255, 255, 0.08)',
                         'borderRadius': 20,
                         'maxWidth': 600, 'margin': '0 auto'})
            ])
        ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': 50})
    ], id='selection-screen', style={'display': 'block', 'backgroundColor': 'transparent',
                                      'minHeight': '100vh', 'paddingTop': 60}),

    # Screen 2: Loading + Animation
    html.Div([
        html.Div([
            html.Div([
                html.Div(className='marine-loader'),
                html.H3(id='loading-message',
                       children="Preparing Proteome Visualization",
                       className='shimmer-text',
                       style={'marginBottom': 12,
                             'fontFamily': '"Inter", sans-serif', 'fontWeight': '300',
                             'fontSize': 28}),
                html.P("Initializing multi-dimensional protein space transformation...",
                      style={'color': '#a8dadc', 'fontSize': 13,
                            'fontFamily': '"JetBrains Mono", monospace',
                            'letterSpacing': '1px'})
            ], id='loading-indicator', style={'textAlign': 'center', 'padding': '120px 20px'}),

            html.Div([
                html.H2("Proteome Space Transformation",
                       className='shimmer-text',
                       style={'textAlign': 'center', 'marginBottom': 10,
                             'fontFamily': '"Inter", sans-serif', 'fontWeight': '300',
                             'fontSize': 36, 'letterSpacing': '1px'}),
                html.P("Watch as species silhouettes morph into their UMAP embeddings",
                      style={'textAlign': 'center', 'color': '#a8dadc', 'fontSize': 14,
                            'marginBottom': 40, 'fontFamily': '"JetBrains Mono", monospace',
                            'letterSpacing': '1px'}),
                html.Div([
                    html.Img(id='animation-gif', src='', style={
                        'maxWidth': '100%',
                        'maxHeight': '70vh',
                        'display': 'block',
                        'margin': '0 auto',
                        'borderRadius': '16px',
                        'boxShadow': '0 20px 60px rgba(0, 0, 0, 0.4), 0 0 40px rgba(100, 255, 218, 0.1)',
                        'border': '1px solid rgba(100, 255, 218, 0.2)',
                    })
                ], style={'textAlign': 'center'})
            ], id='animation-container', style={'display': 'none'})
        ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': 50})
    ], id='animation-screen', style={'display': 'none', 'backgroundColor': 'transparent',
                                     'minHeight': '100vh', 'paddingTop': 60, 'opacity': 1}),

    # Screen 3: Interactive Explorer
    html.Div([
        # Header
        html.Div([
            html.H2("Proteome Space Explorer",
                   className='shimmer-text',
                   style={'textAlign': 'center', 'marginBottom': 8,
                         'fontFamily': '"Inter", sans-serif', 'fontWeight': '300',
                         'fontSize': 32, 'letterSpacing': '1px'}),
            html.P("Interactive UMAP embedding - hover and click to explore proteins",
                  style={'textAlign': 'center', 'color': '#a8dadc', 'fontSize': 13,
                        'marginBottom': 20, 'fontFamily': '"JetBrains Mono", monospace',
                        'letterSpacing': '1px'}),
        ]),

        # Main content: Plot on left, Controls on right
        html.Div([
            # Left side - Plot
            html.Div([
                dcc.Graph(id='interactive-graph',
                         style={'height': '60vh', 'width': '100%'},
                         config={'displayModeBar': True, 'displaylogo': False,
                                'modeBarButtonsToRemove': ['select2d', 'lasso2d']})
            ], style={'flex': '1', 'minWidth': '0', 'background': '#ffffff',
                     'borderRadius': 16, 'padding': 10,
                     'boxShadow': '0 10px 40px rgba(0, 0, 0, 0.2)',
                     'border': '1px solid rgba(100, 255, 218, 0.2)'}),

            # Right side - Controls Panel
            html.Div([
                html.Button('← Back to Selection', id='back-button', n_clicks=0,
                           className='ocean-btn',
                           style={'width': '100%', 'padding': '10px 15px', 'marginBottom': 20,
                                 'backgroundColor': 'transparent', 'color': '#64ffda',
                                 'border': '1px solid rgba(100, 255, 218, 0.3)',
                                 'borderRadius': 20,
                                 'cursor': 'pointer', 'fontFamily': '"JetBrains Mono", monospace',
                                 'fontSize': 10, 'letterSpacing': '1px', 'fontWeight': '500',
                                 'textTransform': 'uppercase'}),

                html.H3("Selected Species",
                       style={'color': '#64ffda', 'marginBottom': 12,
                             'fontSize': 10, 'fontFamily': '"JetBrains Mono", monospace',
                             'letterSpacing': '2px', 'textTransform': 'uppercase',
                             'fontWeight': '500'}),

                html.Div(id='selected-animals-display', style={'marginBottom': 20}),

                html.Hr(style={'border': 'none', 'borderTop': '1px solid rgba(100, 255, 218, 0.15)',
                              'margin': '15px 0'}),

                html.H3("Visualization",
                       style={'color': '#64ffda', 'marginBottom': 15,
                             'fontSize': 10, 'fontFamily': '"JetBrains Mono", monospace',
                             'letterSpacing': '2px', 'textTransform': 'uppercase',
                             'fontWeight': '500'}),

                html.Div([
                    html.Label("Point Size",
                              style={'fontWeight': '500', 'fontSize': 11,
                                    'color': '#a8dadc', 'fontFamily': '"Inter", sans-serif',
                                    'marginBottom': 6, 'display': 'block'}),
                    dcc.Slider(id='point-size', min=1, max=10, value=3, step=1,
                              marks={1: {'label': '1', 'style': {'color': '#a8dadc', 'fontSize': '10px'}},
                                    5: {'label': '5', 'style': {'color': '#a8dadc', 'fontSize': '10px'}},
                                    10: {'label': '10', 'style': {'color': '#a8dadc', 'fontSize': '10px'}}})
                ], style={'marginBottom': 20}),

                html.Div([
                    html.Label("Opacity",
                              style={'fontWeight': '500', 'fontSize': 11,
                                    'color': '#a8dadc', 'fontFamily': '"Inter", sans-serif',
                                    'marginBottom': 6, 'display': 'block'}),
                    dcc.Slider(id='opacity', min=0.1, max=1.0, value=0.6, step=0.1,
                              marks={0.1: {'label': '.1', 'style': {'color': '#a8dadc', 'fontSize': '10px'}},
                                    0.5: {'label': '.5', 'style': {'color': '#a8dadc', 'fontSize': '10px'}},
                                    1.0: {'label': '1', 'style': {'color': '#a8dadc', 'fontSize': '10px'}}})
                ]),

                html.Hr(style={'border': 'none', 'borderTop': '1px solid rgba(100, 255, 218, 0.15)',
                              'margin': '15px 0'}),

                html.Div([
                    html.Label("Protein Filter",
                              style={'fontWeight': '500', 'fontSize': 10,
                                    'color': '#64ffda', 'fontFamily': '"JetBrains Mono", monospace',
                                    'letterSpacing': '2px', 'textTransform': 'uppercase',
                                    'marginBottom': 10, 'display': 'block'}),
                    dcc.RadioItems(
                        id='protein-filter',
                        options=[
                            {'label': 'ALL PROTEINS', 'value': 'all'},
                            {'label': 'SPECIES-UNIQUE', 'value': 'unique'},
                            {'label': 'SHARED ACROSS SPECIES', 'value': 'shared'}
                        ],
                        value='all',
                        style={'fontFamily': '"JetBrains Mono", monospace', 'fontSize': 10,
                               'letterSpacing': '0.5px'},
                        labelStyle={'display': 'block', 'marginBottom': 8,
                                    'color': '#a8dadc', 'cursor': 'pointer'},
                        inputStyle={'marginRight': 8}
                    ),
                    html.Div(id='filter-count-display',
                             style={'fontSize': 10, 'color': '#457b9d',
                                    'fontFamily': '"JetBrains Mono", monospace',
                                    'marginTop': 8, 'letterSpacing': '0.5px'})
                ], id='protein-filter-container')

            ], className='glass-card',
               style={'width': '260px', 'flexShrink': '0', 'padding': 20,
                     'background': 'rgba(10, 25, 47, 0.9)',
                     'backdropFilter': 'blur(15px)',
                     'borderRadius': 16,
                     'border': '1px solid rgba(100, 255, 218, 0.15)',
                     'boxShadow': '0 15px 40px rgba(0, 0, 0, 0.3)',
                     'marginLeft': 20})

        ], style={'display': 'flex', 'gap': '0', 'alignItems': 'flex-start',
                 'maxWidth': '1400px', 'margin': '0 auto', 'padding': '0 30px'}),

        # Protein Selection Table - Below the plot
        html.Div([
            html.Div([
                html.H3("Selected Proteins",
                       style={'color': '#64ffda', 'marginBottom': 8,
                             'fontSize': 12, 'fontFamily': '"JetBrains Mono", monospace',
                             'letterSpacing': '2px', 'textTransform': 'uppercase',
                             'fontWeight': '500', 'display': 'inline-block'}),
                html.Span(" — Use lasso or click to select proteins from the plot",
                         style={'fontSize': 12, 'color': '#a8dadc',
                               'fontFamily': '"JetBrains Mono", monospace',
                               'letterSpacing': '0.5px'})
            ], style={'marginBottom': 15}),
            html.Div(id='selection-table-container',
                    children="No proteins selected. Click or drag to select.",
                    style={'fontSize': 13, 'fontFamily': '"Inter", sans-serif', 'color': '#a8dadc'})
        ], className='glass-card',
           style={'marginTop': 20, 'padding': 25,
                 'background': 'rgba(10, 25, 47, 0.7)',
                 'backdropFilter': 'blur(10px)',
                 'borderRadius': 16,
                 'border': '1px solid rgba(100, 255, 218, 0.1)',
                 'boxShadow': '0 15px 40px rgba(0, 0, 0, 0.2)',
                 'maxWidth': '1400px', 'margin': '20px auto 0', 'marginLeft': 30, 'marginRight': 30})

    ], id='explorer-screen', style={'display': 'none', 'fontFamily': '"Inter", sans-serif',
                                    'padding': '20px 25px', 'backgroundColor': 'transparent', 'opacity': 0}),

    dcc.Interval(id='animation-timer', interval=3400, disabled=True, n_intervals=0, max_intervals=1),
    dcc.Interval(id='fade-complete-timer', interval=800, disabled=True, n_intervals=0, max_intervals=1),
    dcc.Store(id='animation-path-store', data=None)

], style={'fontFamily': 'Arial, sans-serif'})

# ============= CALLBACKS =============

for species in SPECIES_DATA.keys():
    @app.callback(
        [Output(f'card-{species}', 'style'),
         Output(f'btn-{species}', 'children'),
         Output(f'btn-{species}', 'style')],
        [Input(f'btn-{species}', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_species(n_clicks, species=species):
        selected = n_clicks % 2 == 1
        species_color = SPECIES_DATA[species]["color"]
        species_glow = SPECIES_DATA[species].get("glow", "rgba(100, 255, 218, 0.3)")

        card_style = {
            'padding': 25, 'textAlign': 'center',
            'transition': 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
            'background': 'rgba(255, 255, 255, 0.08)' if selected else 'rgba(255, 255, 255, 0.03)',
            'backdropFilter': 'blur(10px)',
            'border': f'1px solid {species_color}' if selected else '1px solid rgba(255, 255, 255, 0.1)',
            'borderRadius': 16,
            'boxShadow': f'0 8px 32px {species_glow}, 0 0 20px {species_glow}' if selected else '0 8px 32px rgba(0, 0, 0, 0.2)',
            'transform': 'translateY(-6px) scale(1.02)' if selected else 'translateY(0) scale(1)'
        }

        btn_text = 'Selected' if selected else 'Select'
        btn_style = {
            'padding': '10px 28px', 'fontSize': 11,
            'backgroundColor': species_color if selected else 'transparent',
            'color': '#0a192f' if selected else '#a8dadc',
            'border': f'1px solid {species_color}',
            'borderRadius': 25, 'cursor': 'pointer',
            'transition': 'all 0.3s ease', 'fontWeight': '500',
            'fontFamily': '"JetBrains Mono", monospace',
            'letterSpacing': '2px', 'textTransform': 'uppercase'
        }

        return card_style, btn_text, btn_style

@app.callback(
    [Output('selected-species-display', 'children'),
     Output('selected-species-store', 'data'),
     Output('view-button', 'disabled'),
     Output('view-button', 'style')],
    [Input(f'btn-{species}', 'n_clicks') for species in SPECIES_DATA.keys()]
)
def update_selected_display(*n_clicks_list):
    selected = [species for species, clicks in zip(SPECIES_DATA.keys(), n_clicks_list)
                if clicks and clicks % 2 == 1]

    if not selected:
        display = "No organisms selected"
        disabled = True
        btn_style = {
            'padding': '16px 48px', 'fontSize': 13,
            'backgroundColor': 'rgba(100, 100, 100, 0.1)',
            'color': '#457b9d',
            'border': '1px solid rgba(100, 100, 100, 0.2)',
            'borderRadius': 30,
            'cursor': 'not-allowed', 'fontWeight': '500',
            'transition': 'all 0.3s ease',
            'fontFamily': '"JetBrains Mono", monospace',
            'letterSpacing': '2px',
            'textTransform': 'uppercase'
        }
    else:
        display = html.Div([
            html.Span([
                html.Span("●", style={'marginRight': 8, 'fontSize': 10}),
                f"{SPECIES_DATA[species]['display_name']}"
            ], style={'color': SPECIES_DATA[species]['color'],
                     'fontWeight': '500', 'marginRight': 20, 'fontSize': 15,
                     'fontFamily': '"Inter", sans-serif',
                     'textShadow': f'0 0 10px {SPECIES_DATA[species].get("glow", "rgba(100, 255, 218, 0.3)")}'})
            for species in selected
        ])
        disabled = False
        btn_style = {
            'padding': '16px 48px', 'fontSize': 13,
            'backgroundColor': '#64ffda',
            'color': '#0a192f',
            'border': 'none',
            'borderRadius': 30,
            'cursor': 'pointer', 'fontWeight': '600',
            'boxShadow': '0 0 30px rgba(100, 255, 218, 0.4), 0 10px 40px rgba(100, 255, 218, 0.2)',
            'transition': 'all 0.3s ease',
            'fontFamily': '"JetBrains Mono", monospace',
            'letterSpacing': '2px',
            'textTransform': 'uppercase'
        }

    return display, selected, disabled, btn_style

@app.callback(
    [Output('selection-screen', 'style'),
     Output('animation-screen', 'style')],
    [Input('view-button', 'n_clicks')],
    [State('selected-species-store', 'data')],
    prevent_initial_call=True
)
def show_animation_screen(n_clicks, selected_species):
    if not selected_species:
        raise dash.exceptions.PreventUpdate
    selection_style = {'display': 'none'}
    animation_style = {'display': 'block', 'backgroundColor': 'transparent',
                      'minHeight': '100vh', 'paddingTop': 60, 'opacity': 1}
    return selection_style, animation_style

@app.callback(
    [Output('loaded-data', 'data'),
     Output('animation-path-store', 'data'),
     Output('loading-message', 'children')],
    [Input('view-button', 'n_clicks')],
    [State('selected-species-store', 'data')],
    prevent_initial_call=True
)
def load_proteome_data(n_clicks, selected_species):
    if not selected_species:
        raise dash.exceptions.PreventUpdate
    animation_path, is_prerendered = get_animation_path(selected_species)
    if not is_prerendered:
        loading_msg = "Rendering Custom Proteome Visualization"
    else:
        loading_msg = "Preparing Proteome Visualization"
    if not is_prerendered:
        print(f"Generating animation on-demand for: {selected_species}")
        animation_path = generate_animation_on_demand(selected_species)
        if not animation_path:
            print("Failed to generate animation, using placeholder")
            animation_path = None
    print(f"Loading data for: {selected_species}")
    placed_animals = load_data_for_species(selected_species)

    # Tag proteins as "unique" or "shared" based on cluster membership
    # across the currently selected species only
    cluster_species = {}
    for p in placed_animals:
        for _, row in p['df'].iterrows():
            cl = row.get('Cluster Label')
            if pd.notna(cl):
                cluster_species.setdefault(int(cl), set()).add(p['name'])

    for p in placed_animals:
        tags = []
        for _, row in p['df'].iterrows():
            cl = row.get('Cluster Label')
            if pd.notna(cl) and int(cl) in cluster_species:
                tags.append('unique' if len(cluster_species[int(cl)]) == 1 else 'shared')
            else:
                tags.append('shared')
        p['df']['filter_tag'] = tags

    data_json = []
    for p in placed_animals:
        data_json.append({
            'name': p['name'],
            'x': p['x'].tolist(),
            'y': p['y'].tolist(),
            'colors': p['colors'].tolist(),
            'color': p['color'],
            'df': p['df'].to_dict('records'),
            'umap_1_scaled': p['df']['UMAP 1 Scaled'].tolist(),
            'umap_2_scaled': p['df']['UMAP 2 Scaled'].tolist()
        })
    return data_json, animation_path, loading_msg

@app.callback(
    [Output('loading-indicator', 'style'),
     Output('animation-container', 'style'),
     Output('animation-gif', 'src'),
     Output('animation-timer', 'disabled')],
    [Input('loaded-data', 'data'),
     Input('animation-path-store', 'data')],
    prevent_initial_call=True
)
def display_animation(loaded_data, animation_path):
    if not loaded_data:
        raise dash.exceptions.PreventUpdate
    print(f"Data loaded! Displaying animation from: {animation_path}")
    if animation_path and Path(animation_path).exists():
        gif_url = f'/animations/{Path(animation_path).name}'
    else:
        gif_url = ''
        print("Warning: No animation available")
    loading_style = {'display': 'none'}
    animation_container_style = {'display': 'block'}
    return loading_style, animation_container_style, gif_url, False

@app.callback(
    Output('animation-gif', 'src', allow_duplicate=True),
    [Input('animation-timer', 'n_intervals')],
    [State('animation-path-store', 'data')],
    prevent_initial_call=True
)
def freeze_animation(n, animation_path):
    if n >= 1 and animation_path:
        last_frame_path = animation_path.replace('.gif', '_last_frame.png')
        if Path(last_frame_path).exists():
            print(f"Freezing animation with last frame: {last_frame_path}")
            return f'/animations/{Path(last_frame_path).name}'
    raise dash.exceptions.PreventUpdate

app.clientside_callback(
    """
    function(n_intervals) {
        if (n_intervals >= 1) {
            const animScreen = document.getElementById('animation-screen');
            const explorerScreen = document.getElementById('explorer-screen');
            if (animScreen && explorerScreen) {
                animScreen.style.transition = 'opacity 0.6s ease-out';
                animScreen.style.opacity = '0';
                explorerScreen.style.display = 'block';
                explorerScreen.style.opacity = '0';
                setTimeout(function() {
                    explorerScreen.style.transition = 'opacity 0.8s ease-in';
                    explorerScreen.style.opacity = '1';
                }, 50);
                setTimeout(function() {
                    animScreen.style.display = 'none';
                }, 700);
            }
            return window.dash_clientside.no_update;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('fade-complete-timer', 'disabled'),
    Input('animation-timer', 'n_intervals')
)

@app.callback(
    Output('selected-animals-display', 'children'),
    [Input('loaded-data', 'data')]
)
def update_selected_animals_display(loaded_data):
    if not loaded_data:
        return html.Div("No species selected", style={'color': '#a8dadc', 'fontSize': 14})
    animals_html = []
    for species_data in loaded_data:
        species_name = species_data['name']
        img_src = animal_icons.get(species_name, '')
        species_color = species_data["color"]
        species_glow = SPECIES_DATA[species_name].get("glow", "rgba(100, 255, 218, 0.3)")
        animals_html.append(
            html.Div([
                html.Img(src=img_src,
                        style={'width': '50px', 'height': '50px', 'objectFit': 'contain',
                              'marginBottom': 6, 'filter': 'brightness(0) invert(1) opacity(0.9)'}),
                html.P(SPECIES_DATA[species_name]['display_name'],
                      style={'fontSize': 10, 'color': species_color, 'fontWeight': '500',
                            'textAlign': 'center', 'margin': 0,
                            'fontFamily': '"Inter", sans-serif'})
            ], style={'display': 'inline-block', 'textAlign': 'center',
                     'marginRight': 8, 'marginBottom': 8, 'verticalAlign': 'top',
                     'padding': 10,
                     'background': 'rgba(255, 255, 255, 0.05)',
                     'backdropFilter': 'blur(5px)',
                     'borderRadius': 12,
                     'border': f'1px solid {species_color}',
                     'boxShadow': f'0 4px 15px {species_glow}'})
        )
    return html.Div(animals_html)

@app.callback(
    Output('interactive-graph', 'figure'),
    [Input('point-size', 'value'),
     Input('opacity', 'value'),
     Input('loaded-data', 'data'),
     Input('protein-filter', 'value')]
)
def update_interactive_graph(point_size, opacity, loaded_data, protein_filter):
    if not loaded_data:
        return go.Figure()
    fig = go.Figure()
    for species_data in loaded_data:
        df_records = species_data['df']

        # Apply protein filter
        if protein_filter and protein_filter != 'all':
            df_records = [r for r in df_records if r.get('filter_tag') == protein_filter]

        if not df_records:
            # Empty trace to preserve curveNumber indexing
            fig.add_trace(go.Scattergl(
                x=[], y=[], mode='markers',
                name=species_data['name'],
                marker=dict(size=point_size, color=species_data['color'],
                           opacity=opacity, line=dict(width=0)),
                customdata=[]
            ))
            continue

        hover_text = []
        for record in df_records:
            text = f"<b>{species_data['name']}</b><br>"
            text += f"Entry: {record['Entry']}<br>"
            if 'Protein names' in record:
                text += f"Protein: {str(record['Protein names'])[:60]}...<br>"
            if 'Cluster Label' in record:
                text += f"Cluster: {record['Cluster Label']}"
            hover_text.append(text)
        fig.add_trace(go.Scattergl(
            x=[r['UMAP 1 Scaled'] for r in df_records],
            y=[r['UMAP 2 Scaled'] for r in df_records],
            mode='markers',
            name=species_data['name'],
            marker=dict(size=point_size, color=species_data['color'],
                       opacity=opacity, line=dict(width=0)),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            customdata=[[r['Entry']] for r in df_records]
        ))
    margin = 50
    fig.update_layout(
        title=None,
        xaxis=dict(range=[-300 - margin, 300 + margin], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-300 - margin, 300 + margin], showgrid=False, zeroline=False, visible=False,
                  scaleanchor="x", scaleratio=1),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        hovermode='closest',
        dragmode='lasso',
        clickmode='event+select',
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor='rgba(10, 25, 47, 0.2)',
            borderwidth=1,
            font=dict(size=12, family='"Inter", sans-serif', color='#0a192f')
        ),
        margin=dict(l=20, r=20, t=30, b=50),
        autosize=True,
        hoverlabel=dict(
            bgcolor='rgba(10, 25, 47, 0.95)',
            bordercolor='rgba(100, 255, 218, 0.3)',
            font=dict(family='"Inter", sans-serif', size=12, color='#f1faee')
        )
    )
    fig.update_traces(
        selectedpoints=[],
        selected=dict(marker=dict(opacity=1.0, size=6)),
        unselected=dict(marker=dict(opacity=0.3))
    )
    return fig

# Update filter count display
@app.callback(
    Output('filter-count-display', 'children'),
    [Input('protein-filter', 'value'),
     Input('loaded-data', 'data')]
)
def update_filter_count(protein_filter, loaded_data):
    if not loaded_data:
        return ""
    total = 0
    shown = 0
    for species_data in loaded_data:
        records = species_data['df']
        total += len(records)
        if protein_filter and protein_filter != 'all':
            shown += sum(1 for r in records if r.get('filter_tag') == protein_filter)
        else:
            shown += len(records)
    if protein_filter and protein_filter != 'all':
        return f"Showing {shown:,}/{total:,} proteins"
    return ""

# Grey out filter radio buttons when only 1 species is loaded
@app.callback(
    [Output('protein-filter-container', 'style'),
     Output('protein-filter', 'value', allow_duplicate=True)],
    [Input('loaded-data', 'data')],
    [State('protein-filter', 'value')],
    prevent_initial_call=True
)
def toggle_filter_enabled(loaded_data, current_filter):
    if not loaded_data or len(loaded_data) <= 1:
        return {'opacity': 0.35, 'pointerEvents': 'none'}, 'all'
    return {}, current_filter

@app.callback(
    Output('selection-table-container', 'children'),
    [Input('interactive-graph', 'selectedData')],
    [State('loaded-data', 'data')]
)
def show_selected_proteins(selectedData, loaded_data):
    if not selectedData or not loaded_data:
        return "No proteins selected. Click or drag to select."
    try:
        # Build lookup: {species_name: {entry_id: record}} for customdata-based lookup
        species_lookup = {}
        for species_data in loaded_data:
            entry_map = {}
            for record in species_data['df']:
                entry_map[record['Entry']] = record
            species_lookup[species_data['name']] = entry_map

        selected_proteins = []
        for point in selectedData['points']:
            curve_num = point['curveNumber']
            if curve_num >= len(loaded_data):
                continue

            species_data = loaded_data[curve_num]

            # Use customdata to find the correct protein (filter-safe)
            customdata = point.get('customdata')
            if customdata and len(customdata) > 0:
                entry_id = customdata[0]
                protein = species_lookup.get(species_data['name'], {}).get(entry_id)
            else:
                continue

            if not protein:
                continue

            entry_id_str = str(protein.get('Entry', ''))
            ann = ANNOTATIONS.get(entry_id_str)

            selected_proteins.append({
                'Entry': protein.get('Entry', 'N/A'),
                'Organism': SPECIES_DATA[species_data['name']]['display_name'],
                'Protein Name': str(protein.get('Protein names', 'N/A'))[:70],
                'Cluster': str(protein.get('Cluster Label', 'N/A')),
                'Length': protein.get('Length', 'N/A'),
                'Classification': ann.get('clipDescription', '') or '\u2014' if ann else '\u2014',
                'Domains': ann.get('hmm_labels', '') or '\u2014' if ann else '\u2014',
                'Order': ann.get('order', '') or '\u2014' if ann else '\u2014',
                'Family': ann.get('family', '') or '\u2014' if ann else '\u2014',
                'Genus': ann.get('genus', '') or '\u2014' if ann else '\u2014',
                'Confidence': ann.get('annotationConfidence', '') or '\u2014' if ann else '\u2014',
                '% Identity': ann.get('percentIdentity', '') or '\u2014' if ann else '\u2014',
                'Coverage': ann.get('queryCoverage', '') or '\u2014' if ann else '\u2014',
            })
        return html.Div([
            html.Div(f"{len(selected_proteins)} protein{'s' if len(selected_proteins) != 1 else ''} selected",
                    style={'fontSize': 13, 'fontWeight': '600', 'marginBottom': 15,
                          'color': '#64ffda', 'fontFamily': '"JetBrains Mono", monospace',
                          'letterSpacing': '1px'}),
            dash_table.DataTable(
                data=selected_proteins,
                columns=[
                    {'name': 'Entry', 'id': 'Entry'},
                    {'name': 'Organism', 'id': 'Organism'},
                    {'name': 'Protein Name', 'id': 'Protein Name'},
                    {'name': 'Cluster', 'id': 'Cluster'},
                    {'name': 'Length', 'id': 'Length'},
                    {'name': 'Classification', 'id': 'Classification'},
                    {'name': 'Domains', 'id': 'Domains'},
                    {'name': 'Order', 'id': 'Order'},
                    {'name': 'Family', 'id': 'Family'},
                    {'name': 'Genus', 'id': 'Genus'},
                    {'name': 'Confidence', 'id': 'Confidence'},
                    {'name': '% Identity', 'id': '% Identity'},
                    {'name': 'Coverage', 'id': 'Coverage'},
                ],
                style_table={'overflowX': 'auto', 'borderRadius': '8px'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '12px 10px',
                    'fontFamily': '"JetBrains Mono", monospace',
                    'fontSize': 11,
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                    'maxWidth': '180px',
                    'backgroundColor': 'transparent',
                    'border': 'none'
                },
                style_header={
                    'backgroundColor': 'rgba(100, 255, 218, 0.1)',
                    'fontWeight': '600',
                    'color': '#64ffda',
                    'borderBottom': '1px solid rgba(100, 255, 218, 0.2)',
                    'fontSize': 10,
                    'textTransform': 'uppercase',
                    'letterSpacing': '1px'
                },
                style_data={
                    'backgroundColor': 'rgba(10, 25, 47, 0.4)',
                    'color': '#f1faee',
                    'borderBottom': '1px solid rgba(100, 255, 218, 0.1)',
                    'cursor': 'pointer'
                },
                css=[{
                    'selector': 'td.dash-cell',
                    'rule': 'color: #f1faee !important;'
                }, {
                    'selector': 'td.dash-cell.focused',
                    'rule': 'background-color: rgba(100, 255, 218, 0.15) !important; color: #f1faee !important; outline: none !important;'
                }, {
                    'selector': 'input.dash-cell-value',
                    'rule': 'background-color: transparent !important; color: #f1faee !important; caret-color: transparent !important; border: none !important;'
                }, {
                    'selector': 'tr:hover td',
                    'rule': 'color: #f1faee !important;'
                }, {
                    'selector': '.dash-spreadsheet-inner tr:hover td::after, .dash-spreadsheet-inner td::after',
                    'rule': 'display: none !important;'
                }, {
                    'selector': '.dash-spreadsheet-inner tr:hover',
                    'rule': 'background: none !important;'
                }],
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgba(10, 25, 47, 0.6)'},
                    {'if': {'state': 'active'}, 'backgroundColor': 'rgba(100, 255, 218, 0.15)',
                     'color': '#f1faee',
                     'border': '1px solid rgba(100, 255, 218, 0.3)',
                     'overflow': 'visible',
                     'whiteSpace': 'normal',
                     'height': 'auto'}
                ],
                editable=False,
                page_size=10,
                page_action='native',
                sort_action='native'
            )
        ])
    except Exception as e:
        return html.Div(f"Error loading selection: {str(e)}",
                       style={'padding': 15, 'color': '#e63946',
                             'fontFamily': '"JetBrains Mono", monospace',
                             'backgroundColor': 'rgba(230, 57, 70, 0.1)',
                             'borderRadius': 8,
                             'border': '1px solid rgba(230, 57, 70, 0.3)'})

@app.callback(
    [Output('selection-screen', 'style', allow_duplicate=True),
     Output('explorer-screen', 'style', allow_duplicate=True),
     Output('animation-timer', 'n_intervals'),
     Output('animation-timer', 'disabled', allow_duplicate=True),
     Output('loading-indicator', 'style', allow_duplicate=True),
     Output('animation-container', 'style', allow_duplicate=True)],
    [Input('back-button', 'n_clicks')],
    prevent_initial_call=True
)
def go_back(n_clicks):
    if n_clicks:
        return ({'display': 'block', 'backgroundColor': 'transparent',
                'minHeight': '100vh', 'paddingTop': 60, 'opacity': 1},
                {'display': 'none', 'opacity': 0},
                0, True,
                {'textAlign': 'center', 'padding': '120px 20px'},
                {'display': 'none'})
    raise dash.exceptions.PreventUpdate

server = app.server

if __name__ == '__main__':
    import os
    print("\n" + "="*60)
    print("Marine Mammal Proteome Explorer - Ocean Edition")
    print("="*60)
    port = int(os.environ.get('PORT', 7860))
    host = '0.0.0.0'
    print(f"Open browser: http://{host}:{port}/")
    print("\nFeatures:")
    print("  - Deep ocean theme with bioluminescent accents")
    print("  - Animated wave background with floating particles")
    print("  - Glassmorphism UI with smooth transitions")
    print("  - Interactive UMAP proteome visualization")
    print("="*60 + "\n")
    app.run(debug=False, host=host, port=port)
