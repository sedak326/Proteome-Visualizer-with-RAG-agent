"""
Marine Mammal Proteome Explorer - Premium UX Version
User Flow: Select Animals (with icons) → Watch Animation → Explore Proteomes
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

# Species configuration
SPECIES_DATA = {
    'Sea Lion': {
        'csv': 'umap_output/sealion_umap.csv',
        'image': 'animation/sealion.png',
        'color': '#FF0000',
        'display_name': 'California Sea Lion',
        'key': 'sealion'
    },
    'Bottlenose Dolphin': {
        'csv': 'umap_output/bottlenose_umap.csv',
        'image': 'animation/bottlenose.png',
        'color': '#0000FF',
        'display_name': 'Bottlenose Dolphin',
        'key': 'bottlenose'
    },
    'Gray Whale': {
        'csv': 'umap_output/graywhale_umap.csv',
        'image': 'animation/graywhale.png',
        'color': '#00CC00',
        'display_name': 'Gray Whale',
        'key': 'graywhale'
    },
    'Orca': {
        'csv': 'umap_output/orca_umap.csv',
        'image': 'animation/orca.png',
        'color': '#8000FF',
        'display_name': 'Orca (Killer Whale)',
        'key': 'orca'
    },
    'Harbor Seal': {
        'csv': 'umap_output/harborseal_umap.csv',
        'image': 'animation/harbor_seal.png',
        'color': '#FF8000',
        'display_name': 'Harbor Seal',
        'key': 'harborseal'
    }
}

SAMPLE_SIZE = 3000  #Maximum proteins to display per species (for performance)
PRERENDERED_DIR = Path('prerendered_animations') #where to store pre-rendered GIFs

# ============= DATA LOADING =============

def encode_image_to_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    try:
        with open(image_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode()

        # Determine mime type based on extension
        ext = Path(image_path).suffix.lower()
        if ext == '.gif':
            mime_type = 'image/gif'
        elif ext == '.png':
            mime_type = 'image/png'
        elif ext in ['.jpg', '.jpeg']:
            mime_type = 'image/jpeg'
        else:
            mime_type = 'image/png'  # default

        return f"data:{mime_type};base64,{encoded}"
    except:
        return None

def get_animation_path(species_list):
    """
    Checks if pre-rendered animation exists for species combination. 
    If not, indicates on-demand generation is needed.
    """
    # Generate filename from species keys
    species_keys = sorted([SPECIES_DATA[species]['key'] for species in species_list])
    filename = '_'.join(species_keys) + '.gif'
    prerendered_path = PRERENDERED_DIR / filename

    # Check if pre-rendered version exists
    if prerendered_path.exists():
        print(f"Using pre-rendered animation: {prerendered_path}")
        return str(prerendered_path), True

    # Otherwise, need to generate on-demand
    print(f"Pre-rendered animation not found, will generate: {filename}")
    return str(prerendered_path), False

def generate_animation_on_demand(species_list):
    """
    Builds the command to run the animation script.
    """
    species_keys = [SPECIES_DATA[species]['key'] for species in species_list]
    animation_path, is_prerendered = get_animation_path(species_list)

    if is_prerendered:
        return animation_path

    # Create directory if needed
    PRERENDERED_DIR.mkdir(exist_ok=True, parents=True)

    # Generate animation using animal_morph_fixed.py
    cmd = [sys.executable, 'animal_morph_fixed.py'] + species_keys + ['--output', animation_path]

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
                print(f"  Image path: {img_path}")
                continue
        else:
            print(f"ERROR: Image file not found for {species}: {img_path}")
            print(f"  Please ensure the image exists at the specified location.")
            continue

        animals_data.append({
            "name": species,
            "coords": coords,
            "colors": colors,
            "df": df,
            "color": info['color']
        })

    # Check if we have any valid data
    if not animals_data:
        raise ValueError("No valid animal data could be loaded. Please check file paths and data integrity.")

    # Place animals and scale UMAP
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

# Encode all animal images for selection screen
print("Loading animal icons...")
animal_icons = {}
for species, info in SPECIES_DATA.items():
    img_path = Path(info['image'])
    if img_path.exists():
        animal_icons[species] = encode_image_to_base64(str(img_path))
print(f"Loaded {len(animal_icons)} animal icons")

# ============= DASH APP =============

app = dash.Dash(__name__)

# Add custom CSS for loading spinner and fade transitions
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }

            @keyframes fadeOut {
                from { opacity: 1; }
                to { opacity: 0; }
            }

            /* Smooth fade transitions for screen changes */
            .fade-in {
                animation: fadeIn 0.8s ease-in-out;
            }

            .fade-out {
                animation: fadeOut 0.6s ease-in-out;
            }

            /* Ensure smooth transitions */
            #animation-screen, #explorer-screen, #selection-screen {
                transition: opacity 0.8s ease-in-out;
            }
        </style>
    </head>
    <body>
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
    # Store for loaded data
    dcc.Store(id='loaded-data', data=None),
    dcc.Store(id='selected-species-store', data=[]),

    # Screen 1: Species Selection
    html.Div([
        html.Div([
            html.H1("Marine Mammal Proteome Explorer",
                   style={'textAlign': 'center', 'color': '#1a1a2e', 'marginBottom': 5,
                         'fontFamily': '"Segoe UI", "Roboto", sans-serif', 'fontWeight': '300',
                         'fontSize': 42, 'letterSpacing': '1px'}),
            html.P("Comparative Proteomics Visualization System",
                   style={'textAlign': 'center', 'color': '#6c757d', 'fontSize': 14,
                         'marginBottom': 10, 'fontFamily': 'monospace', 'letterSpacing': '2px',
                         'textTransform': 'uppercase'}),
            html.P("Select species for multi-dimensional protein space analysis",
                   style={'textAlign': 'center', 'color': '#495057', 'fontSize': 15,
                         'marginBottom': 50, 'fontFamily': '"Segoe UI", sans-serif'}),

            # Animal selection cards
            html.Div([
                html.Div([
                    html.Div([
                        html.Img(src=animal_icons.get(species, ''),
                                style={'width': '140px', 'height': '140px', 'objectFit': 'contain',
                                      'marginBottom': 12, 'filter': 'grayscale(100%) brightness(1.1)',
                                      'transition': 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)'}),
                        html.H3(info['display_name'],
                               style={'color': '#212529', 'marginBottom': 8, 'fontSize': 16,
                                     'fontFamily': '"Segoe UI", sans-serif', 'fontWeight': '500'}),
                        html.Button('Select', id=f'btn-{species}', n_clicks=0,
                                   style={'padding': '8px 24px', 'fontSize': 13,
                                         'backgroundColor': 'transparent', 'color': '#495057',
                                         'border': '1.5px solid #dee2e6', 'borderRadius': 4,
                                         'cursor': 'pointer', 'transition': 'all 0.3s ease',
                                         'fontWeight': '500', 'fontFamily': 'monospace',
                                         'letterSpacing': '1px', 'textTransform': 'uppercase'})
                    ], style={'padding': 18, 'backgroundColor': '#fafafa',
                             'borderRadius': 8, 'boxShadow': '0 2px 8px rgba(0,0,0,0.06)',
                             'textAlign': 'center', 'transition': 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
                             'border': '2px solid #e9ecef'},
                       id=f'card-{species}')
                ], style={'width': '18%', 'display': 'inline-block', 'margin': '1%',
                         'verticalAlign': 'top'})
                for species, info in SPECIES_DATA.items()
            ], style={'textAlign': 'center', 'marginBottom': 50}),

            html.Div([
                html.H4("Selected Organisms", style={'color': '#212529', 'marginBottom': 12,
                       'fontSize': 15, 'fontFamily': 'monospace', 'letterSpacing': '1px',
                       'textTransform': 'uppercase'}),
                html.Div(id='selected-species-display',
                        children="No organisms selected",
                        style={'color': '#868e96', 'fontSize': 14, 'marginBottom': 25,
                              'fontFamily': '"Segoe UI", sans-serif'}),
                html.Button('Analyze Proteomes', id='view-button', n_clicks=0,
                           style={'padding': '14px 42px', 'fontSize': 14,
                                 'backgroundColor': '#1a1a2e', 'color': '#ffffff',
                                 'border': 'none', 'borderRadius': 4,
                                 'cursor': 'pointer', 'fontWeight': '500',
                                 'boxShadow': '0 4px 12px rgba(26,26,46,0.15)',
                                 'transition': 'all 0.3s ease',
                                 'fontFamily': 'monospace', 'letterSpacing': '1.5px',
                                 'textTransform': 'uppercase'},
                           disabled=True)
            ], style={'textAlign': 'center'})

        ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': 50})
    ], id='selection-screen', style={'display': 'block', 'backgroundColor': '#ffffff',
                                      'minHeight': '100vh', 'paddingTop': 60}),

    # Screen 2: Loading + Animation
    html.Div([
        html.Div([
            # Loading state (shows before animation is ready)
            html.Div([
                html.Div(className='loader', style={
                    'border': '8px solid #f3f3f3',
                    'borderTop': '8px solid #1a1a2e',
                    'borderRadius': '50%',
                    'width': '60px',
                    'height': '60px',
                    'animation': 'spin 1s linear infinite',
                    'margin': '0 auto 20px auto'
                }),
                html.H3(id='loading-message',
                       children="Preparing Proteome Visualization",
                       style={'color': '#1a1a2e', 'marginBottom': 10,
                       'fontFamily': '"Segoe UI", sans-serif', 'fontWeight': '300'}),
                html.P("Initializing multi-dimensional protein space transformation...",
                      style={'color': '#868e96', 'fontSize': 13, 'fontFamily': 'monospace'})
            ], id='loading-indicator', style={'textAlign': 'center', 'padding': '100px 20px'}),

            # Animation state (shows GIF when ready)
            html.Div([
                html.H2("Proteome Space Transformation",
                       style={'textAlign': 'center', 'color': '#1a1a2e', 'marginBottom': 8,
                             'fontFamily': '"Segoe UI", sans-serif', 'fontWeight': '300',
                             'fontSize': 32, 'letterSpacing': '0.5px'}),
                html.P("Watch as species silhouettes morph into their UMAP embeddings",
                      style={'textAlign': 'center', 'color': '#868e96', 'fontSize': 13,
                            'marginBottom': 35, 'fontFamily': 'monospace', 'letterSpacing': '1px'}),
                html.Div([
                    html.Img(id='animation-gif', src='', style={
                        'maxWidth': '100%',
                        'maxHeight': '70vh',
                        'display': 'block',
                        'margin': '0 auto',
                        'borderRadius': '8px',
                        'boxShadow': '0 4px 12px rgba(0,0,0,0.1)',
                        'animation': 'none'  # Prevent CSS animation loops
                    })
                ], style={'textAlign': 'center'})
            ], id='animation-container', style={'display': 'none'})
        ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': 50})
    ], id='animation-screen', style={'display': 'none', 'backgroundColor': '#f8f9fa',
                                     'minHeight': '100vh', 'paddingTop': 60, 'opacity': 1}),

    # Screen 3: Interactive Explorer
    html.Div([
        html.Div([
            # Main graph container - centered like animation, same max-width
            html.Div([
                html.H2("Proteome Space Transformation",
                       style={'textAlign': 'center', 'color': '#1a1a2e', 'marginBottom': 8,
                             'fontFamily': '"Segoe UI", sans-serif', 'fontWeight': '300',
                             'fontSize': 32, 'letterSpacing': '0.5px'}),
                html.P("Interactive UMAP embedding - hover and click to explore proteins",
                      style={'textAlign': 'center', 'color': '#868e96', 'fontSize': 13,
                            'marginBottom': 35, 'fontFamily': 'monospace', 'letterSpacing': '1px'}),

                # Graph in same position as animation
                dcc.Graph(id='interactive-graph',
                         style={'height': '68vh', 'width': '100%'},
                         config={'displayModeBar': True, 'displaylogo': False,
                                'modeBarButtonsToRemove': ['select2d', 'lasso2d']})
            ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': 50}),

        ], style={'marginBottom': 20}),

        # Controls - RIGHT SIDE (narrower panel)
        html.Div([
            html.Div([
                html.Button('← Back', id='back-button', n_clicks=0,
                           style={'width': '100%', 'padding': 10, 'marginBottom': 25,
                                 'backgroundColor': 'transparent', 'color': '#495057',
                                 'border': '1.5px solid #dee2e6', 'borderRadius': 4,
                                 'cursor': 'pointer', 'fontFamily': 'monospace',
                                 'fontSize': 12, 'letterSpacing': '1px', 'fontWeight': '500'}),

                html.H3("Selected Organisms", style={'color': '#212529', 'marginBottom': 15,
                       'fontSize': 14, 'fontFamily': 'monospace', 'letterSpacing': '1px',
                       'textTransform': 'uppercase', 'fontWeight': '500'}),

                # Dynamic selected animal icons
                html.Div(id='selected-animals-display', style={'marginBottom': 25}),

                html.H3("Visualization Controls", style={'color': '#212529', 'marginBottom': 15,
                       'fontSize': 14, 'fontFamily': 'monospace', 'letterSpacing': '1px',
                       'textTransform': 'uppercase', 'fontWeight': '500'}),

                html.Div([
                    html.Label("Point Size", style={'fontWeight': '500', 'fontSize': 12,
                              'color': '#495057', 'fontFamily': '"Segoe UI", sans-serif'}),
                    dcc.Slider(id='point-size', min=1, max=10, value=3, step=1,
                              marks={1: '1', 5: '5', 10: '10'})
                ], style={'marginBottom': 22}),

                html.Div([
                    html.Label("Opacity", style={'fontWeight': '500', 'fontSize': 12,
                              'color': '#495057', 'fontFamily': '"Segoe UI", sans-serif'}),
                    dcc.Slider(id='opacity', min=0.1, max=1.0, value=0.6, step=0.1,
                              marks={0.1: '0.1', 0.5: '0.5', 1.0: '1.0'})
                ], style={'marginBottom': 22}),

                html.Div(id='species-legend', style={'marginTop': 25})

            ], style={'width': '29%', 'display': 'inline-block', 'verticalAlign': 'top',
                     'marginLeft': '2%', 'padding': 22, 'backgroundColor': '#f8f9fa',
                     'borderRadius': 6, 'border': '1px solid #e9ecef'})
        ], style={'marginBottom': 20}),

        # Protein selection info
        html.Div([
            html.Div([
                html.H3("Selected Proteins", style={'color': '#212529', 'marginBottom': 5,
                       'fontSize': 14, 'fontFamily': 'monospace', 'letterSpacing': '1px',
                       'textTransform': 'uppercase', 'fontWeight': '500'}),
                html.Div("Click a protein or drag to select multiple proteins (lasso/box select)",
                        style={'fontSize': 11, 'color': '#868e96', 'fontFamily': 'monospace',
                              'marginBottom': 15})
            ]),
            html.Div(id='selection-table-container',
                    children="No proteins selected. Click or drag to select.",
                    style={'padding': 20, 'backgroundColor': '#ffffff',
                          'borderRadius': 5, 'fontSize': 13,
                          'fontFamily': '"Segoe UI", sans-serif', 'color': '#6c757d',
                          'border': '1px solid #e9ecef'})
        ], style={'marginTop': 30, 'padding': 22, 'backgroundColor': '#f8f9fa',
                 'borderRadius': 6, 'marginLeft': 20, 'marginRight': 20,
                 'border': '1px solid #e9ecef'})

    ], id='explorer-screen', style={'display': 'none', 'fontFamily': '"Segoe UI", sans-serif',
                                    'padding': 25, 'backgroundColor': '#ffffff', 'opacity': 0}),

    # Timer to freeze GIF and start fade transition (3.4 seconds to catch last frame before loop)
    dcc.Interval(id='animation-timer', interval=3400, disabled=True, n_intervals=0, max_intervals=1),

    # Timer for completing the fade transition (triggered after animation-timer)
    dcc.Interval(id='fade-complete-timer', interval=800, disabled=True, n_intervals=0, max_intervals=1),

    # Store for animation path
    dcc.Store(id='animation-path-store', data=None)

], style={'fontFamily': 'Arial, sans-serif'})

# ============= CALLBACKS =============

# Handle species card selection
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

        card_style = {
            'padding': 18, 'backgroundColor': '#ffffff' if selected else '#fafafa',
            'borderRadius': 8, 'boxShadow': '0 4px 12px rgba(0,0,0,0.12)' if selected else '0 2px 8px rgba(0,0,0,0.06)',
            'textAlign': 'center', 'transition': 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
            'border': f'2px solid {SPECIES_DATA[species]["color"]}' if selected else '2px solid #e9ecef',
            'transform': 'translateY(-2px)' if selected else 'translateY(0)'
        }

        btn_text = 'Selected' if selected else 'Select'
        btn_style = {
            'padding': '8px 24px', 'fontSize': 13,
            'backgroundColor': SPECIES_DATA[species]["color"] if selected else 'transparent',
            'color': '#ffffff' if selected else '#495057',
            'border': f'1.5px solid {SPECIES_DATA[species]["color"]}',
            'borderRadius': 4, 'cursor': 'pointer',
            'transition': 'all 0.3s ease', 'fontWeight': '500',
            'fontFamily': 'monospace', 'letterSpacing': '1px', 'textTransform': 'uppercase'
        }

        return card_style, btn_text, btn_style

# Update selected species display and enable/disable view button
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
            'padding': '14px 42px', 'fontSize': 14,
            'backgroundColor': '#e9ecef', 'color': '#adb5bd',
            'border': 'none', 'borderRadius': 4,
            'cursor': 'not-allowed', 'fontWeight': '500',
            'boxShadow': 'none', 'transition': 'all 0.3s ease',
            'fontFamily': 'monospace', 'letterSpacing': '1.5px',
            'textTransform': 'uppercase'
        }
    else:
        display = html.Div([
            html.Span(f"{SPECIES_DATA[species]['display_name']}",
                     style={'color': SPECIES_DATA[species]['color'],
                           'fontWeight': '500', 'marginRight': 12, 'fontSize': 14,
                           'fontFamily': '"Segoe UI", sans-serif'})
            for species in selected
        ])
        disabled = False
        btn_style = {
            'padding': '14px 42px', 'fontSize': 14,
            'backgroundColor': '#1a1a2e', 'color': '#ffffff',
            'border': 'none', 'borderRadius': 4,
            'cursor': 'pointer', 'fontWeight': '500',
            'boxShadow': '0 4px 12px rgba(26,26,46,0.15)',
            'transition': 'all 0.3s ease',
            'fontFamily': 'monospace', 'letterSpacing': '1.5px',
            'textTransform': 'uppercase'
        }

    return display, selected, disabled, btn_style

# View button: show animation screen with loading indicator
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
    animation_style = {'display': 'block', 'backgroundColor': '#f8f9fa',
                      'minHeight': '100vh', 'paddingTop': 60, 'opacity': 1}

    return selection_style, animation_style

# Load data and animation when view button is clicked
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

    # Check if animation exists or needs to be generated
    animation_path, is_prerendered = get_animation_path(selected_species)

    if not is_prerendered:
        # Update loading message for on-demand generation
        loading_msg = "Rendering Custom Proteome Visualization"
    else:
        loading_msg = "Preparing Proteome Visualization"

    # Generate animation if needed (this will block while generating)
    if not is_prerendered:
        print(f"Generating animation on-demand for: {selected_species}")
        animation_path = generate_animation_on_demand(selected_species)

        if not animation_path:
            print("Failed to generate animation, using placeholder")
            animation_path = None

    # Load data for selected species
    print(f"Loading data for: {selected_species}")
    placed_animals = load_data_for_species(selected_species)

    # Convert to JSON-serializable format
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

# Display animation GIF when data and animation are loaded
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

    # Encode GIF to base64 for display
    if animation_path and Path(animation_path).exists():
        gif_base64 = encode_image_to_base64(animation_path)
    else:
        # Fallback: no animation available
        gif_base64 = ''
        print("Warning: No animation available")

    # Hide loading, show animation, enable timer (to transition to explorer after 4s)
    loading_style = {'display': 'none'}
    animation_container_style = {'display': 'block'}

    return loading_style, animation_container_style, gif_base64, False

# Swap GIF with last frame PNG right before transition to freeze it
@app.callback(
    Output('animation-gif', 'src', allow_duplicate=True),
    [Input('animation-timer', 'n_intervals')],
    [State('animation-path-store', 'data')],
    prevent_initial_call=True
)
def freeze_animation(n, animation_path):
    if n >= 1 and animation_path:
        # Replace .gif with _last_frame.png
        last_frame_path = animation_path.replace('.gif', '_last_frame.png')
        if Path(last_frame_path).exists():
            print(f"Freezing animation with last frame: {last_frame_path}")
            return encode_image_to_base64(last_frame_path)
    raise dash.exceptions.PreventUpdate

# Transition to explorer after animation with smooth fade (clientside callback for smoother CSS)
app.clientside_callback(
    """
    function(n_intervals) {
        if (n_intervals >= 1) {
            // Get both screens
            const animScreen = document.getElementById('animation-screen');
            const explorerScreen = document.getElementById('explorer-screen');

            if (animScreen && explorerScreen) {
                // Fade out animation screen
                animScreen.style.transition = 'opacity 0.6s ease-out';
                animScreen.style.opacity = '0';

                // Show explorer screen and start fade in
                explorerScreen.style.display = 'block';
                explorerScreen.style.opacity = '0';

                // Trigger fade in after a brief delay
                setTimeout(function() {
                    explorerScreen.style.transition = 'opacity 0.8s ease-in';
                    explorerScreen.style.opacity = '1';
                }, 50);

                // Hide animation screen after fade completes
                setTimeout(function() {
                    animScreen.style.display = 'none';
                }, 700);
            }

            return window.dash_clientside.no_update;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('fade-complete-timer', 'disabled'),  # Dummy output
    Input('animation-timer', 'n_intervals')
)

# Update selected animals display in explorer
@app.callback(
    Output('selected-animals-display', 'children'),
    [Input('loaded-data', 'data')]
)
def update_selected_animals_display(loaded_data):
    if not loaded_data:
        return html.Div("No species selected", style={'color': '#7f8c8d', 'fontSize': 14})

    animals_html = []
    for species_data in loaded_data:
        species_name = species_data['name']
        img_src = animal_icons.get(species_name, '')

        animals_html.append(
            html.Div([
                html.Img(src=img_src,
                        style={'width': '60px', 'height': '60px', 'objectFit': 'contain',
                              'marginBottom': 5}),
                html.P(SPECIES_DATA[species_name]['display_name'],
                      style={'fontSize': 12, 'color': '#2c3e50', 'fontWeight': 'bold',
                            'textAlign': 'center', 'margin': 0})
            ], style={'display': 'inline-block', 'textAlign': 'center',
                     'marginRight': 10, 'marginBottom': 10, 'verticalAlign': 'top',
                     'padding': 8, 'backgroundColor': 'white', 'borderRadius': 8,
                     'border': f'2px solid {species_data["color"]}',
                     'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        )

    return html.Div(animals_html)

# Interactive graph
@app.callback(
    Output('interactive-graph', 'figure'),
    [Input('point-size', 'value'),
     Input('opacity', 'value'),
     Input('loaded-data', 'data')]
)
def update_interactive_graph(point_size, opacity, loaded_data):
    if not loaded_data:
        return go.Figure()

    fig = go.Figure()

    for species_data in loaded_data:
        df_records = species_data['df']

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

    # Match animation plot boundaries exactly
    # Animation uses: plot_x_min=-300, plot_x_max=300, margin=50
    margin = 50

    fig.update_layout(
        # Remove title to match animation
        title=None,
        # Set exact same axis ranges as animation
        xaxis=dict(
            range=[-300 - margin, 300 + margin],  # [-350, 350]
            showgrid=False,
            zeroline=False,
            visible=False  # Hide axes to match animation
        ),
        yaxis=dict(
            range=[-300 - margin, 300 + margin],  # [-350, 350]
            showgrid=False,
            zeroline=False,
            visible=False,
            scaleanchor="x",  # Force equal aspect ratio
            scaleratio=1
        ),
        plot_bgcolor='#fafafa',  # Match animation background
        paper_bgcolor='#f8f9fa',  # Match animation screen background
        hovermode='closest',
        # Enable selection tools
        dragmode='lasso',  # Default to lasso select (can also use 'select' for box)
        clickmode='event+select',  # Allow both clicking and selecting
        # Position legend at bottom to match animation
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.05,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor='#dee2e6',
            borderwidth=1,
            font=dict(size=11, family='"Segoe UI", sans-serif')
        ),
        # Match animation figure size and margins
        margin=dict(l=20, r=20, t=20, b=20),
        # Maintain aspect ratio
        autosize=True
    )

    # Configure selection appearance
    fig.update_traces(
        selectedpoints=[],
        selected=dict(marker=dict(opacity=1.0, size=6)),
        unselected=dict(marker=dict(opacity=0.3))
    )

    return fig

# Handle protein selection (single or multiple)
@app.callback(
    Output('selection-table-container', 'children'),
    [Input('interactive-graph', 'selectedData')],
    [State('loaded-data', 'data')]
)
def show_selected_proteins(selectedData, loaded_data):
    if not selectedData or not loaded_data:
        return "No proteins selected. Click or drag to select."

    try:
        # Collect all selected proteins
        selected_proteins = []

        for point in selectedData['points']:
            curve_num = point['curveNumber']
            point_num = point['pointNumber']

            species_data = loaded_data[curve_num]
            protein = species_data['df'][point_num]

            selected_proteins.append({
                'Entry': protein.get('Entry', 'N/A'),
                'Organism': SPECIES_DATA[species_data['name']]['display_name'],
                'Protein Name': str(protein.get('Protein names', 'N/A'))[:70],
                'Gene': str(protein.get('Gene Names', 'N/A')),
                'Cluster': str(protein.get('Cluster Label', 'N/A')),
                'UMAP 1': f"{protein.get('UMAP 1', 0):.3f}",
                'UMAP 2': f"{protein.get('UMAP 2', 0):.3f}",
                'Length': protein.get('Length', 'N/A')
            })

        # Create DataTable
        return html.Div([
            html.Div(f"{len(selected_proteins)} protein{'s' if len(selected_proteins) != 1 else ''} selected",
                    style={'fontSize': 12, 'fontWeight': '600', 'marginBottom': 10,
                          'color': '#212529', 'fontFamily': 'monospace'}),
            dash_table.DataTable(
                data=selected_proteins,
                columns=[
                    {'name': 'Entry', 'id': 'Entry'},
                    {'name': 'Organism', 'id': 'Organism'},
                    {'name': 'Protein Name', 'id': 'Protein Name'},
                    {'name': 'Gene', 'id': 'Gene'},
                    {'name': 'Cluster', 'id': 'Cluster'},
                    {'name': 'Length', 'id': 'Length'},
                    {'name': 'UMAP 1', 'id': 'UMAP 1'},
                    {'name': 'UMAP 2', 'id': 'UMAP 2'}
                ],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '8px',
                    'fontFamily': 'monospace',
                    'fontSize': 11,
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                    'maxWidth': 0
                },
                style_header={
                    'backgroundColor': '#f8f9fa',
                    'fontWeight': '600',
                    'color': '#495057',
                    'borderBottom': '2px solid #dee2e6',
                    'fontSize': 10,
                    'textTransform': 'uppercase',
                    'letterSpacing': '1px'
                },
                style_data={
                    'backgroundColor': '#ffffff',
                    'color': '#212529',
                    'borderBottom': '1px solid #e9ecef'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f8f9fa'
                    }
                ],
                page_size=15,
                page_action='native',
                sort_action='native',
                filter_action='native',
                export_format='csv',
                export_headers='display'
            )
        ])
    except Exception as e:
        return html.Div(f"Error loading selection: {str(e)}",
                       style={'padding': 15, 'color': '#dc3545', 'fontFamily': 'monospace'})

# Back button
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
        return ({'display': 'block', 'backgroundColor': '#ffffff',
                'minHeight': '100vh', 'paddingTop': 60, 'opacity': 1},
                {'display': 'none', 'opacity': 0},  # Reset opacity
                0,  # Reset animation counter
                True,  # Disable timer
                {'textAlign': 'center', 'padding': '100px 20px'},  # Reset loading indicator
                {'display': 'none'})  # Hide animation container
    raise dash.exceptions.PreventUpdate

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Marine Mammal Proteome Explorer - Premium UX")
    print("="*60)
    print("Open browser: http://127.0.0.1:8050/")
    print("\nUser Flow:")
    print("  1. Select animals by clicking their images")
    print("  2. Click 'View Proteomes' to start")
    print("  3. Watch the morphing animation")
    print("  4. Explore the interactive proteome space")
    print("="*60 + "\n")

    app.run(debug=True, host='127.0.0.1', port=8050)
