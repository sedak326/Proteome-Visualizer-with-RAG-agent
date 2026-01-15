"""
Fast Interactive Proteome Visualizer with Quick Loading Animation
Optimized for speed - shows animation as a loading screen, then full interactivity
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

# Define available species
SPECIES_DATA = {
    'Sea Lion': {'csv': 'umap_output/sealion_umap.csv', 'color': '#FF0000'},
    'Bottlenose Dolphin': {'csv': 'umap_output/bottlenose_umap.csv', 'color': '#0000FF'},
    'Gray Whale': {'csv': 'umap_output/graywhale_umap.csv', 'color': '#00CC00'},
    'Orca': {'csv': 'umap_output/orca_umap.csv', 'color': '#8000FF'},
    'Harbor Seal': {'csv': 'umap_output/harborseal_umap.csv', 'color': '#FF8000'}
}

SAMPLE_SIZE = 3000  

def load_proteome_data():
    """Load proteome data only (skip animal images for speed)"""
    data = {}
    for species, info in SPECIES_DATA.items():
        csv_path = Path(info['csv'])
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping {species}")
            continue

        df = pd.read_csv(csv_path)
        essential_cols = ['Entry', 'Protein names', 'Gene Names', 'Organism',
                         'UMAP 1', 'UMAP 2', 'Cluster Label', 'Length']
        available_cols = [col for col in essential_cols if col in df.columns]
        df = df[available_cols]

        # Subsample for performance
        if len(df) > SAMPLE_SIZE:
            df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

        data[species] = {
            'df': df,
            'color': info['color']
        }
        print(f"Loaded {len(df)} proteins for {species}")

    return data

print("Loading proteome data...")
proteome_data = load_proteome_data()

# Scale UMAP coordinates uniformly
all_dfs = [d['df'] for d in proteome_data.values()]
umap_x_min = min(df['UMAP 1'].min() for df in all_dfs)
umap_x_max = max(df['UMAP 1'].max() for df in all_dfs)
umap_y_min = min(df['UMAP 2'].min() for df in all_dfs)
umap_y_max = max(df['UMAP 2'].max() for df in all_dfs)

umap_center_x = (umap_x_min + umap_x_max) / 2
umap_center_y = (umap_y_min + umap_y_max) / 2
umap_max_range = max(umap_x_max - umap_x_min, umap_y_max - umap_y_min)

# Normalize UMAP to -1 to 1 range for easier manipulation
for species, data in proteome_data.items():
    data['df']['UMAP_X_norm'] = (data['df']['UMAP 1'] - umap_center_x) / (umap_max_range / 2)
    data['df']['UMAP_Y_norm'] = (data['df']['UMAP 2'] - umap_center_y) / (umap_max_range / 2)

print(f"Loaded {len(proteome_data)} species")

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    # Loading screen (shown initially)
    html.Div([
        html.Div([
            html.H1("🐋 Marine Mammal Proteome Explorer 🦭",
                   style={'color': '#2c3e50', 'marginBottom': 20}),
            html.Div([
                html.Div(id='loading-animation-container',
                        style={'width': '100%', 'height': '400px'})
            ]),
            html.H3("Loading Proteomes...", id='loading-text',
                   style={'color': '#7f8c8d', 'marginTop': 20}),
            html.Div([
                html.Div(style={
                    'width': '0%',
                    'height': '10px',
                    'backgroundColor': '#3498db',
                    'borderRadius': '5px',
                    'transition': 'width 0.3s ease'
                }, id='progress-bar')
            ], style={
                'width': '60%',
                'margin': '0 auto',
                'backgroundColor': '#ecf0f1',
                'borderRadius': '5px',
                'marginTop': 20
            })
        ], style={
            'textAlign': 'center',
            'padding': '100px 20px',
            'maxWidth': '800px',
            'margin': '0 auto'
        })
    ], id='loading-screen', style={'display': 'block'}),

    # Main app (hidden initially)
    html.Div([
        html.Div([
            html.H1("Interactive Proteome Visualizer",
                    style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
            html.P("Explore and compare proteomes across marine mammal species",
                   style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': 20})
        ]),

        html.Div([
            # Control panel
            html.Div([
                html.H3("Controls", style={'color': '#2c3e50', 'marginBottom': 15}),

                html.Div([
                    html.Label("Select Species:", style={'fontWeight': 'bold', 'marginBottom': 10}),
                    dcc.Checklist(
                        id='species-selector',
                        options=[{'label': name, 'value': name} for name in proteome_data.keys()],
                        value=list(proteome_data.keys()),
                        labelStyle={'display': 'block', 'marginBottom': 8},
                        style={'fontSize': 14}
                    )
                ], style={'marginBottom': 20}),

                html.Div([
                    html.Label("Point Size:", style={'fontWeight': 'bold'}),
                    dcc.Slider(
                        id='point-size-slider',
                        min=1,
                        max=10,
                        step=1,
                        value=3,
                        marks={i: str(i) for i in [1, 5, 10]},
                        tooltip={"placement": "bottom"}
                    )
                ], style={'marginBottom': 20}),

                html.Div([
                    html.Label("Opacity:", style={'fontWeight': 'bold'}),
                    dcc.Slider(
                        id='opacity-slider',
                        min=0.1,
                        max=1.0,
                        step=0.1,
                        value=0.6,
                        marks={0.1: '0.1', 0.5: '0.5', 1.0: '1.0'},
                        tooltip={"placement": "bottom"}
                    )
                ])
            ], style={'width': '22%', 'display': 'inline-block', 'verticalAlign': 'top',
                     'padding': 20, 'backgroundColor': '#ecf0f1', 'borderRadius': 10}),

            # Main visualization
            html.Div([
                dcc.Graph(
                    id='proteome-scatter',
                    style={'height': '70vh'},
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], style={'width': '75%', 'display': 'inline-block', 'marginLeft': '2%'}),
        ]),

        # Protein details panel
        html.Div([
            html.H3("Protein Details", style={'color': '#2c3e50', 'marginBottom': 15}),
            html.Div(id='protein-details',
                    children="Click on a protein to see details",
                    style={'padding': 20, 'backgroundColor': '#f8f9fa',
                          'borderRadius': 5, 'minHeight': 100, 'fontSize': 14})
        ], style={'marginTop': 30, 'padding': 20, 'backgroundColor': '#ecf0f1',
                 'borderRadius': 10, 'marginLeft': 20, 'marginRight': 20})

    ], id='main-app', style={'display': 'none', 'fontFamily': 'Arial, sans-serif',
                             'padding': 20, 'backgroundColor': '#ffffff'}),

    # Timer for loading animation
    dcc.Interval(id='loading-interval', interval=100, n_intervals=0, max_intervals=30),
    dcc.Store(id='animation-complete', data=False)
])

@app.callback(
    [Output('loading-animation-container', 'children'),
     Output('progress-bar', 'style'),
     Output('loading-text', 'children')],
    [Input('loading-interval', 'n_intervals')]
)
def update_loading_animation(n):
    """Create a simple scatter animation as loading screen"""
    progress = min(n / 30 * 100, 100)

    # Create simple morphing visualization
    fig = go.Figure()

    # Simple circle to scattered points animation
    t = n / 30  # 0 to 1

    for i, (species, data) in enumerate(proteome_data.items()):
        df = data['df']

        # Start in a circle, morph to UMAP positions
        angle = np.linspace(0, 2*np.pi, len(df))
        radius = 0.3 + i * 0.15

        circle_x = radius * np.cos(angle)
        circle_y = radius * np.sin(angle)

        # Smooth transition
        smooth_t = 3 * t**2 - 2 * t**3  # Smooth step function
        x = (1 - smooth_t) * circle_x + smooth_t * df['UMAP_X_norm']
        y = (1 - smooth_t) * circle_y + smooth_t * df['UMAP_Y_norm']

        fig.add_trace(go.Scattergl(
            x=x,
            y=y,
            mode='markers',
            name=species,
            marker=dict(size=2, color=data['color'], opacity=0.6),
            showlegend=(n > 10)
        ))

    fig.update_layout(
        xaxis=dict(visible=False, range=[-1.5, 1.5]),
        yaxis=dict(visible=False, range=[-1.5, 1.5]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )

    progress_style = {
        'width': f'{progress}%',
        'height': '10px',
        'backgroundColor': '#3498db',
        'borderRadius': '5px',
        'transition': 'width 0.3s ease'
    }

    if progress < 100:
        text = f"Loading proteomes... {int(progress)}%"
    else:
        text = "Ready! Launching explorer..."

    return dcc.Graph(figure=fig, config={'displayModeBar': False}), progress_style, text

@app.callback(
    [Output('loading-screen', 'style'),
     Output('main-app', 'style'),
     Output('animation-complete', 'data')],
    [Input('loading-interval', 'n_intervals')]
)
def hide_loading_screen(n):
    """Hide loading screen after animation completes"""
    if n >= 30:
        return {'display': 'none'}, {'display': 'block', 'fontFamily': 'Arial, sans-serif',
                                      'padding': 20, 'backgroundColor': '#ffffff'}, True
    return {'display': 'block'}, {'display': 'none'}, False

@app.callback(
    Output('proteome-scatter', 'figure'),
    [Input('species-selector', 'value'),
     Input('point-size-slider', 'value'),
     Input('opacity-slider', 'value'),
     Input('animation-complete', 'data')]
)
def update_scatter(selected_species, point_size, opacity, animation_complete):
    """Update main scatter plot"""
    if not animation_complete:
        return go.Figure()

    fig = go.Figure()

    if not selected_species:
        fig.add_annotation(
            text="Please select at least one species",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="#7f8c8d")
        )
        return fig

    for species in selected_species:
        if species not in proteome_data:
            continue

        data = proteome_data[species]
        df = data['df']

        # Create hover text
        hover_text = []
        for _, row in df.iterrows():
            text = f"<b>{species}</b><br>"
            text += f"Entry: {row['Entry']}<br>"
            if 'Protein names' in df.columns:
                text += f"Protein: {str(row['Protein names'])[:60]}...<br>"
            if 'Cluster Label' in df.columns:
                text += f"Cluster: {row['Cluster Label']}"
            hover_text.append(text)

        fig.add_trace(go.Scattergl(
            x=df['UMAP 1'],
            y=df['UMAP 2'],
            mode='markers',
            name=species,
            marker=dict(
                size=point_size,
                color=data['color'],
                opacity=opacity,
                line=dict(width=0)
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            customdata=df[['Entry']].values
        ))

    fig.update_layout(
        title=dict(
            text=f"Proteome Comparison: {', '.join(selected_species)}",
            font=dict(size=18, color="#2c3e50")
        ),
        xaxis=dict(title="UMAP 1", showgrid=True, gridcolor='#dfe6e9'),
        yaxis=dict(title="UMAP 2", showgrid=True, gridcolor='#dfe6e9'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#2c3e50", borderwidth=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    return fig

@app.callback(
    Output('protein-details', 'children'),
    [Input('proteome-scatter', 'clickData')],
    [State('species-selector', 'value')]
)
def display_protein_details(clickData, selected_species):
    """Display protein details on click"""
    if not clickData:
        return html.Div([
            html.P("Click on a protein in the plot to see detailed information",
                  style={'color': '#7f8c8d', 'fontStyle': 'italic'})
        ])

    try:
        point = clickData['points'][0]
        curve_num = point['curveNumber']
        point_num = point['pointNumber']

        species = selected_species[curve_num]
        df = proteome_data[species]['df']
        protein_data = df.iloc[point_num]

        details = html.Div([
            html.H4(f"{species} - {protein_data['Entry']}",
                   style={'color': proteome_data[species]['color'], 'marginBottom': 10}),
            html.Hr(),
            html.Div([
                html.Div([html.Strong("Protein: "), html.Span(str(protein_data.get('Protein names', 'N/A'))[:100])],
                        style={'marginBottom': 10}),
                html.Div([html.Strong("Gene: "), html.Span(str(protein_data.get('Gene Names', 'N/A')))],
                        style={'marginBottom': 10}),
                html.Div([html.Strong("Organism: "), html.Span(str(protein_data.get('Organism', 'N/A')))],
                        style={'marginBottom': 10}),
                html.Div([html.Strong("Length: "), html.Span(f"{protein_data.get('Length', 'N/A')} aa")],
                        style={'marginBottom': 10}),
                html.Div([html.Strong("Cluster: "), html.Span(str(protein_data.get('Cluster Label', 'N/A')))],
                        style={'marginBottom': 10}),
                html.Div([html.Strong("UMAP: "),
                         html.Span(f"({protein_data['UMAP 1']:.2f}, {protein_data['UMAP 2']:.2f})")])
            ])
        ])

        return details

    except Exception as e:
        return html.Div([html.P(f"Error: {str(e)}", style={'color': '#e74c3c'})])

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting Fast Interactive Proteome Visualizer")
    print("="*60)
    print(f"Loaded {len(proteome_data)} species:")
    for species in proteome_data.keys():
        print(f"  - {species}")
    print("\nOpen your browser to: http://127.0.0.1:8050/")
    print("Watch the loading animation, then explore!")
    print("="*60 + "\n")

    app.run(debug=True, host='127.0.0.1', port=8050)
