# -*- coding: utf-8 -*-
"""Fixed animal morph with proper UMAP scaling"""

import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib import animation
import sys
import argparse

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for script generation

# Mapping of species keys to file paths
SPECIES_MAP = {
    'sealion': ("C:\\Users\\Seda Kavlak\\Desktop\\Master\\Project_Work\\POC\\animation\\sealion.png", "umap_output/sealion_umap.csv"),
    'bottlenose': ("C:\\Users\\Seda Kavlak\\Desktop\\Master\\Project_Work\\POC\\animation\\bottlenose.png", "umap_output/bottlenose_umap.csv"),
    'graywhale': ("C:\\Users\\Seda Kavlak\\Desktop\\Master\\Project_Work\\POC\\animation\\graywhale.png", "umap_output/graywhale_umap.csv"),
    'orca': ("C:\\Users\\Seda Kavlak\\Desktop\\Master\\Project_Work\\POC\\animation\\orca.png", "umap_output/orca_umap.csv"),
    'harborseal': ("C:\\Users\\Seda Kavlak\\Desktop\\Master\\Project_Work\\POC\\animation\\harbor_seal.png", "umap_output/harborseal_umap.csv"),
}

# Color mapping for each species
SPECIES_COLORS = {
    'sealion': np.array([1.0, 0.0, 0.0]),      # red
    'bottlenose': np.array([0.0, 0.0, 1.0]),   # blue
    'graywhale': np.array([0.0, 0.8, 0.0]),    # green
    'orca': np.array([0.5, 0.0, 1.0]),         # purple
    'harborseal': np.array([1.0, 0.5, 0.0])    # orange
}

# Plot boundaries - increased to fit all animals
plot_x_min, plot_x_max = -300, 300
plot_y_min, plot_y_max = -300, 300
plot_width = plot_x_max - plot_x_min
plot_height = plot_y_max - plot_y_min

# Function to process images and sample coordinates
def process_image(fn, num_points):
    """Extract sampled coordinates and normalized colors from silhouette image."""
    image = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image file: {fn}")

    # Handle color and grayscale cases
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

    # Build silhouette mask
    if alpha is not None and np.std(alpha) > 1:
        mask = (alpha > 20).astype(np.uint8) * 255
    else:
        _, mask = cv2.threshold(image_gray, 40, 255, cv2.THRESH_BINARY)
        if np.mean(image_gray[mask == 255]) < np.mean(image_gray[mask == 0]):
            mask = cv2.bitwise_not(mask)

    # Extract largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found — check silhouette or background threshold.")
    largest_contour = max(contours, key=cv2.contourArea)

    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, [largest_contour], -1, 255, cv2.FILLED)

    white_coords = np.column_stack(np.where(filled_mask == 255))
    if white_coords.shape[0] < num_points:
        raise ValueError(f"Not enough silhouette pixels ({white_coords.shape[0]} < {num_points})")

    np.random.seed(0)
    sampled_coords = white_coords[np.random.choice(len(white_coords), num_points, replace=False)]
    sampled_colors = image_original[sampled_coords[:, 0], sampled_coords[:, 1]]
    sampled_colors_normalized = sampled_colors / 255.0

    return sampled_coords, sampled_colors_normalized

def scale_and_position(sampled_coords, offset_x=0, offset_y=0, scale_factor_y=1.0, x_scale_ratio=1.0):
    """Scale coordinates while maintaining aspect ratio."""
    original_width = sampled_coords[:, 1].max() - sampled_coords[:, 1].min()
    original_height = sampled_coords[:, 0].max() - sampled_coords[:, 0].min()
    scale_y = scale_factor_y * (plot_height) / original_height
    scale_x = scale_y * x_scale_ratio
    scaled_x = (sampled_coords[:, 1] - sampled_coords[:, 1].min()) * scale_x + plot_x_min + offset_x
    scaled_y = plot_y_max - ((sampled_coords[:, 0] - sampled_coords[:, 0].min()) * scale_y) + offset_y
    return scaled_x, scaled_y

def boxes_overlap(box1, box2, margin=2):
    """Check if two bounding boxes overlap (x_min, y_min, x_max, y_max)."""
    return not (box1[2] + margin < box2[0] - margin or
                box1[0] - margin > box2[2] + margin or
                box1[3] + margin < box2[1] - margin or
                box1[1] - margin > box2[3] + margin)

def place_animals_auto(animals):
    """Automatically position animals to avoid overlap."""
    placed = []
    grid_x = np.linspace(plot_x_min, plot_x_max, 10)
    grid_y = np.linspace(plot_y_min, plot_y_max, 10)

    for animal in animals:
        coords = animal["coords"]
        placed_ok = False
        for x_try in grid_x:
            for y_try in grid_y:
                # Use smaller scale (0.5) to keep all animals in frame
                scaled_x, scaled_y = scale_and_position(coords, offset_x=x_try, offset_y=y_try, scale_factor_y=0.5)
                bbox = (scaled_x.min(), scaled_y.min(), scaled_x.max(), scaled_y.max())
                if not any(boxes_overlap(bbox, p["bbox"]) for p in placed):
                    placed.append({
                        "name": animal["name"],
                        "x": scaled_x,
                        "y": scaled_y,
                        "colors": animal["colors"],
                        "df": animal["df"],
                        "bbox": bbox
                    })
                    placed_ok = True
                    break
            if placed_ok:
                break
        if not placed_ok:
            print(f"⚠️ Could not place {animal['name']} without overlap.")
    return placed

# Smart auto-placement: divide space into regions
def place_animals_smart(animals, plot_bounds):
    """Intelligently position animals in grid layout with automatic scaling."""
    n_animals = len(animals)
    placed = []

    # Create grid layout (rows x cols)
    if n_animals <= 2:
        rows, cols = 1, 2
    elif n_animals <= 4:
        rows, cols = 2, 2
    else:
        rows = int(np.ceil(np.sqrt(n_animals)))
        cols = int(np.ceil(n_animals / rows))

    plot_x_min_local, plot_x_max_local, plot_y_min_local, plot_y_max_local = plot_bounds
    cell_width = (plot_x_max_local - plot_x_min_local) / cols
    cell_height = (plot_y_max_local - plot_y_min_local) / rows

    # Add padding within each cell (10% on each side)
    padding = 0.1
    usable_width = cell_width * (1 - 2 * padding)
    usable_height = cell_height * (1 - 2 * padding)

    for idx, animal in enumerate(animals):
        # Calculate grid position
        row = idx // cols
        col = idx % cols

        # Get center of this cell
        cell_center_x = plot_x_min_local + (col + 0.5) * cell_width
        cell_center_y = plot_y_max_local - (row + 0.5) * cell_height  # Y increases downward

        coords = animal["coords"]

        # Calculate original dimensions
        orig_width = coords[:, 1].max() - coords[:, 1].min()
        orig_height = coords[:, 0].max() - coords[:, 0].min()
        aspect_ratio = orig_width / orig_height

        # Determine scale to fit in cell
        if aspect_ratio > (usable_width / usable_height):
            # Width is limiting factor
            scale = usable_width / orig_width
        else:
            # Height is limiting factor
            scale = usable_height / orig_height

        # Scale coordinates
        scaled_coords_x = (coords[:, 1] - coords[:, 1].min()) * scale
        scaled_coords_y = (coords[:, 0] - coords[:, 0].min()) * scale

        # Center in cell
        final_x = scaled_coords_x - scaled_coords_x.mean() + cell_center_x
        final_y = -(scaled_coords_y - scaled_coords_y.mean()) + cell_center_y

        bbox = (final_x.min(), final_y.min(), final_x.max(), final_y.max())
        placed.append({
            "name": animal["name"],
            "x": final_x,
            "y": final_y,
            "colors": animal["colors"],
            "df": animal["df"],
            "bbox": bbox,
            "species_key": animal.get("species_key")
        })
        print(f"Placed {animal['name']}: bbox X[{bbox[0]:.1f}, {bbox[2]:.1f}], Y[{bbox[1]:.1f}, {bbox[3]:.1f}]")

    return placed

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Generate animal morph animation')
    parser.add_argument('species', nargs='+', help='Species keys to include (e.g., sealion bottlenose)')
    parser.add_argument('--output', required=True, help='Output GIF path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Filter animal files based on selected species
    selected_species_keys = args.species
    animal_files = [SPECIES_MAP[key] for key in selected_species_keys if key in SPECIES_MAP]

    if not animal_files:
        print(f"ERROR: No valid species found in: {selected_species_keys}")
        sys.exit(1)

    print(f"Generating animation for: {selected_species_keys}")

    animals = []
    SAMPLE_SIZE = 5000  # Subsample to 5000 points for clearer visualization

    for img_file, csv_file in animal_files:
        df = pd.read_csv(csv_file)[['Entry', 'UMAP 1', 'UMAP 2', 'Cluster Label']]

        # Subsample the dataframe if it's too large
        if len(df) > SAMPLE_SIZE:
            df = df.sample(n=SAMPLE_SIZE, random_state=42)

        coords, colors = process_image(img_file, len(df))
        animals.append({
            "name": img_file.split('\\')[-1].split('.')[0],
            "coords": coords,
            "colors": colors,
            "df": df,
            "species_key": [k for k, v in SPECIES_MAP.items() if v == (img_file, csv_file)][0]
        })

    # Use smart placement
    plot_bounds = (plot_x_min, plot_x_max, plot_y_min, plot_y_max)
    placed_animals = place_animals_smart(animals, plot_bounds)

    # ============= KEY FIX: SCALE UMAP COORDINATES TO PLOT SPACE =============

    # Determine combined UMAP limits
    umap_x_min = min(p["df"]['UMAP 1'].min() for p in placed_animals)
    umap_x_max = max(p["df"]['UMAP 1'].max() for p in placed_animals)
    umap_y_min = min(p["df"]['UMAP 2'].min() for p in placed_animals)
    umap_y_max = max(p["df"]['UMAP 2'].max() for p in placed_animals)

    print(f"UMAP X range: [{umap_x_min:.2f}, {umap_x_max:.2f}]")
    print(f"UMAP Y range: [{umap_y_min:.2f}, {umap_y_max:.2f}]")

    # Scale UMAP coordinates to fill most of the plot space (80% to avoid edges)
    # Scale to match the UX expectations (center at origin, uniformly scaled)
    umap_center_x = (umap_x_min + umap_x_max) / 2
    umap_center_y = (umap_y_min + umap_y_max) / 2
    umap_max_range = max(umap_x_max - umap_x_min, umap_y_max - umap_y_min)

    scale_factor = (plot_x_max - plot_x_min) * 0.8 / umap_max_range

    for p in placed_animals:
        p["df"]['UMAP 1 Scaled'] = (p["df"]['UMAP 1'] - umap_center_x) * scale_factor
        p["df"]['UMAP 2 Scaled'] = (p["df"]['UMAP 2'] - umap_center_y) * scale_factor

    # ============= CREATE ANIMATION WITH SCALED COORDINATES =============

    fig, ax = plt.subplots(figsize=(10, 6))
    scatters = [ax.scatter([], [], s=2, alpha=0.6, edgecolors='none') for _ in placed_animals]

    def init():
        # Set limits to match plot boundaries with small margin
        margin = 50
        ax.set_xlim(plot_x_min - margin, plot_x_max + margin)
        ax.set_ylim(plot_y_min - margin, plot_y_max + margin)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Animal to UMAP Morph Animation")
        ax.axis('off')
        return scatters

    # Animation timing parameters
    total_frames = 80
    initial_pause = 10
    final_pause = 10
    transition_frames = total_frames - initial_pause - final_pause

    def update(frame):
        # Compute lambda for transition (0 → 1 smoothly)
        if frame < initial_pause:
            lambda_value = 0
        elif frame > initial_pause + transition_frames:
            lambda_value = 1
        else:
            lambda_value = (frame - initial_pause) / transition_frames

        for i, (p, sc) in enumerate(zip(placed_animals, scatters)):
            # Use species-specific color from SPECIES_COLORS
            end_color = SPECIES_COLORS.get(p["species_key"], np.array([0.5, 0.5, 0.5]))
            color_transition = (1 - lambda_value) * p["colors"] + lambda_value * end_color

            # USE SCALED UMAP COORDINATES
            x_data = (1 - lambda_value) * p["x"] + lambda_value * p["df"]['UMAP 1 Scaled']
            y_data = (1 - lambda_value) * p["y"] + lambda_value * p["df"]['UMAP 2 Scaled']

            sc.set_offsets(np.c_[x_data, y_data])
            sc.set_color(np.clip(color_transition, 0, 1))

        return scatters

    # Build and save animation
    print(f"Creating animation with {total_frames} frames...")
    ani = animation.FuncAnimation(fig, update, frames=np.arange(total_frames), init_func=init, blit=False)

    print(f"Saving animation to: {args.output}")
    ani.save(args.output, writer='pillow', fps=20, dpi=150)

    # Save last frame as PNG
    last_frame_path = args.output.replace('.gif', '_last_frame.png')
    update(total_frames - 1)  # Render last frame
    plt.savefig(last_frame_path, dpi=150, bbox_inches='tight', facecolor='#fafafa')
    print(f"Saved last frame to: {last_frame_path}")

    print(f"Animation saved successfully: {args.output}")
    plt.close()
