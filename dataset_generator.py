import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

# --- Configuration ---
DATASET_SIZE = 2  
MAIN_DIR = "moire_ml_dataset_test"

# Subdirectory setup
folders = {
    "moire": os.path.join(MAIN_DIR, "moire"),
    "layer1": os.path.join(MAIN_DIR, "layer1"),
    "layer2": os.path.join(MAIN_DIR, "layer2")
}

for folder in folders.values():
    os.makedirs(folder, exist_ok=True)



def plot_moire_final_unified(L=100, rotation_angle=1.0, spacing=1.0, lattice_type='square', 
                             radius_a=0.7, radius_b=0.7, gradient_steps=15, gamma=2.2,
                             enable_vacancies=False, vacancy_rate_a=0.05, vacancy_rate_b=0.05, sample_id=0):
    
    # 1. Grid Generation logic
    indices = np.arange(L)
    i, j = np.meshgrid(indices, indices)
    
    if lattice_type.lower() == 'square':
        x, y = i * spacing, j * spacing
    else: # Hexagonal logic
        x = spacing * (i + 0.5 * (j % 2))
        y = spacing * (j * np.sqrt(3) / 2)

    positions = np.stack([x.flatten(), y.flatten()], axis=1)
    
    # 2. Assign Radii (Checkerboard mask)
    mask = (i + j) % 2 == 0
    flat_radii = np.where(mask.flatten(), radius_a, radius_b) * spacing
    
    # 3. Rotation math
    cx, cy = np.mean(positions, axis=0)
    theta = np.deg2rad(rotation_angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    rot_positions = np.stack([
        (positions[:, 0] - cx) * cos_t - (positions[:, 1] - cy) * sin_t + cx,
        (positions[:, 0] - cx) * sin_t + (positions[:, 1] - cy) * cos_t + cy
    ], axis=1)

    # 4. VACANCY LOGIC
    # We create a mask for both layers. True = Keep, False = Remove.
    if enable_vacancies:
        # Generate random floats between 0 and 1 for every atom in both layers
        # If the number is > vacancy_rate, we keep the atom
        v_mask_layer1 = np.random.rand(len(positions)) > vacancy_rate_a
        v_mask_layer2 = np.random.rand(len(rot_positions)) > vacancy_rate_b
        
        # Apply masks to positions and radii
        positions = positions[v_mask_layer1]
        layer1_radii = flat_radii[v_mask_layer1]
        
        rot_positions = rot_positions[v_mask_layer2]
        layer2_radii = flat_radii[v_mask_layer2]
    else:
        layer1_radii = flat_radii
        layer2_radii = flat_radii

    # Combine layers for unified rendering
    combined_pos = np.vstack([positions, rot_positions])
    combined_radii = np.concatenate([layer1_radii, layer2_radii])


    moire_path = os.path.join(folders["moire"], f"sample_{sample_id}.png")
    l1_path = os.path.join(folders["layer1"], f"sample_{sample_id}.png")
    l2_path = os.path.join(folders["layer2"], f"sample_{sample_id}.png")

    plot_layer(gradient_steps, positions, layer1_radii, gamma, spacing, L, x, y, l1_path)
    plot_layer(gradient_steps, rot_positions, layer2_radii, gamma, spacing, L, x, y, l2_path)
    plot_layer(gradient_steps, combined_pos, combined_radii, gamma, spacing, L, x, y, moire_path)

    return {
        "sample_id": sample_id,
        "moire_path": moire_path,
        "layer1_path": l1_path,
        "layer2_path": l2_path,
        "angle": rotation_angle,
        "lattice_type": 1 if lattice_type == 'square' else 0,
        "enable_vacancies": enable_vacancies,
        "vacancy_rate_a": vacancy_rate_a,
        "vacancy_rate_b": vacancy_rate_b
    }

def plot_layer(gradient_steps, pos, radii, gamma, spacing, L, x, y, path):
        # 5. Setup Plot
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
    ax.set_facecolor('black')
    layer_alpha = 0.8

    def draw_glow_unified(pos_list, radii_list):
        r_multipliers = np.linspace(1.0, 0.02, gradient_steps)
        c_vals = np.linspace(0.1, 1.0, gradient_steps)**gamma 
        
        for frac, c in zip(r_multipliers, c_vals):
            circles = [Circle((p[0], p[1]), radius=r * frac) for p, r in zip(pos_list, radii_list)]
            col = PatchCollection(circles, facecolors='white', edgecolors='none', 
                                  alpha=(c * layer_alpha) / gradient_steps, antialiased=True)
            ax.add_collection(col)

    draw_glow_unified(pos, radii)

    margin = max(spacing, (L * spacing) * 0.1) 

    ax.set_xlim(np.min(x) + margin, np.max(x) - margin)
    ax.set_ylim(np.min(y) + margin, np.max(y) - margin)
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    #plt.show()

# --- RUN ---

metadata = []
for s in range(DATASET_SIZE):
    if s % 50 == 0: print(f"Generating sample {s}...")
    L = 100
    rotation_angle =  np.random.uniform(0.5, 45)
    lattice_type = np.random.choice(['square', 'hexagonal'])
    radius_a = np.random.choice([0.3, 0.4, 0.5, 0.6, 0.7])
    radius_b = np.random.choice([0.3, 0.4, 0.5, 0.6, 0.7])
    enable_vacancies = np.random.choice([True, False])
    vacancy_rate_a = np.random.choice([0, 0.02, 0.04, 0.06, 0.08, 0.1])
    vacancy_rate_b = np.random.choice([0, 0.02, 0.04, 0.06, 0.08, 0.1])
    metadata.append(plot_moire_final_unified(L=L, rotation_angle=rotation_angle, lattice_type=lattice_type, enable_vacancies=enable_vacancies, radius_a=radius_a, radius_b=radius_b, vacancy_rate_a=vacancy_rate_a, vacancy_rate_b=vacancy_rate_b, sample_id=s))

pd.DataFrame(metadata).to_csv("metadata_with_layers_test.csv", index=False)

