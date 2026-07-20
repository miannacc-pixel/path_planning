# plot_prm_path.py

import matplotlib.pyplot as plt
import csv
import matplotlib.patches as patches
import numpy as np
import os

def read_obstacles(filename):
    obstacles = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            type_ = row[0]
            num_vertices = int(row[1])
            vertices = []
            verts = row[2:]
            for i in range(0, len(verts), 2):
                x = float(verts[i])
                y = float(verts[i+1])
                vertices.append((x, y))
            obstacles.append((type_, vertices))
    return obstacles

def plot_obstacles(obstacles, ax):
    for obs in obstacles:
        type_, vertices = obs
        polygon = patches.Polygon(vertices, closed=True, fill=True, edgecolor='black', facecolor='gray', alpha=0.5)
        ax.add_patch(polygon)

def read_path(filename):
    path = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x = float(row['x'])
            y = float(row['y'])
            P11 = float(row['P11'])
            P12 = float(row['P12'])
            P22 = float(row['P22'])
            path.append((x, y, P11, P12, P22))
    return path

def plot_path(path, ax):
    xs = [state[0] for state in path]
    ys = [state[1] for state in path]
    ax.plot(xs, ys, '-o', color='blue', label='PRM* Path')

    # Plot uncertainty ellipsoids
    for idx, state in enumerate(path):
        x, y, P11, P12, P22 = state
        cov = np.array([[P11, P12],
                        [P12, P22]])

        # Ensure covariance matrix is symmetric
        cov = (cov + cov.T) / 2

        # Eigenvalues and eigenvectors for the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Handle negative eigenvalues
        if np.any(eigenvalues < 0):
            print(f"Negative eigenvalues at index {idx}, position ({x}, {y}): {eigenvalues}")
            # Adjust small negative eigenvalues
            eigenvalues[eigenvalues < 0] = 0.0
            # If eigenvalues are significantly negative, skip plotting
            if np.any(eigenvalues < -1e-6):
                print(f"Significant negative eigenvalues encountered, skipping ellipsoid at index {idx}.")
                continue

        # Proceed with plotting
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle = np.degrees(np.arctan2(*eigenvectors[:,0][::-1]))
        width, height = 2 * np.sqrt(eigenvalues)  # 1-sigma ellipse

        # Check for NaN or infinite values
        if not np.isfinite(width) or not np.isfinite(height):
            print(f"Non-finite ellipse dimensions at index {idx}, skipping.")
            continue

        ellip = patches.Ellipse((x, y), width, height, angle=angle,
                    edgecolor='red', facecolor='none', linestyle='--', alpha=0.5)
        ax.add_patch(ellip)

def main():
    obstacles_filename = 'obstacles_prm.csv'
    path_filename = 'path_data_prm.csv'
    if not os.path.exists(obstacles_filename) or not os.path.exists(path_filename):
        print("Required files not found.")
        return

    obstacles = read_obstacles(obstacles_filename)
    path = read_path(path_filename)

    fig, ax = plt.subplots(figsize=(10,10))
    plot_obstacles(obstacles, ax)
    plot_path(path, ax)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('PRM* Path with Uncertainty Ellipsoids')
    ax.legend()
    ax.set_aspect('equal')
    plt.grid(True)
    plt.savefig('prm_path.png')
    plt.close(fig)
    print("Figure saved as 'prm_path.png'.")

if __name__ == "__main__":
    main()