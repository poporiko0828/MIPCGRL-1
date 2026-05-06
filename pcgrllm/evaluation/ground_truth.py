import json
import numpy as np
import matplotlib.pyplot as plt
from os import path

# Load the JSON file containing the alphabet grids
json_file = path.join(path.dirname(__file__), 'ground_truth_64x64.json')
with open(json_file, 'r') as f:
    ground_truth_data = json.load(f)


# Set up the visualization function for multiple letters
def visualize_multiple_alphabets(grid_data: dict, columns: int = 3):
    """
    Visualizes multiple alphabet grids in a grid format using matplotlib.

    Parameters:
    - grid_data (dict): Dictionary containing alphabet grids.
    - columns (int): Number of columns in the plot grid.
    """
    num_letters = len(grid_data)
    rows = (num_letters + columns - 1) // columns  # Calculate rows based on number of letters and columns

    fig, axes = plt.subplots(rows, columns, figsize=(columns * 6, rows * 6))  # Increase the figure size
    axes = axes.flatten()  # Flatten axes for easy iteration

    for idx, (alphabet, grid) in enumerate(grid_data.items()):
        arr = np.array(grid)  # Convert the list to a Numpy array
        axes[idx].imshow(arr, cmap='gray', interpolation='nearest')
        axes[idx].set_title(f'{alphabet}', fontsize=20)  # Larger title font size
        axes[idx].axis('off')  # Hide axes for cleaner visualization

    # Hide any remaining empty subplots
    for ax in axes[num_letters:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# Visualize multiple letters in a grid layout with larger letters (3 letters per row)
visualize_multiple_alphabets(ground_truth_data, columns=3)
