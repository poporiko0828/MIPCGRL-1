# Updated to save images and numpy arrays into separate folders within 'iteration_1/inference/'
import json
import numpy as np
from PIL import Image
from os.path import join, dirname, abspath
import os

# Load the grid from the uploaded JSON file
json_file_path = 'ground_truth_16x16.json'
with open(json_file_path, 'r') as file:
    grid_data = json.load(file)

# Define the paths for tile images (solid and empty)
file_path = abspath(join(dirname(__file__), '..', '..', 'envs', 'probs', 'tile_ims'))

tile_images = {
    1: Image.open(join(file_path, 'empty.png')),
    2: Image.open(join(file_path, 'solid.png'))
}

# Set grid and cell size
grid_size = (16, 16)  # Target grid size
cell_size = 18  # Cell size in pixels

# Directory structure
output_dir = 'output_images'
iteration_dir = 'iteration_1/inference'
os.makedirs(output_dir, exist_ok=True)

# Function to save numpy array and image
def save_grid_image_and_array(output_dir, letter, level, array, grid_image, iteration_dir):
    letter_dir = join(output_dir, letter, iteration_dir)
    images_dir = join(letter_dir, 'images')
    numpy_dir = join(letter_dir, 'numpy')

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(numpy_dir, exist_ok=True)

    # Save the grid image
    output_image_path = join(images_dir, f'level_{level}.png')
    grid_image.save(output_image_path)

    # Save the numpy array
    npy_path = join(numpy_dir, f'level_{level}.npy')
    np.save(npy_path, array)

# Iterate over each letter in grid_data
for letter, array in grid_data.items():
    array = np.array(array)
    np.set_printoptions(threshold=np.inf, linewidth=1000)

    # Create a new image for the grid
    base_grid_image = Image.new('RGB', (cell_size * grid_size[0], cell_size * grid_size[1]))

    # Fill the base grid based on the array values
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            tile_img = tile_images[array[i][j]].resize((cell_size, cell_size))
            base_grid_image.paste(tile_img, (j * cell_size, i * cell_size))

    # Save the original image and array
    save_grid_image_and_array(output_dir, letter, 0, array, base_grid_image, iteration_dir)

    # Create transformations and save each
    transformations = {
        1: np.fliplr(array),  # Left-right flip
        2: np.flipud(array),  # Top-bottom flip
        3: np.rot90(array, k=1),  # 90-degree clockwise
        4: np.rot90(array, k=2),  # 180-degree
        5: np.rot90(array, k=3),  # 90-degree counterclockwise
        6: np.rot90(array, k=1)[:, ::-1],  # 90-degree clockwise and flip
        7: np.rot90(array, k=2)[:, ::-1],  # 180-degree flip
        8: np.rot90(array, k=3)[:, ::-1],  # 90-degree counterclockwise and flip
        9: np.rot90(array, k=2),  # 90-degree counterclockwise
    }

    for level, transformed_array in transformations.items():
        # Create the image for the transformed array
        transformed_grid_image = Image.new('RGB', (cell_size * grid_size[0], cell_size * grid_size[1]))
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                tile_img = tile_images[transformed_array[i][j]].resize((cell_size, cell_size))
                transformed_grid_image.paste(tile_img, (j * cell_size, i * cell_size))

        # Save the transformed image and array
        save_grid_image_and_array(output_dir, letter, level, transformed_array, transformed_grid_image, iteration_dir)

print("Processing complete!")
