# Generate 64x64 binary grids for each letter (A-Z) with a larger font size for more prominent letters
from PIL import ImageFont, Image, ImageDraw
import numpy as np
import json

# Use a larger font size to fill the 64x64 grid
font_size = 56  # Larger font size to make the letters bigger
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ground_truth = {}

try:
    # Use a standard available font, such as DejaVuSans
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
except IOError:
    # Fallback to default font if not available
    font = ImageFont.load_default()

# Create the grid for each character (A-Z)
for char in alphabet:
    img = Image.new('L', (64, 64), color=1)  # Background color is 1
    draw = ImageDraw.Draw(img)

    # Calculate the position to center the letter
    w, h = draw.textsize(char, font=font)
    position = ((64 - w) // 2, (64 - h) // 2)

    # Draw the letter in the center of the grid using the number 2
    draw.text(position, char, font=font, fill=2)

    # Convert image to numpy array and store the result
    grid = np.array(img)
    ground_truth[char] = grid.tolist()

# Save the updated ground_truth dictionary to a JSON file
ground_truth_path = 'ground_truth_64x64.json'
with open(ground_truth_path, 'w') as f:
    json.dump(ground_truth, f, indent=4)
