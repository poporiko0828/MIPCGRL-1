import json
import cv2
import numpy as np
from os.path import dirname, join, basename
from typing import Tuple
from scipy.spatial.distance import hamming

from pcgrllm.evaluation.base import *
from pcgrllm.utils.storage import Iteration


class HeuristicEvaluator(LevelEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Load the large ground truth alphabet data (64x64) from a JSON file
        with open(join(dirname(__file__), 'ground_truth_64x64.json'), 'r') as f:
            self.ground_truth_data = json.load(f)

    def resize_input(self, numpy_data: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize the input numpy array to the new size.
        """
        # Directly use the new_size (width, height) tuple for resizing
        resized_input = cv2.resize(numpy_data, new_size[::-1], interpolation=cv2.INTER_NEAREST)
        return np.rint(resized_input).astype(int)

    def sliding_window(self, larger_grid: np.ndarray, smaller_grid: np.ndarray,
                       show_visualization: bool = False) -> float:
        """
        Perform sliding window convolution to match the smaller grid to the larger ground truth grid and
        return the best (lowest) Hamming distance found across all positions.
        """
        best_score = float('inf')  # Initialize with a large value
        best_position = (0, 0)  # Keep track of the best match position
        larger_h, larger_w = larger_grid.shape
        smaller_h, smaller_w = smaller_grid.shape

        if smaller_h > larger_h or smaller_w > larger_w:
            return float('inf')

        # Slide the smaller grid over the larger grid
        for y in range(larger_h - smaller_h + 1):
            for x in range(larger_w - smaller_w + 1):
                # Extract the subgrid from the larger grid
                subgrid = larger_grid[y:y + smaller_h, x:x + smaller_w]

                # Calculate Hamming distance between subgrid and smaller grid
                distance = hamming(subgrid.flatten(), smaller_grid.flatten())

                # Keep track of the best (smallest) Hamming distance
                if distance < best_score:
                    best_score = distance
                    best_position = (x, y)

                # Show the visualization for the current sliding window if enabled
                if show_visualization:
                    self.visualize_sliding_window(larger_grid, subgrid, smaller_grid, (x, y), best_position, distance,
                                                  best_score)

        if best_score == float('inf'):
            print("Warning: No valid matches found during sliding window.")

        return best_score

    def visualize_sliding_window(self, larger_grid, current_subgrid, smaller_grid, current_position, best_position,
                                 current_loss, best_loss):
        """
        Visualize the sliding window comparison, showing:
        - Red rectangle around current window
        - Blue rectangle around the best match
        - Both grids side-by-side with black background (empty space) and white alphabet (occupied space).
        - Display the current loss, best loss, and window positions on the image.
        """
        grid_h, grid_w = larger_grid.shape
        small_h, small_w = smaller_grid.shape

        # Create a color version of the larger grid (convert to 3 channels for color rectangles)
        # Convert 1 (empty space) to black, 2 (alphabet) to white
        larger_grid_img = np.where(larger_grid == 1, 0, 255).astype(np.uint8)  # Black for 1, White for 2
        larger_grid_img = np.stack([larger_grid_img] * 3, axis=-1)  # Convert to 3 channels (RGB)

        # Draw a red rectangle for the current sliding window
        cv2.rectangle(larger_grid_img, (current_position[0], current_position[1]),
                      (current_position[0] + small_w, current_position[1] + small_h),
                      (0, 0, 255), 1)

        # Draw a blue rectangle for the best matching window
        cv2.rectangle(larger_grid_img, (best_position[0], best_position[1]),
                      (best_position[0] + small_w, best_position[1] + small_h),
                      (255, 0, 0), 1)

        # Resize the images for display (300x300)
        display_size = (300, 300)

        # Convert smaller and current subgrids to RGB: 1 (empty) to black, 2 (alphabet) to white
        smaller_grid_resized = np.where(smaller_grid == 1, 0, 255).astype(np.uint8)
        smaller_grid_resized = cv2.resize(smaller_grid_resized, display_size, interpolation=cv2.INTER_NEAREST)
        smaller_grid_resized = np.stack([smaller_grid_resized] * 3, axis=-1)  # Convert to 3 channels

        current_subgrid_resized = np.where(current_subgrid == 1, 0, 255).astype(np.uint8)
        current_subgrid_resized = cv2.resize(current_subgrid_resized, display_size, interpolation=cv2.INTER_NEAREST)
        current_subgrid_resized = np.stack([current_subgrid_resized] * 3, axis=-1)  # Convert to 3 channels

        # Resize the larger grid image
        larger_grid_resized = cv2.resize(larger_grid_img, display_size, interpolation=cv2.INTER_NEAREST)

        # Add text to show the current loss, best loss, current position, and best position on the larger grid
        font_scale = 0.5
        font_thickness = 1
        text_color = (255, 255, 255)

        # Ensure the text is positioned within the image boundaries
        cv2.putText(larger_grid_resized, f'Loss: {current_loss:.4f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(larger_grid_resized, f'Best Loss: {best_loss:.4f}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(larger_grid_resized, f'Current Pos: {current_position}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(larger_grid_resized, f'Best Pos: {best_position}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Combine the grids for side-by-side display
        combined_img = np.hstack([smaller_grid_resized, current_subgrid_resized, larger_grid_resized])

        # Display the combined image with the grid, rectangles, and text
        cv2.imshow("Sliding Window Visualization (Left: Input, Center: Current, Right: Map)", combined_img)
        cv2.waitKey(1)  # Add a slight delay to allow visualization

    def run(self, iteration: Iteration, target_character: str, visualize: bool = False) -> EvaluationResult:
        numpy_files = iteration.get_numpy_files()
        scales = sorted(np.array(list(range(4, 10, 1))) * 0.1,
                        reverse=True)  # Define the different scales for the ground truth

        hamming_distances = list()  # Track Hamming distances instead of similarity scores

        # Iterate over the numpy files and evaluate each
        for file in numpy_files:
            numpy_data = file.load()  # Load numpy data (larger grid)

            if numpy_data.size == 0:
                print(f"Error: Loaded numpy file is empty.")
                continue

            # Get the dimensions of the larger grid
            larger_h, larger_w = numpy_data.shape

            # Get the original ground truth grid for the target character
            original_ground_truth = np.array(self.ground_truth_data[target_character])
            ground_truth_h, ground_truth_w = original_ground_truth.shape

            # Calculate the size ratio between the larger grid and the original ground truth
            height_ratio = larger_h / ground_truth_h
            width_ratio = larger_w / ground_truth_w

            best_loss = float('inf')

            # Iterate through each scale for resizing the ground truth
            for scale in scales:
                # Calculate the new size for the ground truth based on the size of numpy_data and the scale
                scale_height = int(ground_truth_h * height_ratio * scale)
                scale_width = int(ground_truth_w * width_ratio * scale)

                # Resize the ground truth to the new dimensions
                resized_ground_truth = self.resize_input(original_ground_truth, (scale_width, scale_height))

                # Perform sliding window matching to find the best match, with optional visualization
                score = self.sliding_window(numpy_data, resized_ground_truth, show_visualization=visualize)

                # Keep track of the best score (smallest Hamming distance)
                best_loss = min(best_loss, score)

            if best_loss == float('inf'):
                print(f"Warning: No valid match found for file {file}")

            best_loss = (best_loss - 0.9) / 0.02
            hamming_distances.append(best_loss)

        # Calculate the mean similarity score as 1 minus the mean Hamming distance
        if hamming_distances:
            mean_similarity = 1 - np.mean(hamming_distances)
        else:
            mean_similarity = float('-inf')

        return EvaluationResult(task=self.task, similarity=mean_similarity, sample_size=len(numpy_files))


def run_cross_validation(evaluator, letters, base_path, visualize=False):
    """
    Run cross-validation for a set of letters, where each letter is evaluated against others.
    The results are sorted by similarity score for each target letter.
    """
    results_dict = {target_letter: [] for target_letter in
                    letters}  # Dictionary to store results for each target letter

    for target_letter in letters:
        for evaluate_letter in letters:
            # Define the path for the iteration folder
            example_path = join(base_path, evaluate_letter, 'iteration_1')

            # Load the iteration
            iteration = Iteration.from_path(path=example_path)

            # Run the evaluator with visualization enabled/disabled
            result = evaluator.run(iteration=iteration, target_character=target_letter, visualize=visualize)

            # Store the result for sorting later
            results_dict[target_letter].append((evaluate_letter, result.similarity))

    # Sort and print the results for each target letter
    for target_letter, evaluations in results_dict.items():
        # Sort evaluations by similarity score in descending order (higher is better)
        sorted_evaluations = sorted(evaluations, key=lambda x: x[1], reverse=True)

        print(f"\nSorted Results for target letter '{target_letter}':")
        for evaluate_letter, similarity in sorted_evaluations:
            print(f"Evaluated Letter: {evaluate_letter}, Similarity Score: {similarity:.4f}")


if __name__ == '__main__':
    # Initialize logger
    logger = logging.getLogger(basename(__file__))
    logger.setLevel(logging.DEBUG)

    evaluator = HeuristicEvaluator(logger=logger)

    # Define the letters and the base path for the evaluations
    letters = ['A', 'B', 'H']
    base_path = join(dirname(__file__), 'example')

    # Run the cross-validation and show sorted results with visualization enabled
    run_cross_validation(evaluator, letters, base_path, visualize=False)
