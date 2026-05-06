import json
import os
import re
from os.path import abspath, join, basename, isdir, isfile
from glob import glob
from typing import List, Optional
import numpy as np
from PIL import Image

from pcgrllm.evaluation import EvaluationResult
from pcgrllm.utils.graph import NodeInfo

# Constants
ITERATION_PREFIX: str = 'iteration_'
INFERENCE_DIR: str = 'inference'
TRAIN_DIR: str = 'train'
IMAGE_DIR: str = 'images'
NUMPY_DIR: str = 'numpy'


class ImageResource:
    """Represents an image resource."""

    def __init__(self, path: str) -> None:
        self.path: str = path

    def __str__(self) -> str:
        return f'ImageResource(path={self.path})'

    def load(self) -> Image.Image:
        """Loads the image as a PIL Image object."""
        return Image.open(self.path)


class NumpyResource:
    """Represents a numpy file resource."""

    def __init__(self, path: str) -> None:
        self.path: str = path

    def __str__(self) -> str:
        return f'NumpyResource(path={self.path})'

    def load(self, dtype=np.uint16) -> np.ndarray:
        """Loads the numpy file as a NumPy array."""
        return np.load(self.path, allow_pickle=True).astype(dtype)


class FeedbackResource:
    """Represents a feedback resource."""

    def __init__(self, path: str) -> None:
        self.path: str = path

    def __str__(self) -> str:
        return f'FeedbackResource(path={self.path})'

    def load(self) -> str:
        """Loads the feedback as a string."""
        with open(self.path, 'r') as f:
            return f.read()


class Iteration:
    """Represents an iteration containing images and numpy files inside the inference folder."""

    def __init__(self, iteration_num: int, root_path: str) -> None:
        self.iteration_num: int = iteration_num
        self.root_path: str = root_path
        self.iterative_mode = True

        self.reward_function_path = self.get_reward_function_path()

    def __str__(self) -> str:
        fitness = None

        if self.get_evaluation():
            fitness = self.get_evaluation().total

        return (
            f"Iteration {self.iteration_num}\n"
            f"\tImages: {len(self.get_images())} files\n"
            f"\tNumpy Files: {len(self.get_numpy_files())} files\n"
            f"\tFitness: {fitness}"
        )

    @staticmethod
    def from_path(path: str) -> Optional['Iteration']:
        """Creates an Iteration object from the given path."""
        try:
            iteration_num = int(basename(path).split('_')[-1])
        except:
            iteration_num = None
        return Iteration(iteration_num, path)

    def get_path(self) -> str:
        return self.root_path

    def get_train_dir(self) -> str:
        return join(self.root_path, TRAIN_DIR if self.iterative_mode else '')

    def get_inference_dir(self) -> str:
        return join(self.root_path, INFERENCE_DIR)

    def get_image_dir(self, train: bool = False) -> str:
        if train:
            return join(self.get_train_dir(), IMAGE_DIR)
        else:
            return join(self.get_inference_dir(), IMAGE_DIR)

    def get_numpy_dir(self, train: bool = False) -> str:
        if train:
            return join(self.get_train_dir(), NUMPY_DIR)
        else:
            return join(self.get_inference_dir(), NUMPY_DIR)

    def get_images(self, train: bool = False, step_filter: str = None) -> List[ImageResource]:
        image_dir = self.get_image_dir(train)
        if step_filter:
            image_paths = glob(join(image_dir, f'*{step_filter}*.png'))
        else:
            image_paths = glob(join(image_dir, '*.png'))
        return [ImageResource(path) for path in image_paths]

    def get_numpy_files(self, train: bool = False, step_filter: str = None) -> List[NumpyResource]:
        numpy_dir = self.get_numpy_dir(train)
        if step_filter:
            numpy_paths = glob(join(numpy_dir, f'*{step_filter}*.npy'))
        else:
            numpy_paths = glob(join(numpy_dir, '*.npy'))
        return [NumpyResource(path) for path in numpy_paths]

    def get_reward_function_path(self) -> Optional[str]:
        pattern = re.compile(rf"reward_outer_{self.iteration_num}_inner_\d+\.py")

        for file_name in os.listdir(self.root_path):
            if pattern.match(file_name):
                reward_path = join(self.root_path, file_name)
                if isfile(reward_path):
                    return reward_path

        return None

    def get_evaluation_dir(self) -> str:
        eval_path = join(self.root_path, 'evaluation.json')

        try:
            eval_json = json.load(open(eval_path, 'r'))
        except:
            return None
        return EvaluationResult.from_dict(eval_json)

    def get_evaluation(self) -> Optional[EvaluationResult]:
        eval_path = join(self.root_path, 'evaluation.json')

        try:
            eval_json = json.load(open(eval_path, 'r'))
        except:
            return None
        return EvaluationResult.from_dict(eval_json)

    def get_feedback_path(self) -> Optional[str]:
        feedback_path = join(self.root_path, 'feedback.txt')
        if not isdir(feedback_path):
            return feedback_path
        return None

    def get_feedback(self) -> Optional[FeedbackResource]:
        feedback_path = self.get_feedback_path()
        if not isdir(feedback_path):
            return FeedbackResource(feedback_path)
        return None

    def get_node(self) -> Optional[NodeInfo]:
        nodeinfo_path = join(self.root_path, 'node.json')

        try:
            nodeinfo_json = json.load(open(nodeinfo_path, 'r'))
        except:
            return None
        return NodeInfo.from_dict(nodeinfo_json)

    def set_node(self, node: NodeInfo) -> None:
        nodeinfo_path = join(self.root_path, 'node.json')
        with open(nodeinfo_path, 'w') as f:
            json.dump(node.to_dict(), f)

    def set_evaluation_context(self, context: dict) -> None:
        eval_path = join(self.root_path, 'evaluation.context.json')

        with open(eval_path, 'w') as f:
            json.dump(context, f)

    @property
    def node(self):
        return self.get_node()


class Storage:
    """Manages multiple iterations and their resources."""

    def __init__(self, path: str) -> None:
        self.root_path: str = abspath(path)
        self.reward_functions_ws_path = join(self.root_path, 'reward_functions')

    def __str__(self) -> str:
        iteration_details = "\n".join(str(iteration) for iteration in self.get_iterations())
        return f"(Storage: {self.root_path})\n{iteration_details}"

    def get_reward_functions_ws_path(self, iteration_num: int) -> str:
        return join(self.reward_functions_ws_path, f'reward_outer_{iteration_num}_inner_1')

    def get_iterations(self) -> List[Iteration]:
        iteration_nums = self._get_iteration_nums()
        return [
            Iteration(num, join(self.root_path, f"{ITERATION_PREFIX}{num}"))
            for num in iteration_nums
        ]

    def _get_iteration_nums(self) -> List[int]:
        iteration_regex = join(self.root_path, f'{ITERATION_PREFIX}*')
        iter_dirs = glob(iteration_regex)
        iter_dirs = [basename(d) for d in iter_dirs if isdir(d)]
        return [int(d.split('_')[-1]) for d in iter_dirs]

    def get_iteration(self, iteration_num: int) -> Optional[Iteration]:
        iteration_path = join(self.root_path, f"{ITERATION_PREFIX}{iteration_num}")
        if isdir(iteration_path):
            return Iteration(iteration_num, iteration_path)
        return None

    def set_auxiliary_prompt(self, iteration_num: int, prompt: str) -> str:
        # check if dir exists
        reward_functions_ws_path = self.get_reward_functions_ws_path(iteration_num)

        if not isdir(reward_functions_ws_path):
            os.makedirs(reward_functions_ws_path, exist_ok=True)

        aux_prompt_path = join(reward_functions_ws_path, 'auxiliary_prompt.txt')
        with open(aux_prompt_path, 'w') as f:
            f.write(prompt)
        return aux_prompt_path

    def get_auxiliary_prompt_path(self, iteration_num: int) -> Optional[str]:
        aux_prompt_path = join(self.get_reward_functions_ws_path(iteration_num), 'auxiliary_prompt.txt')
        if not isdir(aux_prompt_path):
            return aux_prompt_path
        return None

    def get_auxiliary_prompt(self, iteration_num: int) -> str:
        aux_prompt_path = self.get_auxiliary_prompt_path()
        if not isdir(aux_prompt_path):
            with open(aux_prompt_path, 'r') as f:
                return f.read()
        return ""


def get_first_file(resource_list: List[ImageResource]) -> Optional[str]:
    """Returns the path of the first file from a resource list."""
    if resource_list:
        return resource_list[0].path
    return None


def main():
    # Set the test path
    test_path = "your_test_path"

    # Create Storage instance
    storage = Storage(test_path)

    # Iterate over each Iteration and get the first image and numpy file
    iterations = storage.get_iterations()
    for iteration in iterations:
        print(f"\nIteration {iteration.iteration_num}")

        # Get the first image file
        first_image_path = get_first_file(iteration.get_images())
        if first_image_path:
            print(f"First Image: {first_image_path}")
        else:
            print("No images found.")

        # Get the first numpy file
        first_numpy_path = get_first_file(iteration.get_numpy_files())
        if first_numpy_path:
            print(f"First Numpy File: {first_numpy_path}")
        else:
            print("No numpy files found.")

    # Print Storage summary
    print("\nStorage Summary:")
    print(storage)


if __name__ == "__main__":
    main()
