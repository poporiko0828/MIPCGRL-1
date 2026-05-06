from os.path import dirname, join, basename

from PIL import Image, ImageEnhance, ImageOps
from scipy.spatial.distance import cosine

from pcgrllm.evaluation.base import *
from pcgrllm.utils.storage import Iteration


class ViTEvaluator(LevelEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, iteration: Iteration, target_character: str, use_train: bool = False,
            step_filter=None) -> EvaluationResult:
        # if the target_chracater is not alphabet, return 0
        if len(target_character) >= 2 or (not target_character.isalpha()):
            return EvaluationResult(similarity=0, diversity=0, sample_size=0)

        target_character = target_character.upper()

        image_files = iteration.get_images(train=use_train, step_filter=step_filter)

        trials = [image_file.path for image_file in image_files]
        similarity_rate = 0
        diversity_rate = 0
        vectorized_trials = []
        if trials:
            total_trials = len(trials)
            # similarity
            for file_path in trials:
                raw_result = self.predict(file_path)
                target_prob = self.search('label', target_character, raw_result)
                vectorized_logits = self.vectorize(raw_result)
                vectorized_trials.append({
                    'trial': file_path,
                    'vector': vectorized_logits
                })
                similarity_rate += target_prob[0]['softmax_prob']
            similarity = similarity_rate / total_trials

            # diversity
            all_pairs = self.generate_cartesian_product(vectorized_trials, vectorized_trials)
            for pair in all_pairs:
                distance = cosine(pair[0]['vector'], pair[1]['vector'])
                diversity_rate += distance

            if total_trials > 1:
                diversity = diversity_rate / (total_trials * (total_trials - 1) / 2)
            else:
                diversity = 0

            # return similarity, diversity
            return EvaluationResult(task=self.task, similarity=similarity, diversity=diversity,
                                    sample_size=total_trials)
        else:
            print("No valid images found in the source folder.")
            return EvaluationResult(task=self.task, similarity=float('-inf'), diversity=float('-inf'), sample_size=0)

    def predict(self, file_path: str):
        image = Image.open(file_path).convert("RGBA")
        pixels = image.load()

        for y in range(image.height):
            for x in range(image.width):
                r, g, b, a = pixels[x, y]
                if (r > 100 and g < 100 and b < 100) or (r > 100 and g > 50 and g < 150 and b < 100):
                    pixels[x, y] = (255, 255, 255, a)

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        image = image.convert("RGB")
        # image.show()

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        softmax_outputs = torch.nn.Softmax(dim=-1)(logits)

        output_list = []
        for i in range(26):
            char = chr(65 + i)
            target_prob = softmax_outputs[0][i]
            output_list.append({
                'label': char,
                'softmax_prob': target_prob.item()
            })
        return output_list

    def vectorize(self, similarity_logits: list):
        similarity_logits.sort(key=lambda x: x['label'])
        vectorized_logits = []
        for i in range(len(similarity_logits)):
            vectorized_logits.append(similarity_logits[i]['softmax_prob'])

        return vectorized_logits

    def generate_cartesian_product(self, arr1: list, arr2: list):
        lst = []
        for i in range(len(arr1)):
            for j in range(i + 1, len(arr2)):
                lst.append((arr1[i], arr2[j]))
        return lst

    def search(self, key, val, arr):
        return [el for el in arr if el[key] == val]


def run_cross_validation(evaluator, letters, base_path):
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
            result = evaluator.run(iteration=iteration, target_character=target_letter)

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

    evaluator = ViTEvaluator(logger=logger)

    # Define the letters and the base path for the evaluations
    letters = ['A', 'B', 'C', 'D', 'E']
    base_path = join(dirname(__file__), 'example')

    # Run the cross-validation and show sorted results with visualization enabled
    run_cross_validation(evaluator, letters, base_path)
