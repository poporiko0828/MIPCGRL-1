import json, re
from os import path
from os.path import dirname, join, basename

from pcgrllm.evaluation.base import *
from pcgrllm.llm_client.llm import UnifiedLLMClient, ChatContext
from pcgrllm.scenario_preset import ScenarioPreset
from pcgrllm.utils.storage import Iteration

class LLMEvaluator(LevelEvaluator):
    def __init__(self, gpt_model: str, seed: int, n_generation_trials: int = 1, **kwargs):
        super().__init__(**kwargs)

        self.prompt_template_path = path.join(path.dirname(path.abspath(__file__)), '..', 'prompt', self.task, 'evaluation', 'llm_evaluation.txt')
        with open(self.prompt_template_path, 'r') as f:
            self.prompt_template = f.read()

        self.client = UnifiedLLMClient()
        self.gpt_model = gpt_model
        self.n_generation_trials = n_generation_trials
        self.seed = seed

    def run(self, iteration: Iteration, target_character: str, use_train: bool = False, step_filter=None) -> EvaluationResult:
        # if the target_character is not alphabet, return 0

        if use_train is True:
            self.logging("LLM evaluator is only for inference. Use ViT evaluator for training.")
            exit(1)

        numpy_files = iteration.get_numpy_files(step_filter=step_filter)
        numpy_str = ''
        for idx, numpy_file in enumerate(numpy_files):
            numpy_data = numpy_file.load()
            numpy_str += f'Level Id {idx + 1}:\n'
            numpy_str += f'{str(numpy_data)}\n\n'

        evaluation_criteria = f'Level looks like "{target_character}"'
        prompt = self.prompt_template.replace('{evaluation_criteria}', evaluation_criteria)
        prompt = prompt.replace('{content}', numpy_str)

        messages = [
            {"role": "user", "content": prompt}
        ]

        attempts = 0

        while attempts < self.n_generation_trials:
            try:
                ctx = ChatContext()
                response = self.client.call_model(ctx, messages, model=self.gpt_model, seed=self.seed, temperature=0)[0]
                response, context = response

                # Extract JSON using regex and attempt to parse it
                json_match = re.search(r'\{.*?\}', response, re.DOTALL)

                if json_match:
                    match = json_match.group()

                    try:
                        parsed_response = json.loads(match)

                        iteration.set_evaluation_context(context.to_json())

                        return EvaluationResult(task=self.task,
                                                **parsed_response,
                                                sample_size=len(numpy_files))
                    except Exception as e:
                        self.logging(f"Model call failed on attempt {attempts + 1}: {e}")
                        attempts += 1
                        continue
                else:
                    self.logging(f"No JSON found in response on attempt {attempts + 1}")
                    attempts += 1
            except Exception as e:
                self.logging(f"Model call failed on attempt {attempts + 1}: {e}")
                attempts += 1

        # If max attempts are reached, print an error message and exit
        self.logging("Maximum attempts reached. Exiting with error.")
        print("Error: Unable to parse response from model after maximum attempts.")
        exit(1)


def run_cross_validation(evaluator, letters, base_path):
    """
    Run cross-validation for a set of letters, where each letter is evaluated against others.
    The results are sorted by similarity score for each target letter.
    """
    results_dict = {target_letter: [] for target_letter in letters}  # Dictionary to store results for each target letter

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

    # Alphabet Task
    # evaluator = LLMEvaluator(logger=logger, gpt_model='gpt-4o', seed=3)
    #
    # # Define the letters and the base path for the evaluations
    # letters = ['A', 'B']
    # base_path = join(dirname(__file__), 'example')
    #
    # # Run the cross-validation and show sorted results with visualization enabled
    # run_cross_validation(evaluator, letters, base_path)



    # Scenario Task
    evaluator = LLMEvaluator(task=TaskType.Scenario, logger=logger, gpt_model='gpt-4o', seed=3)
    iteration = Iteration.from_path(join(dirname(__file__), 'example', 'scenario_1', 'iteration_1'))
    scenario_num = '1'
    scenario = ScenarioPreset().scenarios[scenario_num]

    result = evaluator.run(iteration=iteration, target_character=scenario.prompt)
    print(result)
