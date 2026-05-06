

def get_reward_score_paired_examples(storage: 'Storage', best_iteration_nums: list) -> str:
    """Returns a formatted reward and score summary for paired examples of the best iterations."""
    reward_score = ""

    for idx, iteration_num in enumerate(best_iteration_nums, 1):
        iteration = storage.get_iteration(iteration_num)
        if iteration:
            # Retrieve reward function path and evaluation prompt
            reward_prompt = iteration.get_reward_function_path()
            # read the reward function
            try:
                with open(reward_prompt, 'r') as f:
                    reward_prompt = f.read()
            except FileNotFoundError:
                reward_prompt = "Reward function not found."

            score_prompt = iteration.get_evaluation().to_prompt()

            # Format each example with a clear header for distinction
            reward_score += f"[Reference {idx}]:\n"
            reward_score += f"Reward Function:\n```\n{reward_prompt}\n```\n"
            reward_score += f"** Score: {score_prompt} **\n\n"

    return reward_score