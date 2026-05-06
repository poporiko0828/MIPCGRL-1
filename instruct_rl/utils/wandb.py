import wandb

def get_image_url(entity: str, project: str, run, reward_i, seed):
    key = f'Image/reward_{reward_i}/seed_{seed}'
    history = run.history(keys=[key])
    image_path = history[key][0]['path']

    image_url = f"https://api.wandb.ai/files/{entity}/{project}/{run.id}/{image_path}"
    return image_url


def get_run_by_id(entity: str, project: str, run_id: str):
    api = wandb.Api(timeout=600)
    return api.run(f"{entity}/{project}/{run_id}")