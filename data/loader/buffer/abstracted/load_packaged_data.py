import numpy as np

from ..types_ import PackagedData


def load_packaged_data(buffer_path: str):
    data = np.load(buffer_path, allow_pickle=True)

    return PackagedData(
        map_pair_indices=data.get("map_pair_indices"),
        instruction_indices=data.get("instruction_indices"),
        env_map=data.get("env_map"),
        map_obs=data.get("map_obs"),
        rewards=data.get("rewards"),
        instructions=data.get("instructions"),
        reward_enums=data.get("reward_enums"),
        embeddings=data.get("embeddings"),
        conditions=data.get("conditions"),
        input_ids=data.get("embeddings"),
        attention_mask=data.get("attention_mask"),
        is_finetuned=data.get("is_finetuned"),
    )
