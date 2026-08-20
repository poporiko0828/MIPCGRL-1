# reward_model.py
import numpy as np
import os

class SimpleRewardModel:
    def __init__(self, mode="dummy"):
        self.mode = mode
        self.weight_file = "rm_weights.npy"
        # 💡 減衰率 (1回の推論ごとに重みをどのくらい維持するか: 0.995なら徐々に0へ戻る)
        self.decay_rate = 0.995 

    def predict(self, env_map: np.ndarray) -> np.ndarray:
        batch_size = env_map.shape[0]

        if os.path.exists(self.weight_file):
            try:
                weight = float(np.load(self.weight_file))
            except Exception:
                weight = 0.0
        else:
            weight = 0.0

        if self.mode == "dummy":
            # 基本のマップスコアを計算
            scores = np.mean(env_map, axis=(1, 2)) * 0.1 * weight
            scores = np.clip(scores, -1.0, 1.0)

            # ========================================================
            # 💡 減衰 (Decay) 処理:
            # 呼び出されるたびに重みを衰退させ、次回以降は自動で 0 (標準) に近づける
            # ========================================================
            new_weight = weight * self.decay_rate
            # 変化が非常に小さくなったら完全に 0 にリセット
            if abs(new_weight) < 0.01:
                new_weight = 0.0
                
            np.save(self.weight_file, float(new_weight))
            # ========================================================

            return scores.astype(np.float32)

        return np.zeros((batch_size,), dtype=np.float32)

reward_model_instance = SimpleRewardModel()

def get_human_reward(env_map: np.ndarray) -> np.ndarray:
    return reward_model_instance.predict(env_map)