# ==============================================================================
# evaluate_models.py (PCGRLフレームワーク連携 & Base vs DPO 比較版)
# ==============================================================================
import sys
import os
import torch
import numpy as np
import jax
import jax.numpy as jnp
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from collections import deque

# MIPCGRL モジュールのインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from envs.pcgrl_env import PCGRLEnv, PCGRLEnvParams, ProbEnum

TILE_CHAR_TO_INT = {'.': 0, 'W': 1, 'B': 2, 'P': 3, 'E': 4}

SYSTEM_PROMPT = (
    "You are an expert PCGRL dungeon map generator. "
    "Output ONLY the raw 16x16 ASCII grid containing '.', 'W', 'B', 'P', and 'E'. "
    "Do NOT include any introduction, explanations, or code blocks."
)

def ascii_to_jax_map(ascii_text: str):
    """ASCII文字を16x16に補正・整形した上で PCGRL が扱える JAX 配列に変換"""
    lines = [line.strip() for line in ascii_text.strip().split("\n") if line.strip()]
    if not lines:
        return None
        
    # 16行に正規化 (足りなければ拡張、多すぎれば切り捨て)
    if len(lines) < 16:
        lines.extend(["." * 16] * (16 - len(lines)))
    lines = lines[:16]
    
    sanitized_grid = []
    for line in lines:
        # 各行を厳密に16文字に正規化
        if len(line) < 16:
            line = line + "." * (16 - len(line))
        else:
            line = line[:16]
            
        row = [TILE_CHAR_TO_INT.get(char, 0) for char in line]
        sanitized_grid.append(row)
        
    return jnp.array(sanitized_grid, dtype=jnp.int32)

def evaluate_with_pcgrl(env, env_map_jax):
    """
    PCGRLラッパーの環境差分を回避し、生成されたマップに対して直接BFSで解法経路を判定する
    """
    grid = np.array(env_map_jax)
    
    # P (3) と E (4) の位置を検索
    p_positions = np.argwhere(grid == TILE_CHAR_TO_INT['P'])
    e_positions = np.argwhere(grid == TILE_CHAR_TO_INT['E'])
    
    if len(p_positions) != 1 or len(e_positions) != 1:
        return False, 0
        
    start = tuple(p_positions[0])  # (row, col)
    goal = tuple(e_positions[0])   # (row, col)
    
    # BFS による最短経路探索
    queue = deque([(start[0], start[1], 0)])
    visited = {start}
    
    # 通行不可能なタイル: 壁 (TILE_CHAR_TO_INT['W'])
    wall_id = TILE_CHAR_TO_INT['W']
    
    while queue:
        r, c, dist = queue.popleft()
        
        if (r, c) == goal:
            return True, dist  # 到達可能！歩数を返す
            
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 16 and 0 <= nc < 16:
                if (nr, nc) not in visited and grid[nr, nc] != wall_id:
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))
                    
    return False, 0  # 壁に阻まれてゴールに辿り着けない

def run_evaluation(model, tokenizer, env, model_name="Model", num_samples=30):
    user_prompt = "Generate a 16x16 dungeon map. Strict rules: Exactly ONE 'P' (Player), exactly ONE 'E' (Exit), filled with '.' (path), 'W' (wall), and 'B' (monster). Ensure P and E are placed."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    valid_format_count = 0
    solvable_count = 0
    path_lengths = []

    print(f"\n--- {model_name} の評価中 ({num_samples}件生成) ---")
    for i in tqdm(range(num_samples)):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        env_map_jax = ascii_to_jax_map(generated_text)
        
        # evaluate_models.py の 67行目付近（generated_text 取得後）に追加
        if i < 3:  # 最初の3件だけ出力してみる
            print(f"\n--- 生成サンプル {i+1} ---")
            print(generated_text)
            print("--------------------")

        if env_map_jax is not None:
            valid_format_count += 1
            is_solvable, path_len = evaluate_with_pcgrl(env, env_map_jax)
            if is_solvable:
                solvable_count += 1
                path_lengths.append(path_len)

    fmt_rate = (valid_format_count / num_samples) * 100
    solvable_rate = (solvable_count / num_samples) * 100
    avg_path_len = np.mean(path_lengths) if path_lengths else 0

    return {
        "fmt_rate": fmt_rate,
        "solvable_rate": solvable_rate,
        "avg_path_len": avg_path_len,
        "valid_count": valid_format_count,
        "solvable_count": solvable_count
    }

def main():
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    dpo_model_path = "./pcgrl_dpo_model"
    num_samples = 30  # サンプル数 (必要に応じて 50 や 100 に増やせます)

    # PCGRL環境の初期化
    params = PCGRLEnvParams(map_shape=(16, 16), problem=ProbEnum.DUNGEON)
    env = PCGRLEnv(params)

    # 1. Baseモデル (DPO適用前) の評価
    print("Baseモデルを読み込んでいます...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    base_model.eval()
    base_results = run_evaluation(base_model, tokenizer, env, model_name="Base Model (DPO前)", num_samples=num_samples)

    # 2. DPOモデル (DPO適用後) の評価
    print("\nDPOモデルを読み込んでいます...")
    dpo_tokenizer = AutoTokenizer.from_pretrained(dpo_model_path, trust_remote_code=True)
    dpo_model = PeftModel.from_pretrained(base_model, dpo_model_path)
    dpo_model.eval()
    dpo_results = run_evaluation(dpo_model, dpo_tokenizer, env, model_name="DPO Model (DPO後)", num_samples=num_samples)

    # 3. 比較結果の表示
    print("\n========================================================")
    print(f"               評価比較結果 (各 {num_samples} 件)")
    print("========================================================")
    print(f"【フォーマット維持率】")
    print(f"  - Base Model : {base_results['fmt_rate']:.1f}% ({base_results['valid_count']}/{num_samples})")
    print(f"  - DPO Model  : {dpo_results['fmt_rate']:.1f}% ({dpo_results['valid_count']}/{num_samples})")
    print("--------------------------------------------------------")
    print(f"【到達可能（クリア可能）率】")
    print(f"  - Base Model : {base_results['solvable_rate']:.1f}% ({base_results['solvable_count']}/{num_samples})")
    print(f"  - DPO Model  : {dpo_results['solvable_rate']:.1f}% ({dpo_results['solvable_count']}/{num_samples})")
    print("--------------------------------------------------------")
    print(f"【解法ルートの平均歩数】")
    print(f"  - Base Model : {base_results['avg_path_len']:.2f} ステップ")
    print(f"  - DPO Model  : {dpo_results['avg_path_len']:.2f} ステップ")
    print("========================================================\n")

if __name__ == "__main__":
    main()