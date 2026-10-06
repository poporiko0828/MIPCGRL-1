import sys
import os
import torch
import jax
import jax.numpy as jnp
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from envs.pcgrl_env import PCGRLEnv, PCGRLEnvParams, ProbEnum

TILE_CHAR_TO_INT = {'.': 0, 'W': 1, 'B': 2, 'P': 3, 'E': 4}

def main():
    dpo_model_path = "./pcgrl_dpo_model"
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    print("モデルの読み込み中...")
    tokenizer = AutoTokenizer.from_pretrained(dpo_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, dpo_model_path)
    model.eval()

    messages = [
        {"role": "system", "content": "You are an expert PCGRL dungeon map generator. Output ONLY the raw 16x16 ASCII grid containing '.', 'W', 'B', 'P', and 'E'. Do NOT include any introduction, explanations, or code blocks."},
        {"role": "user", "content": "Generate a 16x16 dungeon map with a clear path from Player to Exit."}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.3, pad_token_id=tokenizer.pad_token_id)
    
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    print("\n--- 生成されたマップ ---")
    print(generated_text)
    print("------------------------\n")

    # タイルカウントの詳細ログ
    p_count = generated_text.count('P')
    e_count = generated_text.count('E')
    w_count = generated_text.count('W')
    dot_count = generated_text.count('.')
    print(f"Pの数: {p_count}, Eの数: {e_count}, W(壁)の数: {w_count}, .(床)の数: {dot_count}")

    # PCGRL内部ロジックでのテスト
    lines = [line.strip() for line in generated_text.strip().split("\n") if line.strip()]
    if len(lines) == 16 and all(len(line) == 16 for line in lines):
        grid = [[TILE_CHAR_TO_INT.get(c, -1) for c in line] for line in lines]
        env_map_jax = jnp.array(grid, dtype=jnp.int32)
        
        params = PCGRLEnvParams(map_shape=(16, 16), problem=ProbEnum.DUNGEON)
        env = PCGRLEnv(params)
        
        try:
            key = jax.random.PRNGKey(0)
            prob_state = env.prob.reset(env_map_jax, key)[1]
            path_coords = env.prob.get_path_coords(env_map=env_map_jax, prob_state=prob_state)
            print(f"探索結果 path_coords: {path_coords}")
        except Exception as e:
            print(f"PCGRL探索中に例外が発生しました: {e}")
    else:
        print("マップサイズが 16x16 ではありませんでした。")

if __name__ == "__main__":
    main()