# ==============================================================================
# generate_map.py
# ==============================================================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SYSTEM_PROMPT = (
    "You are an expert PCGRL dungeon map generator. "
    "Output ONLY the raw 16x16 ASCII grid containing '.', 'W', 'B', 'P', and 'E'. "
    "Do NOT include any introduction, explanations, or code blocks."
)

def main():
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    sft_model_path = "./pcgrl_sft_model"

    print("SFTモデルとトークナイザーを読み込んでいます...")
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
    
    # ベースモデルのロード
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRAアダプターの適用
    model = PeftModel.from_pretrained(base_model, sft_model_path)
    model.eval()

    # テスト用プロンプト（データセットに含まれるような条件を指定）
    user_prompt = "Generate a 16x16 dungeon map with a clear path from Player to Exit."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    print("\nマップを生成中...\n" + "="*40)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )

    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(generated_text)
    print("="*40)

if __name__ == "__main__":
    main()