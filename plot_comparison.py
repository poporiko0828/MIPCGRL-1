import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. それぞれの実験の progress.csv への正確なパスを指定
# ※ RoBERTa側のフォルダ名は、saves/ 以下に新しくできた実際のフォルダ名に変更してください
bert_csv = "/home/hosozawa/MIPCGRL_local/MIPCGRL/saves/embed-bert_inst-None_model-nlpconv_exp-def_cls_st_mk_s-0/progress.csv"

plt.figure(figsize=(10, 6))

# 2. BERT（過去のデータ）のプロット
if os.path.exists(bert_csv):
    df_bert = pd.read_csv(bert_csv)
    # NaN（空行）を排除し、必要な列だけをプロット
    df_bert_clean = df_bert.dropna(subset=['timestep', 'Train/ep_return'])
    plt.plot(df_bert_clean['timestep'], df_bert_clean['Train/ep_return'], 
             label='BERT (Baseline)', color='#95a5a6', alpha=0.7, linestyle='--')


# 4. 論文・発表用スタイリッシュ装飾
plt.title('PCGRL Training Progress: BERT vs RoBERTa Encoder', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Timesteps', fontsize=12)
plt.ylabel('Episode Mean Return (Reward)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11, loc='lower right')

# 5. 高解像度画像として保存
output_path = 'reward_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[+] Beautiful graph successfully saved as '{output_path}'")
