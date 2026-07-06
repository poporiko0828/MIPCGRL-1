import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_learning_curve():
    # 💡 ユーザーからいただいた具体的な絶対パスを指定
    csv_path = "/home/hosozawa/MIPCGRL_local/MIPCGRL/saves/embed-bert_inst-None_model-nlpconv_exp-def_cls_st_mk_s-0/progress.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ エラー: 指定されたパスに progress.csv が見つかりません:\n{csv_path}")
        return

    # 1. CSVデータの読み込み
    df = pd.read_csv(csv_path)

    # JAXの30,720歩ごとのリセットを補正し、本当の累積ステップ（Total Steps）を計算
    total_steps = []
    current_offset = 0
    prev_step = 0

    for step in df['Train/Step']:
        if pd.isna(step):
            total_steps.append(None)
            continue
            
        # ステップ数が減少、または30,720に達して次に進んだら「新しいチャンク」とみなしてオフセットを加算
        if step < prev_step:
            current_offset += 30720
            
        total_steps.append(current_offset + step)
        prev_step = step

    df['Total_Step'] = total_steps

    # データのクリーニング（プロット用に有効な行だけを抽出）
    train_data = df.dropna(subset=['Total_Step', 'Train/ep_return'])
    eval_data = df.dropna(subset=['Total_Step', 'Eval/ep_return'])

    # 2. グラフの描画設定
    plt.figure(figsize=(12, 6))
    
    # Trainの報酬推移（メインの学習）
    plt.plot(train_data['Total_Step'], train_data['Train/ep_return'], 
             label='Train Episodic Return (Mean)', color='#1f77b4', alpha=0.6, linewidth=1.5)
    
    # Evalの報酬推移（評価時の実力）
    plt.scatter(eval_data['Total_Step'], eval_data['Eval/ep_return'], 
                label='Eval Episodic Return (Mean)', color='#ff7f0e', marker='o', s=50, zorder=5)
    plt.plot(eval_data['Total_Step'], eval_data['Eval/ep_return'], 
             color='#ff7f0e', linestyle='--', alpha=0.7)

    # 3. グラフの装飾
    plt.title('Learning Progress with Instruction Changes', fontsize=14, fontweight='bold')
    plt.xlabel('Total Timesteps (Cumulative)', fontsize=12)
    plt.ylabel('Episodic Return', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc='upper left')

    plt.tight_layout()
    
    # 💡 グラフ画像も、CSVと同じディレクトリ（実証フォルダ内）に保存します
    output_dir = os.path.dirname(csv_path)
    output_img = os.path.join(output_dir, "learning_progress.png")
    
    plt.savefig(output_img, dpi=300)
    print(f"📊 グラフを正常に作成し、以下に保存しました！\n {output_img}")
    
    # GUI環境（Ubuntuのデスクトップ等）があれば画面にもポップアップ表示します
    try:
        plt.show()
    except Exception:
        print("ℹ CUI環境のため画面表示はスキップしました（画像ファイルは正しく生成されています）。")

if __name__ == "__main__":
    plot_learning_curve()