import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_advanced_learning_curve():
    csv_path = "/home/hosozawa/MIPCGRL_local/MIPCGRL/saves/embed-bert_inst-None_model-nlpconv_exp-def_cls_st_mk_s-0/progress.csv"
    
    if not os.path.exists(csv_path):
        print(f" ❌ エラー: progress.csv が見つかりません")
        return

    df = pd.read_csv(csv_path)

    # ==================== 💡 累積ステップの計算を完全適応化 ====================
    total_steps = []
    current_offset = 0
    prev_step = 0

    # Train/Step が空（Eval行など）の場合は、直前のステップ数を引き継ぐ
    for step in df['Train/Step']:
        if pd.isna(step):
            # Train/Stepがなくても、その行が30720のタイミング（Eval）なら直前の値をベースにする
            total_steps.append(current_offset + prev_step)
            continue
        if step < prev_step:
            current_offset += 30720
        total_steps.append(current_offset + step)
        prev_step = step

    df['Total_Step'] = total_steps

    # ==================== 🛠️ 全ステップのプロット用データを抽出 ====================
    # 欠損値を除外してプロット用の変数を定義
    train_data = df.dropna(subset=['Total_Step', 'Train/ep_return'])
    eval_data = df.dropna(subset=['Total_Step', 'Eval/ep_return'])
    
    if train_data.empty and eval_data.empty:
        print(" ⚠️ 警告: プロットできるデータが存在しません。")
        return

    # 2段のグラフを作成（ax2は下段用ですが、現状は上段 ax1 のみ描画処理があります）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # ==================== 上段: 報酬の推移と信頼区間 ====================
    # ① Mean のプロット
    ax1.plot(train_data['Total_Step'], train_data['Train/ep_return'], 
             label='Train Ep Return (Mean)', color='#1f77b4', linewidth=2)
    
    # Max-Min の範囲を薄い青色で塗りつぶす
    ax1.fill_between(train_data['Total_Step'], 
                     train_data['Train/ep_return_min'], 
                     train_data['Train/ep_return_max'], 
                     color='#1f77b4', alpha=0.15, label='Train Ep Return (Max/Min Range)')
    
    # ② Eval（テスト評価）のプロット
    ax1.scatter(eval_data['Total_Step'], eval_data['Eval/ep_return'], 
                label='Eval Ep Return (Test)', color='#d62728',  s=80, zorder=5)
    
    ax1.set_title('Advanced Learning Analytics (Performance & Training Loss) - All Steps', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Episodic Return', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=10, loc='upper left')

    #ax2.set_xlim(0, 2150400) # 自動スケールのためコメントアウトのままにします

    plt.tight_layout()
    
    output_dir = os.path.dirname(csv_path)
    output_img = os.path.join(output_dir, "change.png")
    plt.savefig(output_img, dpi=300)
    print(f" 詳細分析グラフ（報酬 ＆ Loss点描版）を保存しました:\n {output_img}")
    
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    plot_advanced_learning_curve()