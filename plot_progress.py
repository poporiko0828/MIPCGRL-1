import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_advanced_learning_curve():
    csv_path = "/home/hosozawa/MIPCGRL_local/MIPCGRL/saves/embed-bert_inst-None_model-nlpconv_exp-def_cls_st_mk_s-0/progress.csv"

    if not os.path.exists(csv_path):
        print(" ❌ エラー: progress.csv が見つかりません")
        return

    df = pd.read_csv(csv_path)

    # 累積ステップ計算
    total_steps = []
    current_offset = 0
    prev_step = 0

    for step in df["Train/Step"]:
        if pd.isna(step):
            total_steps.append(current_offset + prev_step)
            continue
        if step < prev_step:
            current_offset += 30720
        total_steps.append(current_offset + step)
        prev_step = step

    df["Total_Step"] = total_steps

    train_data = df.dropna(subset=["Total_Step", "Train/ep_return"])
    eval_data = df.dropna(subset=["Total_Step", "Eval/ep_return"])
    loss_cols = [col for col in df.columns if col.startswith("Loss")]

    # 🎓 文字特大化 ＆ 論文用スタイル設定
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 20,  # 軸ラベル (Total Steps, Episodic Return, Loss)
        "xtick.labelsize": 16,  # x軸目盛り数字
        "ytick.labelsize": 16,  # y軸目盛り数字
        "legend.fontsize": 15,  # 凡例テキスト
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    # ==================== 上段: 報酬の推移 ====================
    ax1.plot(
        train_data["Total_Step"],
        train_data["Train/ep_return"],
        label="Train Ep Return (Mean)",
        color="#1f77b4",
        linewidth=3.0,
    )

    ax1.fill_between(
        train_data["Total_Step"],
        train_data["Train/ep_return_min"],
        train_data["Train/ep_return_max"],
        color="#1f77b4",
        alpha=0.18,
        label="Train Ep Return (Range)",
    )

    ax1.scatter(
        eval_data["Total_Step"],
        eval_data["Eval/ep_return"],
        label="Eval Ep Return (Test)",
        color="#d62728",
        s=130,
        zorder=5,
    )

    ax1.set_ylabel("Episodic Return", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.7, linewidth=1.3)

    # 💡 凡例を横一列 (ncol=3) にしてグラフ枠のすぐ上に配置
    ax1.set_ylim(-3, 16.5)
    ax1.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=3,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#cccccc",
    )
    ax1.tick_params(direction="in", length=6, width=1.5)

    # ==================== 下段: Loss の推移 ====================
    loss_colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    for i, col in enumerate(loss_cols):
        loss_data = df.dropna(subset=["Total_Step", col])
        if not loss_data.empty:
            ax2.plot(
                loss_data["Total_Step"],
                loss_data[col],
                label=f"Training Loss ({col})",
                color=loss_colors[i % len(loss_colors)],
                linestyle="--",
                marker="o",
                linewidth=2.5,
                markersize=8,
            )

    ax2.set_xlabel("Total Steps", fontweight="bold")
    ax2.set_ylabel("Loss", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.7, linewidth=1.3)

    ax2.set_ylim(-5, 90)
    ax2.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#cccccc",
    )
    ax2.tick_params(direction="in", length=6, width=1.5)

    # x軸右下の 1e6 表記も拡大・太字化
    ax2.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")
    ax2.xaxis.get_offset_text().set_fontsize(16)
    ax2.xaxis.get_offset_text().set_weight("bold")

    plt.tight_layout()

    output_dir = os.path.dirname(csv_path)
    output_img = os.path.join(output_dir, "change.png")
    plt.savefig(output_img, dpi=300, bbox_inches="tight")
    print(f" 修正済みグラフを保存しました:\n {output_img}")


if __name__ == "__main__":
    plot_advanced_learning_curve()