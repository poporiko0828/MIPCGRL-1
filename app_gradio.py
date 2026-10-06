import glob
import os
import gradio as gr
import numpy as np
from PIL import Image

IMAGE_DIR = "/home/hosozawa/MIPCGRL_local/MIPCGRL/saves/embed-bert_inst-None_model-nlpconv_exp-def_cls_st_mk_s-0/images"
DATASET_PATH = "human_dataset.npz"

# 状態管理変数
image_list = []  # ディレクトリ内の画像パス一覧 (生成日時順)
current_index = -1  # 現在表示中の画像インデックス
evaluated_paths = set()  # 評価済み画像パスの集合


def update_image_list():
    """ディレクトリ内の全画像をタイムスタンプ順（古い→新しい）で更新する"""
    global image_list
    paths = glob.glob(os.path.join(IMAGE_DIR, "*.png")) + glob.glob(
        os.path.join(IMAGE_DIR, "*.jpg")
    )
    # 古い順にソート（末尾が最新画像）
    image_list = sorted(paths, key=os.path.getmtime)


def load_image_at_index(index):
    """指定されたインデックスの画像を読み込んでステータスを更新する"""
    global image_list, current_index, evaluated_paths

    update_image_list()

    if not image_list:
        current_index = -1
        return None, "❌ 画像が見つかりません。"

    # インデックスの範囲ガード
    current_index = max(0, min(index, len(image_list) - 1))
    target_path = image_list[current_index]

    image = Image.open(target_path)
    filename = os.path.basename(target_path)

    # 評価済みかどうかの判定
    if target_path in evaluated_paths:
        status = f"⚠️ 評価済み ({current_index + 1}/{len(image_list)}): {filename}"
    else:
        status = (
            f"👀 未評価 ({current_index + 1}/{len(image_list)}): {filename}"
        )

    return image, status


def navigate(direction):
    """前後の画像へ移動（direction: -1 で前へ, 1 で次へ）"""
    global current_index
    return load_image_at_index(current_index + direction)


def jump_to_latest():
    """最新の画像に直接ジャンプする"""
    update_image_list()
    return load_image_at_index(len(image_list) - 1)


def save_feedback(rating):
    """現在の画像を評価して保存し、自動的に次の画像へ進む"""
    global image_list, current_index, evaluated_paths

    update_image_list()

    if (
        current_index < 0
        or current_index >= len(image_list)
        or not image_list
    ):
        return "❌ 評価できる画像がありません。", None

    current_path = image_list[current_index]

    # 💡 同じ画像の二重評価をブロック
    if current_path in evaluated_paths:
        img, _ = load_image_at_index(current_index)
        return (
            f"⚠️ この画像はすでに評価済みです！ (進む/戻るボタンで未評価画像を選んでください)",
            img,
        )

    # 画像保存処理
    img = Image.open(current_path).convert("RGB")
    img_array = np.array(img, dtype=np.float32) / 255.0

    new_x = np.expand_dims(img_array, axis=0)
    new_y = np.array([rating], dtype=np.float32)

    if os.path.exists(DATASET_PATH):
        data = np.load(DATASET_PATH)
        X_data = np.concatenate([data["x"], new_x], axis=0)
        Y_data = np.concatenate([data["y"], new_y], axis=0)
    else:
        X_data = new_x
        Y_data = new_y

    np.savez(DATASET_PATH, x=X_data, y=Y_data)
    count = len(Y_data)

    # 評価済みに登録
    evaluated_paths.add(current_path)

    # 💡 自動で次の画像（index + 1）へ進む
    next_img, _ = load_image_at_index(current_index + 1)
    status_msg = f"✅ 保存完了！ (累計: {count} 件) ➔ 次の画像を表示中"

    return status_msg, next_img


def reset_dataset(confirm):
    """データセットと評価履歴を全リセット"""
    global evaluated_paths, current_index

    if not confirm:
        return (
            "⚠️ リセットするにはチェックを入れてください。",
            None,
        )

    if os.path.exists(DATASET_PATH):
        os.remove(DATASET_PATH)

    evaluated_paths.clear()
    next_img, _ = load_image_at_index(current_index)

    return "🗑️ 評価データをリセットしました！ (累計: 0 件)", next_img


# --- Gradio UI 構築 ---
with gr.Blocks(title="PCGRL - Human Feedback Collector") as demo:
    gr.Markdown("## 🎮 PCGRL リアルタイム人間嗜好データ収集UI")

    with gr.Row():
        with gr.Column():
            map_image = gr.Image(label="Generated Map", type="pil")
            status_text = gr.Textbox(
                label="ステータス", value="準備完了", interactive=False
            )

            # ナビゲーションボタン（戻る・最新・進む）
            with gr.Row():
                btn_prev = gr.Button("◀ 前の画像")
                btn_latest = gr.Button("⚡ 最新画像へジャンプ")
                btn_next = gr.Button("次の画像 ▶")

            # 評価ボタン
            with gr.Row():
                btn_bad = gr.Button("👎 Bad (0)", variant="stop")
                btn_good = gr.Button("👍 Good (1)", variant="primary")

            # リセット機能
            with gr.Accordion("⚙️ データリセット管理", open=False):
                chk_confirm = gr.Checkbox(
                    label="本当にリセットする（npzファイルを削除）"
                )
                btn_reset = gr.Button("🗑️ データをすべてリセット", variant="stop")

    # イベントハンドラ
    demo.load(fn=jump_to_latest, outputs=[map_image, status_text])

    # ナビゲーション操作
    btn_prev.click(
        fn=lambda: navigate(-1), outputs=[map_image, status_text]
    )
    btn_next.click(fn=lambda: navigate(1), outputs=[map_image, status_text])
    btn_latest.click(fn=jump_to_latest, outputs=[map_image, status_text])

    # 評価操作
    btn_good.click(
        fn=lambda: save_feedback(1.0), outputs=[status_text, map_image]
    )
    btn_bad.click(
        fn=lambda: save_feedback(0.0), outputs=[status_text, map_image]
    )

    # リセット操作
    btn_reset.click(
        fn=reset_dataset,
        inputs=[chk_confirm],
        outputs=[status_text, map_image],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)