# human_feedback_ui.py
import gradio as gr
import os
import glob
import numpy as np

# 💡 プロジェクトの絶対パスを取得し、画像保存フォルダを正確に指定
# human_feedback_ui.py の上部

# 💡 画像が保存されている正確なパスを指定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(
    BASE_DIR, 
    "saves", 
    "embed-bert_inst-None_model-nlpconv_exp-def_cls_st_mk_s-0", 
    "images"
)

WEIGHT_FILE = os.path.join(BASE_DIR, "rm_weights.npy")

def get_latest_image():
    """フォルダ内の最新の png / jpg 画像を取得する"""
    if not os.path.exists(IMG_DIR):
        print(f"⚠️ フォルダが存在しません: {IMG_DIR}")
        return None
    
    # png と jpg の両方を検索
    images = glob.glob(os.path.join(IMG_DIR, "*.png")) + glob.glob(os.path.join(IMG_DIR, "*.jpg"))
    if not images:
        print(f"⚠️ 画像ファイルが見つかりません: {IMG_DIR}")
        return None
        
    # 最も更新日時が新しい画像を取得
    latest_image = max(images, key=os.path.getctime)
    return latest_image

def submit_feedback(score):
    """ボタンが押されたら一時的なスコアパルス（刺激）を与える"""
    # スコアパルスを与える（例: 👍なら +5.0、👎なら -5.0 をセットして一気に減衰させる）
    pulse_score = score * 5.0 
    np.save(WEIGHT_FILE, float(pulse_score))
    
    latest_img = get_latest_image()
    status = f"✅ フィードバック送信完了! 一時スコア: {pulse_score}"
    return latest_img, status

# 🎨 Gradio UIの構築
with gr.Blocks(title="MIPCGRL - Human Feedback") as demo:
    gr.Markdown("## 🗺️ マップ生成 リアルタイム評価 UI")
    
    with gr.Row():
        img_output = gr.Image(label="最新の生成マップ", type="filepath", interactive=False)
    
    with gr.Row():
        btn_good = gr.Button("👍 いいね (スコアUP)", variant="primary")
        btn_bad = gr.Button("👎 イマイチ (スコアDOWN)", variant="stop")
        btn_refresh = gr.Button("🔄 最新画像をリロード")
        
    status_text = gr.Textbox(label="ステータス", interactive=False)
    
    # イベント紐付け
    btn_good.click(fn=lambda: submit_feedback(1.0), outputs=[img_output, status_text])
    btn_bad.click(fn=lambda: submit_feedback(-1.0), outputs=[img_output, status_text])
    btn_refresh.click(fn=lambda: (get_latest_image(), "画像を更新しました"), outputs=[img_output, status_text])
    
    # 起動時に最初の画像をロード
    demo.load(fn=lambda: (get_latest_image(), "待機中..."), outputs=[img_output, status_text])

if __name__ == "__main__":
    # 💡 allowed_paths に IMG_DIR を渡すことで Gradio のファイル参照エラーを防止
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=True,
        allowed_paths=[IMG_DIR]
    )