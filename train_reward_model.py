import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DATASET_PATH = "human_dataset.npz"
OUTPUT_WEIGHTS_PATH = "rm_weights.npy"


# --- 1. 軽量CNN（報酬モデル）の定義 ---
class RewardModel(nn.Module):

    def __init__(self):
        super(RewardModel, self).__init__()
        # 画像入力: (B, 3, H, W)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 半分にダウンサイズ
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # サイズ調整
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # 出力を 0.0 ~ 1.0 の確率（人間好み度）にする
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def train_rm():
    if not os.path.exists(DATASET_PATH):
        print(
            f"データセット ({DATASET_PATH}) が見つかりません。Gradio UIで評価を溜めてください。"
        )
        return

    # --- 2. データの読み込みと前処理 ---
    data = np.load(DATASET_PATH)
    X_data = data["x"]  # (N, H, W, 3)
    Y_data = data["y"]  # (N,)

    if len(Y_data) < 5:
        print(
            f"データ数が少なすぎます (現在 {len(Y_data)} 件)。最低5件以上のデータを用意してください。"
        )
        return

    # PyTorchの入力形式 (N, C, H, W) に転置
    X_tensor = torch.tensor(X_data, dtype=torch.float32).permute(0, 3, 1, 2)
    Y_tensor = torch.tensor(Y_data, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # --- 3. モデル・損失関数・最適化関数の初期化 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RewardModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- 4. 学習ループ ---
    model.train()
    epochs = 20
    print(
        f"--- 報酬モデルの学習開始 (データ数: {len(Y_data)}, Device: {device}) ---"
    )

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    print(
        f"学習完了 (Epochs: {epochs}, Final Loss: {total_loss / len(dataloader):.4f})"
    )

    # --- 5. 重みを NumPy 配列として保存 (train.py が読み込める形式) ---
    weights_dict = {}
    for name, param in model.state_dict().items():
        weights_dict[name] = param.cpu().numpy()

    np.save(OUTPUT_WEIGHTS_PATH, weights_dict)
    print(f"最新の重みを保存しました ➔ {OUTPUT_WEIGHTS_PATH}")


if __name__ == "__main__":
    train_rm()