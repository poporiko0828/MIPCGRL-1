import numpy as np
from collections import Counter
import os

# =============================================================
# 指標計算モジュール
# =============================================================

def get_tile_distribution(maps, num_tile_types=8):
    """【指標①-1】 タイル出現比率の確率分布を計算"""
    flat = maps.flatten()
    counts = Counter(flat)
    total = len(flat)
    dist = np.array([counts.get(t, 0) / total for t in range(num_tile_types)])
    return dist

def get_patch_distribution(maps, patch_size=3):
    """【指標①-2】 3x3 局所パッチのパターン出現頻度を抽出"""
    N, H, W = maps.shape
    patch_counts = Counter()
    
    for m in range(N):
        for r in range(H - patch_size + 1):
            for c in range(W - patch_size + 1):
                patch = tuple(maps[m, r:r+patch_size, c:c+patch_size].flatten())
                patch_counts[patch] += 1
                
    total = sum(patch_counts.values())
    return {k: v / total for k, v in patch_counts.items()}

def calculate_kl_divergence(p, q, epsilon=1e-10):
    """KLダイバージェンスの計算 (0に近いほど類似度が高い)"""
    p = np.asarray(p, dtype=np.float64) + epsilon
    q = np.asarray(q, dtype=np.float64) + epsilon
    p /= p.sum()
    q /= q.sum()
    return np.sum(p * np.log(p / q))

def calculate_patch_kl(dict_p, dict_q, epsilon=1e-10):
    """辞書形式の局所パッチ分布間の KL ダイバージェンス"""
    all_keys = set(dict_p.keys()) | set(dict_q.keys())
    vec_p = np.array([dict_p.get(k, 0.0) for k in all_keys])
    vec_q = np.array([dict_q.get(k, 0.0) for k in all_keys])
    return calculate_kl_divergence(vec_p, vec_q, epsilon)

def analyze_unconditioned_attributes(maps, wall_tile_id=1):
    """【指標③】 非指示要素 (例: 壁タイルの総量・密度) の平均と標準偏差"""
    N, H, W = maps.shape
    total_cells = H * W
    wall_ratios = []
    
    for m in range(N):
        wall_count = np.sum(maps[m] == wall_tile_id)
        wall_ratios.append(wall_count / total_cells)
        
    mean_ratio = np.mean(wall_ratios)
    std_ratio = np.std(wall_ratios)
    return mean_ratio, std_ratio

# =============================================================
# メイン評価実行スクリプト
# =============================================================

def main():
    data_dir = "eval_data"
    
    # 💡 指定された評価用データの読み込み
    path_A1 = os.path.join(data_dir, "maps_step_00122880_manybats.npy")
    path_A3 = os.path.join(data_dir, "maps_step_00248320_manybats.npy")
    
    if not (os.path.exists(path_A1) and os.path.exists(path_A3)):
        print("エラー: 評価用データ (.npy) が見つかりません。")
        print(f"  - 検索パス1: {path_A1}")
        print(f"  - 検索パス2: {path_A3}")
        print("`eval_data` ディレクトリ内のファイル名を確認してください。")
        return

    maps_A1 = np.load(path_A1)
    maps_A3 = np.load(path_A3)
    
    print("=== 破滅的忘却・表現保持の定量評価結果 ===")
    print(f"比較元 (Phase 1): {os.path.basename(path_A1)} | Shape: {maps_A1.shape}")
    print(f"比較先 (Phase 3): {os.path.basename(path_A3)} | Shape: {maps_A3.shape}\n")
    
    # 1. 全体タイル分布の KL ダイバージェンス
    tile_dist_A1 = get_tile_distribution(maps_A1)
    tile_dist_A3 = get_tile_distribution(maps_A3)
    tile_kl = calculate_kl_divergence(tile_dist_A1, tile_dist_A3)
    print(f"【指標①-1】 タイル分布の KL ダイバージェンス: {tile_kl:.5f}")
    print("  (0 に近いほど全体的なタイル構成比率が忘却されず保持されている)")
    
    # 2. 3x3 局所パッチの KL ダイバージェンス
    patch_dist_A1 = get_patch_distribution(maps_A1, patch_size=3)
    patch_dist_A3 = get_patch_distribution(maps_A3, patch_size=3)
    patch_kl = calculate_patch_kl(patch_dist_A1, patch_dist_A3)
    print(f"【指標①-2】 3x3 局所パッチの KL ダイバージェンス: {patch_kl:.5f}")
    print("  (0 に近いほどマップの局所構造や細部のパターンを描く能力が維持されている)")
    
    # 3. 非指示要素 (壁タイル密度) の分析
    mean1, std1 = analyze_unconditioned_attributes(maps_A1, wall_tile_id=1) # 1を壁IDと仮定
    mean3, std3 = analyze_unconditioned_attributes(maps_A3, wall_tile_id=1)
    print(f"\n【指標③】 非指示要素 (壁タイル密度) の比較:")
    print(f"  Phase 1 (初回学習完了時): 平均 = {mean1:.4f}, 標準偏差 = {std1:.4f}")
    print(f"  Phase 3 (別指示学習後):   平均 = {mean3:.4f}, 標準偏差 = {std3:.4f}")
    
    inv_ratio = mean3 / (mean1 + 1e-10)
    print(f"  不変比率 (Invariance Ratio): {inv_ratio:.4f}")
    print("  (1.0 に近いほど指示と無関係なマップ構造が崩れていない)")

if __name__ == "__main__":
    main()