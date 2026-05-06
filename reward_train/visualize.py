import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from sklearn.manifold import TSNE


def create_scatter_plot(df, epoch, config, min_val=0, max_val=1,
                        xlim=(-10, 10), ylim=(-10, 10), postfix=""):


    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    sm = cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    _ = sns.scatterplot(
        data=df, x="ground_truth", y="prediction",
        hue="reward_id", palette="bright", alpha=0.5, ax=ax
    )

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Epoch")

    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Prediction")

    ax.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.join(config.exp_dir, config.figure_dir), exist_ok=True)
    fig_path = os.path.join(config.exp_dir, config.figure_dir, f"scatter_epoch_{epoch}{postfix}.png")
    plt.savefig(fig_path)
    plt.close(fig)

    return fig_path


def split_digits(ids):
    return [[int(digit) for digit in str(num)] for num in ids]


def create_embedding_figure(embed_queue, reward_df: pd.DataFrame, epoch, config, postfix="") -> str:


    reward_ids = [e.reward_id for e in embed_queue]
    inst_cols = reward_df.iloc[reward_ids][['reward_enum']].reset_index()
    split_reward_ids = split_digits(inst_cols['reward_enum'])

    if config.encoder.separated_reward:
        embeds = np.array([e.embedding[idx - 1] for e, rid in zip(embed_queue, split_reward_ids) for idx in rid])
    else:
        embeds = np.array([e.embedding for e in embed_queue])

    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeds = tsne.fit_transform(embeds)

    flat_reward_ids = [idx for sublist in split_reward_ids for idx in sublist]

    raw_reward_enums = inst_cols['reward_enum'].tolist()
    flat_edge_ids = [reward_id for reward_id, split in zip(raw_reward_enums, split_reward_ids) for _ in split]

    df = pd.DataFrame(tsne_embeds, columns=['tsne_x', 'tsne_y']).reset_index(drop=True)
    df['reward_enum'] = flat_reward_ids
    df['edge_enum'] = flat_edge_ids

    df['edge_tens'] = df['edge_enum'] // 10
    df_two_digits = df[df['edge_enum'] >= 10].copy()
    df_one_digit = df[df['edge_enum'] < 10].copy()

    df_two_digits_sampled = (
        df_two_digits.groupby('edge_tens', group_keys=False)
        .apply(lambda x: x.sample(frac=0.4, random_state=42))
        .reset_index(drop=True)
    )

    filtered_df = pd.concat([df_one_digit, df_two_digits_sampled], ignore_index=True)

    inner_palette = sns.color_palette("bright", len(set(filtered_df['reward_enum'])))
    edge_palette = sns.color_palette("dark", len(set(filtered_df['edge_enum'])))
    inner_color_map = dict(zip(sorted(set(filtered_df['reward_enum'])), inner_palette))
    edge_color_map = dict(zip(sorted(set(filtered_df['edge_enum'])), edge_palette))

    filtered_df['facecolor'] = filtered_df['reward_enum'].map(inner_color_map)
    filtered_df['edgecolor'] = filtered_df['edge_enum'].map(edge_color_map)

    marker_styles = ['o', 'o', 'o', 'o', 'o', 'P', '^', 's', 'X', 'p']
    unique_enums = sorted(filtered_df['edge_enum'].unique())
    marker_map = {eid: marker_styles[i % len(marker_styles)] for i, eid in enumerate(unique_enums)}
    filtered_df['marker'] = filtered_df['edge_enum'].map(marker_map)

    # draw scatter plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))
    for edge_enum in unique_enums:
        sub_df = filtered_df[filtered_df['edge_enum'] == edge_enum]
        sns.scatterplot(
            data=sub_df,
            x='tsne_x',
            y='tsne_y',
            hue='reward_enum',
            palette=inner_color_map,
            marker=marker_map[edge_enum],
            s=80,
            alpha=0.9,
            ax=ax,
            legend=False
        )

    ax.set_xlabel("Projection X")
    ax.set_ylabel("Projection Y")
    ax.grid(True)
    plt.tight_layout()

    save_path = os.path.join(config.exp_dir, config.figure_dir, f"embed_epoch_{epoch}{postfix}.csv")
    df.to_csv(save_path, index=False)

    os.makedirs(os.path.join(config.exp_dir, config.figure_dir), exist_ok=True)
    fig_path = os.path.join(config.exp_dir, config.figure_dir, f"embed_epoch_{epoch}{postfix}.png")
    plt.savefig(fig_path)
    plt.close(fig)

    return fig_path