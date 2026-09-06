# import matplotlib.pyplot as plt
# import numpy as np

# def plot_swim_vs_random_boxplots(df_swim, df_random, title_prefix="TF1", save=True, save_dir="static", save_path=None):
#     """Side-by-side boxplots comparing SG-KAN(SWIM) vs SG-KAN(random) RMSE
#     distributions across 50 seeds, for train and test splits."""
#     fig, axes = plt.subplots(1, 2, figsize=(11, 6))

#     for ax, split in zip(axes, ["train", "test"]):
#         swim_vals = df_swim[f"{split}_rmse"].values
#         random_vals = df_random[f"{split}_rmse"].values

#         bp = ax.boxplot(
#             [swim_vals, random_vals],
#             tick_labels=["SG-KAN\n(SWIM)", "SG-KAN\n(Random)"],
#             patch_artist=True,
#             showmeans=True,
#             meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=6),
#         )

#         colors = ['#1f77b4', '#ff7f0e']
#         for patch, color in zip(bp['boxes'], colors):
#             patch.set_facecolor(color)
#             patch.set_alpha(0.5)

#         ax.set_yscale('log')
#         ax.set_ylabel('RMSE (log scale)', fontsize=12)
#         ax.set_title(f'{split.capitalize()} RMSE', fontsize=13)
#         ax.grid(True, alpha=0.3, axis='y')
#         ax.tick_params(axis='both', labelsize=11)

#     fig.suptitle(f'{title_prefix} — SG-KAN(SWIM) vs SG-KAN(Random), 50 runs', fontsize=15)
#     plt.tight_layout()

#     if save:
#         import os
#         os.makedirs(save_dir, exist_ok=True)
#         out_path = save_path or os.path.join(save_dir, f"{title_prefix}_swim_vs_random_boxplot.png")
#         plt.savefig(out_path, dpi=300, bbox_inches='tight')
#         print(f"saved to {out_path}")

#     plt.show()


# plot_swim_vs_random_boxplots(tf1_sgkan_50_runs_df, tf1_sgkan_random_50_runs_df, title_prefix="TF1")


# import numpy as np
# from scipy.stats import skew

# def compare_distributions(df_swim, df_random, split="test", metric="rmse"):
#     col = f"{split}_{metric}"
#     swim_vals = df_swim[col].values
#     random_vals = df_random[col].values

#     print(f"=== {split.upper()} {metric.upper()} — 50-run distribution comparison ===\n")

#     for name, vals in [("SWIM", swim_vals), ("Random", random_vals)]:
#         print(f"{name}:")
#         print(f"  min    = {vals.min():.3e}")
#         print(f"  median = {np.median(vals):.3e}")
#         print(f"  mean   = {vals.mean():.3e}")
#         print(f"  max    = {vals.max():.3e}")
#         print(f"  std    = {vals.std():.3e}")
#         print(f"  skew   = {skew(vals):.3f}")
#         # ratio of mean to median: >>1 means a few large outliers pulling the mean up
#         print(f"  mean/median ratio = {vals.mean() / np.median(vals):.2f}")
#         print()

#     return swim_vals, random_vals


# # Run for TF1 train and test
# swim_train, random_train = compare_distributions(tf1_sgkan_50_runs_df, tf1_sgkan_random_50_runs_df, split="train")
# swim_test, random_test = compare_distributions(tf1_sgkan_50_runs_df, tf1_sgkan_random_50_runs_df, split="test")