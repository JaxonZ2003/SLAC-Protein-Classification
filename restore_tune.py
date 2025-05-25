# import os
# import getpass
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

from ray import tune
from ray.tune.execution.placement_groups import PlacementGroupFactory
# from matplotlib.patches import Rectangle, FancyArrowPatch
# from matplotlib import cm

from SLAC25.tune import Trainable
# from search_space import Experiments

trainable_with_resources = tune.with_resources(
    Trainable,
    resources=PlacementGroupFactory([{"CPU": 2, "GPU": 1}])
)

exp_path = "/home/jaxonz/slac_experiments/BestSoFar"
exp_name = "BestSoFar"

print("Restore training from:")
print(f"• Path        : {exp_path}")
print(f"• Experiment  : {exp_name}")

Tunner = tune.Tuner.restore(path=exp_path, 
                      trainable=trainable_with_resources,
                      resume_unfinished=True,
                      resume_errored=True)

print(f"Training start...")
results= Tunner.fit()
print(f"Training Complete.")

best_result = results.get_best_result("val_accuracy", mode="max")
save_dir = f"~/best_model_checkpoint/{exp_name}"

if best_result:
    print("Files are saved at:", best_result.checkpoint.to_directory(save_dir))
else:
    print("No successful trials - see logs in", results.logdir)

best_config = best_result.config
print("Best config:", best_config)

# path = {}
# result_dfs = {}
# rows = []
# param_map = {
#     "1": "lr",
#     "2": "keep_prob",
#     "3": "lr_scheduler",
#     "4": "lr"
# }

# param_map_full = {
#     "1": "Learning Rate",
#     "2": "Drop %",
#     "3": "Learning Rate Scheduler",
#     "4": "Learning Rate (Dimension 256)"
# }

# for exp_id in list(Experiments.keys())[:4]:
#     path[exp_id] = f"/home/jaxonz/slac_experiments/{Experiments[exp_id].get("name", "Unnamed")}"
#     restored_tuner = tune.Tuner.restore(path[exp_id], trainable=trainable_with_resources)
#     result_grid = restored_tuner.get_results()

#     if result_grid.errors:
#         print(f"One of the trials failed for Experiment: {exp_id}")
#     else:
#         print("No errors!")

#     result_df = result_grid.get_dataframe()
#     result_dfs[exp_id] = result_df


# for key, param in param_map.items():
#     x_labels = result_dfs[key][f"config/{param}"].tolist()
#     values = result_dfs[key]["val_accuracy"].tolist()

#     sorted_indices = sorted(range(len(x_labels)), key=lambda i: x_labels[i])

#     # Permute both x_labels and values based on sorted indices
#     sorted_x_labels = [x_labels[i] for i in sorted_indices]
#     sorted_values = [values[i] for i in sorted_indices]

#     row = {
#         "y_label": param,
#         "x_labels": sorted_x_labels,
#         "values": sorted_values
#     }

#     # row = {
#     #     "y_label": param,
#     #     "x_labels": result_dfs[key][f"config/{param}"].tolist(),
#     #     "values": result_dfs[key]["val_accuracy"].tolist()
#     # }
#     rows.append(row)

# rows[1]["x_labels"] = [round(1 - keep_prob, 2) for keep_prob in rows[1]["x_labels"]]

# for i in range(len(param_map)):
#     rows[i]["y_label"] = param_map_full[str(i + 1)]

# print(path)
# print(rows)


# df = pd.read_csv(path)


# # exp_4_lr4 = result_dfs["4"].loc[result_dfs["4"]["config/lr"] == 0.0001]
# x = df["training_iteration"]
# y1 = df["train_accuracy"]
# y2 = df["val_accuracy"]

# plt.figure()
# plt.plot(x, y1, label="Training")
# plt.plot(x, y2, label="Validation")
# # plt.axhline(0.25, color="gray", linestyle=":", linewidth=1.2, label="Baseline 0.25")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.title("Training vs Accuracy over 20 Epochs")
# plt.xticks(range(1, 21))
# plt.legend()
# plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
# plt.tight_layout()
# plt.savefig('/home/jaxonz/SLAC/capstone-SLAC/img/test_img.png', bbox_inches="tight", pad_inches=0.1)


# ##########

# import matplotlib.pyplot as plt
# import numpy as np

# # --------------------------
# # Data preparation
# # --------------------------
# # Counts for each label in each dataset
# counts_train = [139865, 53044, 23299, 199567]
# counts_val   = [15837,  5936,  2573,   22683]
# counts_test  = [15659,  6200,  2810,   22360]

# datasets = ["Train", "Validation", "Test"]
# label_names = ["Label 0", "Label 1", "Label 2", "Label 3"]

# # Percentages (identical across splits, use train as reference)
# perc_labels = [33.64, 12.76, 5.60, 48.00]
# colors = ["#a6cee3", "#1f78b4", "#045a8d", "#08306b"] 

# # Dataset‑level sizes and their share in the total
# size_train = sum(counts_train)
# size_val   = sum(counts_val)
# size_test  = sum(counts_test)

# sizes = [size_train, size_val, size_test]
# total_size = sum(sizes)
# share = [round(s / total_size * 100, 2) for s in sizes]  # percentage of total

# # --------------------------
# # Figure 1 : One bar ‑ label proportions
# # --------------------------
# fig1, ax1 = plt.subplots(figsize=(4, 6))

# bottom = 0
# for i, pct in enumerate(perc_labels):
#     ax1.bar(["All Splits"], [pct], bottom=bottom, label=label_names[i], color=colors[i])
#     # annotate the percentage on each segment
#     ax1.text(0, bottom + pct / 2, f"{pct:.2f}%", ha='center', va='center', fontsize=9, color="white")
#     bottom += pct

# ax1.set_ylabel("Percentage of Samples (%)")
# ax1.set_ylim(0, 100)
# ax1.set_title("Class Distribution")
# # ax1.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
# ax1.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.tight_layout()
# plt.savefig('/home/jaxonz/SLAC/capstone-SLAC/img/test_img3.png', bbox_inches="tight", pad_inches=0.1)

# # --------------------------
# # Figure 2 : Stacked bars ‑ counts + dataset share
# # --------------------------
# fig2, ax2 = plt.subplots(figsize=(6, 6))

# bottom = np.zeros(len(datasets))
# for i, lbl in enumerate(label_names):
#     vals = [counts_train[i], counts_val[i], counts_test[i]]
#     ax2.bar(datasets, vals, bottom=bottom, label=lbl, color=colors[i])
#     # annotate segment counts if space allows
#     for j, v in enumerate(vals):
#         if v > 5000:  # only annotate bigger blocks for clarity
#             ax2.text(j, bottom[j] + v / 2, f"{v:,}", ha='center', va='center', fontsize=8, color="white")
#     bottom += vals

# # Annotate total counts & share above each bar
# for idx, (size, pct) in enumerate(zip(sizes, share)):
#     ax2.text(idx, size + total_size * 0.02, f"n={size:,}\n{pct:.1f}%", ha='center', va='bottom', fontsize=9)

# ax2.set_ylabel("Number of Samples")
# ax2.set_title("Dataset Sizes & Class Breakdown")
# # ax2.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
# ax2.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.tight_layout()


# plt.savefig('/home/jaxonz/SLAC/capstone-SLAC/img/test_img2.png', bbox_inches="tight", pad_inches=0.1)





# def draw_plot():
#     # all_vals = np.concatenate([r["values"] for r in rows])
#     norm = plt.Normalize(0, 1)
#     cmap = plt.get_cmap("Blues")

#     fig, ax = plt.subplots(figsize=(16,8))
#     ax.set_axis_off()
#     ax.set_aspect("equal")

#     # parameters for drawing
#     row_height = 1.0
#     rect_size = 0.8
#     gap = 0.1  # gap between rectangles

#     y_axis_x = 0  # x position of y‑axis line

#     current_y = 0  # bottom of first row

#     for r in rows:
#         ax.text(
#             2.5 * (rect_size + gap),
#             current_y - rect_size/2,
#             r["y_label"],
#             ha="center",
#             va="center",
#             fontsize=10,
#         )

#         # Draw x‑axis line with arrow for this row
#         x_start = y_axis_x
#         # width depends on number of squares
#         # num_sq = len(r["values"])
#         x_end = x_start + 5 * (rect_size + gap) - gap
#         arrow = FancyArrowPatch(
#             (x_start, current_y),
#             (x_end + 0.3, current_y),
#             arrowstyle="-|>",
#             linewidth=1.2,
#             mutation_scale=8,
#             color="black",
#         )
#         ax.add_patch(arrow)

#         # rectangles
#         for i, (val, label) in enumerate(zip(r["values"], r["x_labels"])):
#             x_rect = 0.2 + i * (rect_size + gap)
#             rect = Rectangle(
#                 (x_rect, current_y + (row_height - rect_size) / 2),
#                 width=rect_size,
#                 height=rect_size,
#                 facecolor=cmap(norm(val)),
#                 edgecolor="white",
#                 linewidth=1.5,
#             )
#             ax.add_patch(rect)
#             # annotate value
#             ax.text(
#                 x_rect + rect_size / 2,
#                 current_y + row_height / 2,
#                 f"{val:.2f}",
#                 ha="center",
#                 va="center",
#                 color="black",
#                 fontsize=8,
#                 fontweight="bold",
#             )
#             # xtick (label below axis)
#             ax.text(
#                 x_rect + rect_size / 2,
#                 current_y - 0.1,
#                 label,
#                 ha="center",
#                 va="top",
#                 fontsize=8,
#                 rotation=45,
#             )

#         current_y += row_height + 0.6  # vertical space to next row

#     # colorbar (validation accuracy)
#     sm = cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
#     cbar.set_label("Validation Accuracy")

#     plt.xlim(-0.5, x_end + 1)
#     plt.ylim(-0.5, current_y)
#     plt.tight_layout()
#     plt.savefig('/home/jaxonz/SLAC/capstone-SLAC/img/test_img.png', bbox_inches="tight", pad_inches=0.1)

# draw_plot()