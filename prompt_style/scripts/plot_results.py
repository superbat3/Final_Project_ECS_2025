import pandas as pd
import matplotlib.pyplot as plt

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..','data','dataset_history')

plt.style.use("seaborn-v0_8-whitegrid")

# Load your evaluated results
csv_path = os.path.join(data_dir, "pilot_results_with_scores.csv")
df = pd.read_csv(csv_path)

# Compute the averages per prompt type
grouped = df.groupby("prompt_style").agg({
    "f1": "mean",
    "bleu": "mean",
    "bert_f1": "mean"
})

print(grouped)

colors = ["#6d9dc5", "#e07a5f", "#81b29a", "#f2cc8f"]
line_color = "#1f2933"

plots = [
    ("f1", "F1 by Prompt Style", "F1 Score", "plot_f1.png"),
    ("bleu", "BLEU by Prompt Style", "BLEU Score", "plot_bleu.png"),
    ("bert_f1", "BERTScore F1 by Prompt Style", "BERTScore F1", "plot_bertscore.png"),
]


def draw_metric(ax, metric, title, ylabel):
    values = grouped[metric]
    x = range(len(values))

    bars = ax.bar(
        x,
        values,
        color=colors[:len(values)],
        width=0.55,
        edgecolor="white",
        linewidth=0.6,
        zorder=2
    )
    ax.plot(
        x,
        values,
        color=line_color,
        marker="o",
        markersize=6,
        linewidth=1.5,
        zorder=3
    )

    ax.set_title(title, pad=14, fontsize=16, weight="bold")
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlabel("Prompt Style", fontsize=13)
    ax.set_xticks(list(x))
    ax.set_xticklabels(grouped.index, rotation=0, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, max(values) * 1.1)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color=line_color
        )


# Individual plots
for metric, title, ylabel, outfile in plots:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    draw_metric(ax, metric, title, ylabel)
    fig.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close(fig)

# Combined overview
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
for ax, (metric, title, ylabel, _) in zip(axes, plots):
    draw_metric(ax, metric, title, ylabel)

fig.tight_layout(w_pad=2.5)
plt.savefig("plot_scores_overview.png", dpi=200)
plt.show()
