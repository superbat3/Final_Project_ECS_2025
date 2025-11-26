import matplotlib.pyplot as plt
import pandas as pd


plt.style.use("seaborn-v0_8-whitegrid")
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..','data','dataset_history')

plt.style.use("seaborn-v0_8-whitegrid")

# Load your evaluated results

# Load data
df = pd.read_csv(os.path.join(data_dir, "pilot_results_with_hallu.csv"))
df["prompt_style"] = pd.Categorical(df["prompt_style"],
                                    ["direct", "explain", "fact"],
                                    ordered=True)

# Style settings
colors = ["#6d9dc5", "#e07a5f", "#81b29a", "#f2cc8f"]
line_color = "#1f2933"


def draw_hallucination_rate(ax):
    """Draw hallucination rate bar chart with line overlay."""
    means = df.groupby("prompt_style", observed=True)["hallu_label"].mean()
    x = range(len(means))
    values = means.values

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

    ax.set_title("Hallucination Rate by Prompt Style", pad=14, fontsize=16, weight="bold")
    ax.set_ylabel("Hallucination Rate", fontsize=13)
    ax.set_xlabel("Prompt Style", fontsize=13)
    ax.set_xticks(list(x))
    ax.set_xticklabels(means.index, rotation=0, fontsize=12)
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


def draw_severity_distribution(ax):
    """Draw stacked bar chart for hallucination severity distribution."""
    dist = df.groupby(["prompt_style", "hallu_label"], observed=True).size().unstack(fill_value=0)
    dist_norm = dist.div(dist.sum(axis=1), axis=0)

    x = range(len(dist_norm.index))
    width = 0.55
    bottom = [0] * len(dist_norm.index)

    for i, label in enumerate(dist_norm.columns):
        ax.bar(
            x,
            dist_norm[label],
            width=width,
            bottom=bottom,
            color=colors[i],
            edgecolor="white",
            linewidth=0.6,
            label=f"Label {label}",
            zorder=2
        )
        bottom = bottom + dist_norm[label]

    ax.set_title("Hallucination Severity Distribution", pad=14, fontsize=16, weight="bold")
    ax.set_ylabel("Proportion", fontsize=13)
    ax.set_xlabel("Prompt Style", fontsize=13)
    ax.set_xticks(list(x))
    ax.set_xticklabels(dist_norm.index, rotation=0, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 1)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(
        title="Severity",
        fontsize=11,
        title_fontsize=12,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        fancybox=False,
        shadow=False,
        framealpha=1.0,
        edgecolor="#e0e0e0",
        facecolor="white",
        borderpad=0.6,
        columnspacing=0.8,
        handlelength=1.2,
        handletextpad=0.5,
        borderaxespad=0.5
    )


# Individual plots
fig, ax = plt.subplots(figsize=(6.8, 4.4))
draw_hallucination_rate(ax)
fig.tight_layout()
plt.savefig("hallucination_bar.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.8, 4.4))
draw_severity_distribution(ax)
fig.tight_layout()
plt.savefig("hallucination_stacked.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Combined overview
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=False)
draw_hallucination_rate(axes[0])
draw_severity_distribution(axes[1])

fig.tight_layout(w_pad=2.5)
plt.savefig("hallucination_overview.png", dpi=200, bbox_inches="tight")
plt.show()

print("Saved hallucination_bar.png, hallucination_stacked.png, and hallucination_overview.png")