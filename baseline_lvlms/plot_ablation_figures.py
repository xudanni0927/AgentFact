"""
Static publication figures for the modality ablation (T/T+E/T+I+E) and the
evidence-quantity ablation (by raw K and by evidence-coverage ratio).

Uses the validated categorical palette (blue/orange/aqua/yellow) and a
single-hue sequential blue ramp for the T->T+E->T+I+E progression, since that
axis is an ordered "more input added" magnitude, not an unordered category.

Usage:
    python plot_ablation_figures.py
Outputs PNGs into baseline_lvlms/figures/.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Validated categorical palette (references/palette.md)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"

# Sequential blue ramp (light -> dark) for the ordered T -> T+E -> T+I+E axis
SEQ_BLUE = ["#aecbe8", "#4a90d9", "#123e73"]

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#888888",
    "axes.grid": True,
    "grid.color": "#e5e5e5",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def fig1_modality_ablation():
    # qwen3-vl and internvl3.5 are fully complete (T/T+E/T+I+E); gemini-3-flash
    # only has T so far — add its TE/TIE bars once that job finishes.
    models = ["qwen3-vl", "internvl3.5"]
    settings = ["T", "T+E", "T+I+E"]
    acc3 = {
        "qwen3-vl":     [0.0750, 0.8128, 0.7743],
        "internvl3.5":  [0.1507, 0.8477, 0.8090],
    }
    acc2 = {
        "qwen3-vl":     [0.7056, 0.8949, 0.8698],
        "internvl3.5":  [0.6969, 0.9187, 0.8777],
    }

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    x = range(len(models))
    width = 0.25

    for ax, data, title in [(axes[0], acc3, "3-class Accuracy"),
                             (axes[1], acc2, "Binary Accuracy (true vs not_true)")]:
        for i, setting in enumerate(settings):
            vals = [data[m][i] for m in models]
            offsets = [xi + (i - 1) * width for xi in x]
            bars = ax.bar(offsets, vals, width=width, color=SEQ_BLUE[i],
                          label=setting, edgecolor="white", linewidth=0.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8, color="#333333")
        ax.set_xticks(list(x))
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Modality ablation: T vs T+E vs T+I+E", y=1.12, fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "modality_ablation.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def fig2_evidence_by_k():
    k_labels = ["1", "2", "3", "5", "7", "9", "All"]
    x = range(len(k_labels))

    overall = {
        "3-class Acc":  [0.5627, 0.6922, 0.7703, 0.8303, 0.8421, 0.8414, 0.8437],
        "Macro-F1":     [0.4983, 0.5781, 0.6314, 0.6698, 0.6772, 0.6648, 0.6700],
        "Weighted-F1":  [0.6224, 0.7349, 0.7958, 0.8397, 0.8498, 0.8477, 0.8500],
    }
    per_class_f1 = {
        "FALSE":     [0.6337, 0.7633, 0.8319, 0.8810, 0.8905, 0.8899, 0.8919],
        "TRUE":      [0.6786, 0.7727, 0.8203, 0.8558, 0.8690, 0.8700, 0.8710],
        "UNPROVEN":  [0.1826, 0.1982, 0.2419, 0.2727, 0.2722, 0.2346, 0.2469],
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)

    for (name, ys), color in zip(overall.items(), [BLUE, ORANGE, AQUA]):
        axes[0].plot(x, ys, marker="o", markersize=5, linewidth=2, color=color, label=name)
    axes[0].set_title("Overall metrics vs. evidence count K")

    for (name, ys), color in zip(per_class_f1.items(), [BLUE, ORANGE, AQUA]):
        axes[1].plot(x, ys, marker="o", markersize=5, linewidth=2, color=color, label=name)
    axes[1].set_title("Per-class F1 vs. evidence count K")

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(k_labels)
        ax.set_xlabel("Evidence snippets shown (K)")
        ax.set_ylim(0, 1.0)
        ax.legend(frameon=False, loc="lower right", fontsize=9)

    axes[0].set_ylabel("Score")
    fig.suptitle("GPT-4o-mini evidence-quantity ablation (T+E), by K", fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "evidence_quantity_by_k.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def fig3_evidence_by_ratio():
    buckets = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%", "100%"]
    x = range(len(buckets))
    n = [494, 1359, 1093, 995, 336, 4550]
    acc3 = [0.5243, 0.6233, 0.7045, 0.8020, 0.7708, 0.8521]
    macro_f1 = [0.4859, 0.5438, 0.5882, 0.6392, 0.6566, 0.6759]
    weighted_f1 = [0.5759, 0.6771, 0.7418, 0.8196, 0.7780, 0.8570]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    ax_top.plot(x, acc3, marker="o", markersize=5, linewidth=2, color=BLUE, label="3-class Acc")
    ax_top.plot(x, macro_f1, marker="o", markersize=5, linewidth=2, color=ORANGE, label="Macro-F1")
    ax_top.plot(x, weighted_f1, marker="o", markersize=5, linewidth=2, color=AQUA, label="Weighted-F1")
    ax_top.set_ylim(0, 1.0)
    ax_top.set_ylabel("Score")
    ax_top.legend(frameon=False, loc="lower right", fontsize=9)
    ax_top.set_title("GPT-4o-mini pooled by evidence-coverage ratio (shown / total)")

    ax_bot.bar(x, n, color="#c7c7c7", width=0.6)
    for xi, ni in zip(x, n):
        ax_bot.text(xi, ni + 60, str(ni), ha="center", va="bottom", fontsize=8, color="#555555")
    ax_bot.set_ylabel("n", fontsize=9)
    ax_bot.set_xticks(list(x))
    ax_bot.set_xticklabels(buckets)
    ax_bot.set_xlabel("Evidence-coverage ratio bucket")
    ax_bot.grid(False)

    out = os.path.join(OUT_DIR, "evidence_quantity_by_ratio.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def fig4_accuracy_by_claim_total_evidence():
    # Full-evidence (TE, 100% coverage) tier only, grouped by each claim's OWN
    # total evidence count — isolates claim-intrinsic difficulty from how much
    # evidence the model was actually shown.
    bins = ["0-2", "3-4", "5-6", "7-8", "9+"]
    x = range(len(bins))
    n = [20, 371, 642, 176, 59]
    acc3 = [0.6500, 0.8841, 0.8427, 0.7784, 0.8644]
    macro_f1 = [0.4988, 0.6352, 0.6818, 0.6364, 0.7730]
    weighted_f1 = [0.7399, 0.8914, 0.8448, 0.7865, 0.8768]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    ax_top.plot(x, acc3, marker="o", markersize=5, linewidth=2, color=BLUE, label="3-class Acc")
    ax_top.plot(x, macro_f1, marker="o", markersize=5, linewidth=2, color=ORANGE, label="Macro-F1")
    ax_top.plot(x, weighted_f1, marker="o", markersize=5, linewidth=2, color=AQUA, label="Weighted-F1")
    ax_top.set_ylim(0, 1.0)
    ax_top.set_ylabel("Score")
    ax_top.legend(frameon=False, loc="lower right", fontsize=9)
    ax_top.set_title("GPT-4o-mini, full evidence (100% coverage):\naccuracy by claim's total evidence count", fontsize=11)

    ax_bot.bar(x, n, color="#c7c7c7", width=0.6)
    for xi, ni in zip(x, n):
        ax_bot.text(xi, ni + 15, str(ni), ha="center", va="bottom", fontsize=8, color="#555555")
    ax_bot.set_ylabel("n", fontsize=9)
    ax_bot.set_xticks(list(x))
    ax_bot.set_xticklabels(bins)
    ax_bot.set_xlabel("Claim's total evidence count (ground-truth)")
    ax_bot.grid(False)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "accuracy_by_claim_total_evidence.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def fig5_k_curve_by_claim_difficulty():
    # The joint analysis: for each claim-difficulty stratum (by total evidence
    # count), plot its own K dose-response curve. Tests whether the K effect
    # (main effect, seen in fig2) is consistent in direction across difficulty
    # levels, and whether harder claims saturate slower / at a lower ceiling.
    # Note: the "0-2" stratum here is really "1-2" (claims with total_count=0
    # have no evidence to vary K over, so they're excluded — unlike fig4's
    # "0-2" bin, which does include the 7 zero-evidence claims).
    k_labels = ["1", "2", "3", "5", "7", "9", "All"]
    x = range(len(k_labels))

    bins = ["1-2", "3-4", "5-6", "7-8", "9+"]
    n_by_bin = {"1-2": 13, "3-4": 371, "5-6": 642, "7-8": 176, "9+": 59}
    acc_by_bin = {
        "1-2": [0.6923, 0.8462, 0.8462, 0.8462, 0.8462, 0.8462, 0.8462],
        "3-4": [0.6388, 0.7601, 0.8518, 0.8787, 0.8760, 0.8814, 0.8841],
        "5-6": [0.5312, 0.6558, 0.7430, 0.8302, 0.8474, 0.8427, 0.8427],
        "7-8": [0.5000, 0.6648, 0.7045, 0.7557, 0.7670, 0.7727, 0.7784],
        "9+":  [0.6102, 0.7458, 0.7797, 0.8136, 0.8644, 0.8475, 0.8644],
    }
    wf1_by_bin = {
        "1-2": [0.7183, 0.8519, 0.8519, 0.8519, 0.8519, 0.8519, 0.8519],
        "3-4": [0.6993, 0.7958, 0.8649, 0.8860, 0.8866, 0.8909, 0.8914],
        "5-6": [0.5929, 0.7006, 0.7670, 0.8345, 0.8505, 0.8443, 0.8448],
        "7-8": [0.5359, 0.7072, 0.7430, 0.7739, 0.7756, 0.7794, 0.7865],
        "9+":  [0.6871, 0.7963, 0.8232, 0.8431, 0.8817, 0.8592, 0.8768],
    }

    # Sequential single-hue ramp (light -> dark blue): total-evidence-count is
    # ordinal (least to most), unlike the unordered per-class / per-metric
    # comparisons in fig2/fig3, so a magnitude ramp reads more naturally than
    # unrelated categorical hues here.
    cmap = plt.get_cmap("Blues")
    ramp = [cmap(v) for v in [0.35, 0.5, 0.65, 0.8, 0.95]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for b, color in zip(bins, ramp):
        label = f"{b} (n={n_by_bin[b]})"
        axes[0].plot(x, acc_by_bin[b], marker="o", markersize=5, linewidth=2, color=color, label=label)
        axes[1].plot(x, wf1_by_bin[b], marker="o", markersize=5, linewidth=2, color=color, label=label)

    axes[0].set_title("3-class Accuracy vs K, by claim's total evidence count")
    axes[1].set_title("Weighted-F1 vs K, by claim's total evidence count")
    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(k_labels)
        ax.set_xlabel("Evidence snippets shown (K)")
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Score")
    axes[1].legend(frameon=False, loc="lower right", fontsize=8, title="Total evidence")

    fig.suptitle("GPT-4o-mini: does the evidence-quantity effect hold within each claim-difficulty stratum?", fontsize=11, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "k_curve_by_claim_difficulty.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    fig1_modality_ablation()
    fig2_evidence_by_k()
    fig3_evidence_by_ratio()
    fig4_accuracy_by_claim_total_evidence()
    fig5_k_curve_by_claim_difficulty()
