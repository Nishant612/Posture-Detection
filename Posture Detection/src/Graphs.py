import os
import sys
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, OUTPUT_DIR

try:
    import matplotlib
    matplotlib.use("Agg")   
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    print("\n[ERROR] matplotlib not installed.")
    print("        Run:  pip install matplotlib seaborn\n")
    sys.exit(1)


HISTORY_FILE = os.path.join(MODELS_DIR, "training_history.json")
GRAPHS_DIR   = os.path.join(OUTPUT_DIR, "graphs")

plt.rcParams.update({
    "figure.facecolor":  "#1e1e2e",
    "axes.facecolor":    "#2a2a3e",
    "axes.edgecolor":    "#555577",
    "axes.labelcolor":   "#ccccee",
    "axes.titlecolor":   "#ffffff",
    "xtick.color":       "#aaaacc",
    "ytick.color":       "#aaaacc",
    "text.color":        "#ccccee",
    "grid.color":        "#3a3a5a",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.labelsize":    12,
    "legend.facecolor":  "#2a2a3e",
    "legend.edgecolor":  "#555577",
    "legend.labelcolor": "#ccccee",
    "savefig.facecolor": "#1e1e2e",
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
})

CLASS_COLORS = ["#4ade80", "#f87171", "#fb923c", "#facc15"]


def plot_training_loss(history, save_path):
    epochs = history["epochs"]
    loss   = history["train_loss"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, loss, color="#60a5fa", linewidth=2.5,
            label="Training Loss")

    if len(loss) >= 5:
        window = max(5, len(loss) // 10)
        smooth = np.convolve(loss, np.ones(window) / window, mode="valid")
        ax.plot(epochs[window - 1:], smooth,
                color="#f472b6", linewidth=1.5,
                linestyle="--", alpha=0.8,
                label=f"Smoothed (window={window})")

    min_idx = int(np.argmin(loss))
    ax.scatter([epochs[min_idx]], [loss[min_idx]],
               color="#fbbf24", s=80, zorder=5,
               label=f"Min: {loss[min_idx]:.4f} (epoch {epochs[min_idx]})")
    ax.annotate(f"Min: {loss[min_idx]:.4f}",
                xy=(epochs[min_idx], loss[min_idx]),
                xytext=(epochs[min_idx] + max(1, len(epochs)//10),
                        loss[min_idx] + max(loss) * 0.05),
                color="#fbbf24", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#fbbf24"))

    ax.set_title("Training Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_val_accuracy(history, save_path):
    epochs  = history["epochs"]
    val_acc = [v * 100 for v in history["val_accuracy"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, val_acc, color="#34d399", linewidth=2.5,
            label="Validation Accuracy")
    ax.fill_between(epochs, val_acc, alpha=0.15, color="#34d399")
    
    max_idx = int(np.argmax(val_acc))
    ax.scatter([epochs[max_idx]], [val_acc[max_idx]],
               color="#fbbf24", s=100, zorder=5,
               label=f"Best: {val_acc[max_idx]:.1f}% (epoch {epochs[max_idx]})")
    ax.annotate(f"Best: {val_acc[max_idx]:.1f}%",
                xy=(epochs[max_idx], val_acc[max_idx]),
                xytext=(epochs[max_idx] + max(1, len(epochs)//10),
                        val_acc[max_idx] - 8),
                color="#fbbf24", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#fbbf24"))

    ax.axhline(y=90, color="#f87171", linestyle=":",
               alpha=0.6, linewidth=1, label="90% target")
    ax.axhline(y=80, color="#fb923c", linestyle=":",
               alpha=0.4, linewidth=1, label="80% target")

    ax.set_title("Validation Accuracy Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_confusion_matrix(history, save_path):
    preds       = history["val_predictions"]
    trues       = history["val_true_labels"]
    label_names = history["label_names"]
    n           = len(label_names)

    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(trues, preds):
        matrix[t][p] += 1

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_pct = matrix / row_sums * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = LinearSegmentedColormap.from_list(
        "custom", ["#1e1e2e", "#1e3a5f", "#2563eb", "#60a5fa"])
    im = ax.imshow(matrix_pct, cmap=cmap, vmin=0, vmax=100)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (%)", color="#ccccee")
    cbar.ax.yaxis.set_tick_params(color="#ccccee")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([l.replace("_", "\n") for l in label_names], fontsize=11)
    ax.set_yticklabels([l.replace("_", "\n") for l in label_names], fontsize=11)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix\n(row-normalised, % recall per class)")

    for i in range(n):
        for j in range(n):
            color = "white" if matrix_pct[i][j] < 50 else "#0f172a"
            ax.text(j, i,
                    f"{matrix_pct[i][j]:.1f}%\n({matrix[i][j]})",
                    ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_per_class_accuracy(history, save_path):
    per_class   = history["per_class_accuracy"]
    label_names = list(per_class.keys())
    accuracies  = [v * 100 for v in per_class.values()]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(label_names, accuracies,
                  color=CLASS_COLORS[:len(label_names)],
                  edgecolor="#1e1e2e", linewidth=1.5, width=0.55)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{acc:.1f}%",
                ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="white")

    ax.axhline(y=90, color="#f87171", linestyle="--",
               alpha=0.7, linewidth=1.2, label="90% target")
    ax.axhline(y=80, color="#fb923c", linestyle="--",
               alpha=0.5, linewidth=1.0, label="80% target")

    ax.set_title("Per-Class Accuracy on Validation Set")
    ax.set_xlabel("Behaviour Class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_xticklabels([l.replace("_", "\n") for l in label_names], fontsize=11)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_behaviour_distribution(history, save_path):
    dist        = history["class_distribution"]
    label_names = list(dist.keys())
    counts      = list(dist.values())
    total       = sum(counts)

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, _, autotexts = ax.pie(
        counts,
        colors=CLASS_COLORS[:len(label_names)],
        autopct=lambda p: f"{p:.1f}%\n({int(round(p * total / 100))})",
        startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(edgecolor="#1e1e2e", linewidth=2, width=0.6))

    for at in autotexts:
        at.set_fontsize(11)
        at.set_color("white")
        at.set_fontweight("bold")

    ax.legend(wedges,
              [f"{l.replace('_', ' ').title()}  ({c})"
               for l, c in zip(label_names, counts)],
              loc="lower center",
              bbox_to_anchor=(0.5, -0.08),
              ncol=2, fontsize=11, framealpha=0.3)

    ax.text(0, 0, f"Total\n{total}\nsamples",
            ha="center", va="center",
            fontsize=13, color="white", fontweight="bold")

    ax.set_title("Training Data — Behaviour Distribution", pad=20)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


def plot_combined(history, save_path):
    fig = plt.figure(figsize=(22, 16))
    fig.suptitle("Classroom Posture Detection — Training Report",
                 fontsize=18, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    epochs  = history["epochs"]
    loss    = history["train_loss"]
    val_acc = [v * 100 for v in history["val_accuracy"]]

    # Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, loss, color="#60a5fa", linewidth=2)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    # Val accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, val_acc, color="#34d399", linewidth=2)
    ax2.fill_between(epochs, val_acc, alpha=0.15, color="#34d399")
    ax2.axhline(y=90, color="#f87171", linestyle=":", alpha=0.6)
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(True)

    # Pie
    ax3 = fig.add_subplot(gs[0, 2])
    dist   = history["class_distribution"]
    counts = list(dist.values())
    labels = list(dist.keys())
    total  = sum(counts)
    wedges, _, autotexts = ax3.pie(
        counts, colors=CLASS_COLORS[:len(labels)],
        autopct=lambda p: f"{p:.1f}%", startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(edgecolor="#1e1e2e", linewidth=1.5, width=0.6))
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color("white")
    ax3.set_title("Behaviour Distribution")
    ax3.legend(wedges,
               [l.replace("_", " ").title() for l in labels],
               loc="lower center", bbox_to_anchor=(0.5, -0.15),
               ncol=2, fontsize=8)

    # Per-class bar
    ax4 = fig.add_subplot(gs[1, 0])
    per_class = history["per_class_accuracy"]
    cls_names = list(per_class.keys())
    cls_accs  = [v * 100 for v in per_class.values()]
    bars = ax4.bar(range(len(cls_names)), cls_accs,
                   color=CLASS_COLORS[:len(cls_names)],
                   edgecolor="#1e1e2e", width=0.55)
    for bar, acc in zip(bars, cls_accs):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 f"{acc:.0f}%",
                 ha="center", va="bottom",
                 fontsize=9, color="white", fontweight="bold")
    ax4.set_xticks(range(len(cls_names)))
    ax4.set_xticklabels([l.replace("_", "\n") for l in cls_names], fontsize=9)
    ax4.set_ylim(0, 115)
    ax4.set_title("Per-Class Accuracy")
    ax4.set_ylabel("Accuracy (%)")
    ax4.axhline(y=90, color="#f87171", linestyle="--", alpha=0.6)
    ax4.grid(True, axis="y")

    # Confusion matrix
    ax5 = fig.add_subplot(gs[1, 1:])
    preds  = history["val_predictions"]
    trues  = history["val_true_labels"]
    n_cls  = len(cls_names)
    matrix = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(trues, preds):
        matrix[t][p] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_pct = matrix / row_sums * 100
    cmap = LinearSegmentedColormap.from_list(
        "c", ["#1e1e2e", "#1e3a5f", "#2563eb", "#60a5fa"])
    im = ax5.imshow(matrix_pct, cmap=cmap, vmin=0, vmax=100)
    fig.colorbar(im, ax=ax5, fraction=0.03)
    ax5.set_xticks(range(n_cls))
    ax5.set_yticks(range(n_cls))
    ax5.set_xticklabels([l.replace("_", "\n") for l in cls_names], fontsize=9)
    ax5.set_yticklabels([l.replace("_", "\n") for l in cls_names], fontsize=9)
    ax5.set_xlabel("Predicted")
    ax5.set_ylabel("True")
    ax5.set_title("Confusion Matrix (% recall)")
    for i in range(n_cls):
        for j in range(n_cls):
            color = "white" if matrix_pct[i][j] < 50 else "#0f172a"
            ax5.text(j, i,
                     f"{matrix_pct[i][j]:.1f}%\n({matrix[i][j]})",
                     ha="center", va="center",
                     fontsize=9, color=color, fontweight="bold")

    plt.savefig(save_path)
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


def generate_graphs():

    if not os.path.exists(HISTORY_FILE):
        print(f"\n[ERROR] Training history not found: {HISTORY_FILE}")
        print("        Run step5_train_classifier.py first.\n")
        sys.exit(1)

    os.makedirs(GRAPHS_DIR, exist_ok=True)

    print(f"\n[INFO] Loading: {HISTORY_FILE}")
    with open(HISTORY_FILE) as f:
        history = json.load(f)

    required = ["epochs", "train_loss", "val_accuracy",
                "per_class_accuracy", "val_predictions",
                "val_true_labels", "label_names",
                "class_distribution"]
    missing = [k for k in required if k not in history]
    if missing:
        print(f"\n[ERROR] training_history.json missing keys: {missing}")
        print("        Re-run step5_train_classifier.py\n")
        sys.exit(1)

    print(f"[INFO] Saving graphs to: {GRAPHS_DIR}\n")

    print("  1/6  Training Loss Curve...")
    plot_training_loss(history,
        os.path.join(GRAPHS_DIR, "1_training_loss.png"))

    print("  2/6  Validation Accuracy Curve...")
    plot_val_accuracy(history,
        os.path.join(GRAPHS_DIR, "2_validation_accuracy.png"))

    print("  3/6  Confusion Matrix...")
    plot_confusion_matrix(history,
        os.path.join(GRAPHS_DIR, "3_confusion_matrix.png"))

    print("  4/6  Per-Class Accuracy Bar Chart...")
    plot_per_class_accuracy(history,
        os.path.join(GRAPHS_DIR, "4_per_class_accuracy.png"))

    print("  5/6  Behaviour Distribution Pie Chart...")
    plot_behaviour_distribution(history,
        os.path.join(GRAPHS_DIR, "5_behaviour_distribution.png"))

    print("  6/6  Combined report page...")
    plot_combined(history,
        os.path.join(GRAPHS_DIR, "all_graphs_combined.png"))

    print("\n" + "=" * 55)
    print("  ALL GRAPHS SAVED SUCCESSFULLY")
    print("=" * 55)
    print(f"\n  Folder: {GRAPHS_DIR}\n")
    for fname in sorted(os.listdir(GRAPHS_DIR)):
        if fname.endswith(".png"):
            size = os.path.getsize(os.path.join(GRAPHS_DIR, fname)) // 1024
            print(f"    {fname:<45} {size:>4} KB")
    print()
    print("  ► Open the folder in File Explorer to view the graphs.")
    print("  ► Share 'all_graphs_combined.png' with your professor.\n")

    # Open folder automatically on Windows
    try:
        import subprocess
        subprocess.Popen(f'explorer "{os.path.abspath(GRAPHS_DIR)}"')
        print("  [INFO] Output folder opened in File Explorer.")
    except Exception:
        pass


if __name__ == "__main__":
    generate_graphs()