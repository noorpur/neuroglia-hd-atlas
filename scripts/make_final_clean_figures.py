from pathlib import Path
import shutil
import pandas as pd
import matplotlib.pyplot as plt

FIGS = Path("reports/figures")
TABLES = Path("reports/tables")
MAIN = FIGS / "main"
MAIN.mkdir(parents=True, exist_ok=True)

to_copy = [
    "atlas_dataset_composition.png",
    "atlas_pca_dataset.png",
    "atlas_pca_condition.png",
    "overfitting_auc_gap_labelled.png",
    "calibration_curves.png",
    "permutation_label_control.png",
    "single_feature_leakage_screen.png",
    "nested_cv_roc_curves.png",
    "nested_cv_precision_recall_curves.png",
]

for name in to_copy:
    src = FIGS / name
    if src.exists():
        shutil.copy2(src, MAIN / name)

# Clean model comparison without Brier mixed into "higher is better" metrics.
summary = pd.read_csv(TABLES / "nested_cv_summary.csv")
metrics = [
    ("mean_test_auc", "nested ROC-AUC"),
    ("mean_average_precision", "nested average precision"),
    ("mean_balanced_accuracy", "balanced accuracy"),
    ("mean_f1", "F1"),
]
available = [(c, label) for c, label in metrics if c in summary.columns]

plot_df = summary[["model"] + [c for c, _ in available]].rename(columns=dict(available))
ax = plot_df.set_index("model").plot(kind="bar", figsize=(10, 6))
ax.set_title("Nested cross-validation discrimination metrics")
ax.set_ylabel("score")
ax.set_ylim(0, 1.05)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(MAIN / "nested_cv_discrimination_metrics_clean.png", dpi=200)
plt.close()

# Separate Brier score because lower is better.
if "mean_brier" in summary.columns:
    ax = summary.set_index("model")["mean_brier"].plot(kind="bar", figsize=(8, 5))
    ax.set_title("Nested cross-validation Brier score")
    ax.set_ylabel("Brier score, lower is better")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(MAIN / "nested_cv_brier_score_clean.png", dpi=200)
    plt.close()

print("Final clean figures in:", MAIN)
for p in sorted(MAIN.glob("*.png")):
    print(" -", p)
