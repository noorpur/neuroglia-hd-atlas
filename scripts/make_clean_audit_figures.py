from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

TABLES = Path("reports/tables")
FIGURES = Path("reports/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

overfit = pd.read_csv(TABLES / "overfitting_diagnostics.csv")
overfit = overfit.sort_values("mean_auc_gap", ascending=True)

plt.figure(figsize=(8, 4.8))
bars = plt.barh(overfit["model"], overfit["mean_auc_gap"])
plt.axvline(0.02, linestyle=":", label="low concern threshold")
plt.axvline(0.08, linestyle="--", label="high concern threshold")
plt.title("Overfitting audit: train-test ROC-AUC gap")
plt.xlabel("mean train ROC-AUC - mean test ROC-AUC")
plt.ylabel("model")

for bar, val in zip(bars, overfit["mean_auc_gap"]):
    plt.text(
        bar.get_width() + 0.0005,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}",
        va="center",
        fontsize=9,
    )

plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIGURES / "overfitting_auc_gap_labelled.png", dpi=240, bbox_inches="tight")
plt.close()

print("Wrote reports/figures/overfitting_auc_gap_labelled.png")
