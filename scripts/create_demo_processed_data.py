"""Create tiny synthetic processed files for smoke-testing the modelling CLI.

This is not scientific data. It only verifies that the training/report pipeline works.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
root = Path(__file__).resolve().parents[1]
out = root / "data" / "processed"
out.mkdir(parents=True, exist_ok=True)

n = 40
features = pd.DataFrame(
    rng.normal(size=(n, 30)),
    index=[f"pb_{i:03d}" for i in range(n)],
    columns=[f"gene::G{i:03d}" for i in range(25)] + [f"signature::S{i:02d}" for i in range(5)],
)
condition = np.array(["control"] * 20 + ["hd"] * 20)
features.loc[condition == "hd", "signature::S00"] += 1.2
metadata = pd.DataFrame(
    {
        "donor_id": [f"d{i//2:02d}" for i in range(n)],
        "condition": condition,
        "brain_region": np.tile(["caudate", "frontal"], n // 2),
        "cell_type": np.tile(["astrocyte", "microglia", "oligodendrocyte", "opc"], n // 4),
    },
    index=features.index,
)
features.to_parquet(out / "feature_matrix.parquet")
metadata.to_parquet(out / "sample_metadata.parquet")
print(f"Wrote demo processed data to {out}")
