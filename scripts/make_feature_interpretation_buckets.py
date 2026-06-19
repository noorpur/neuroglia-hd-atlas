from pathlib import Path
import pandas as pd

TABLES = Path("reports/tables")
coef_path = TABLES / "primary_250_feature_logistic_coefficients_annotated.csv"

if not coef_path.exists():
    raise FileNotFoundError("Missing primary_250_feature_logistic_coefficients_annotated.csv")

df = pd.read_csv(coef_path)

def bucket(symbol, name):
    s = str(symbol).upper()
    n = str(name).lower()

    if s in {"DCN", "PRELP", "EFEMP1", "VWA1", "RELN"} or any(k in n for k in ["collagen", "matrix", "extracellular", "leucine rich repeat", "reelin"]):
        return "extracellular_matrix_or_tissue_structure"

    if s.startswith("MT1") or s in {"MT2A", "SELENOP"} or "metallothionein" in n or "selenoprotein" in n:
        return "metal_ion_oxidative_stress"

    if s in {"DUSP1", "GADD45B", "FOS", "ZFP36", "HIF3A", "NFKBIA", "SESN1"}:
        return "stress_inflammation_immediate_early"

    if s in {"PCP4", "CTXN1", "PPP1R14A", "DIO2", "APBB3", "MRAS"}:
        return "neuronal_signalling_or_cell_state"

    if s in {"HBB"} or "hemoglobin" in n:
        return "blood_or_sample_composition_watchlist"

    if "pseudogene" in n:
        return "pseudogene_or_mapping_watchlist"

    return "other_candidate"

df["interpretation_bucket"] = [
    bucket(row.get("symbol", ""), row.get("name", ""))
    for _, row in df.iterrows()
]

bucket_summary = (
    df.head(100)
    .groupby("interpretation_bucket")
    .size()
    .reset_index(name="n_top100_features")
    .sort_values("n_top100_features", ascending=False)
)

df.to_csv(TABLES / "primary_250_feature_logistic_coefficients_bucketed.csv", index=False)
bucket_summary.to_csv(TABLES / "primary_250_feature_bucket_summary.csv", index=False)

print("=== Bucket summary, top 100 coefficients ===")
print(bucket_summary.to_string(index=False))

print("\n=== Top 30 bucketed coefficients ===")
cols = ["feature", "symbol", "name", "coefficient", "abs_coefficient", "single_feature_auc_directionless", "interpretation_bucket"]
print(df[cols].head(30).to_string(index=False))
