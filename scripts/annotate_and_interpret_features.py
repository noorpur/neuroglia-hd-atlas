from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

PROCESSED = Path("data/processed")
TABLES = Path("reports/tables")
FIGS = Path("reports/figures")
MAIN = FIGS / "main"
TABLES.mkdir(parents=True, exist_ok=True)
MAIN.mkdir(parents=True, exist_ok=True)

def strip_version(feature):
    return str(feature).split(".")[0]

def annotate_features(features):
    base = [strip_version(f) for f in features]
    unique = sorted(set(base))
    ann = {b: {"ensembl_base": b, "symbol": "", "name": "", "entrezgene": "", "type_of_gene": ""} for b in unique}

    try:
        import mygene
        mg = mygene.MyGeneInfo()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mg.querymany(
                unique,
                scopes="ensembl.gene",
                fields="symbol,name,entrezgene,type_of_gene",
                species="human",
                as_dataframe=False,
                verbose=False,
            )
        for r in res:
            q = r.get("query")
            if not q or r.get("notfound"):
                continue
            ann[q] = {
                "ensembl_base": q,
                "symbol": r.get("symbol", ""),
                "name": r.get("name", ""),
                "entrezgene": r.get("entrezgene", ""),
                "type_of_gene": r.get("type_of_gene", ""),
            }
    except Exception as e:
        print(f"WARNING: mygene annotation failed. Continuing without symbols. Error: {e}")

    return pd.DataFrame([ann[strip_version(f)] | {"feature": f} for f in features])

def align_supervised_matrix():
    x = pd.read_parquet(PROCESSED / "feature_matrix.parquet")
    meta = pd.read_parquet(PROCESSED / "sample_metadata.parquet").copy()

    eligible = []
    for dataset, df in meta.groupby("dataset_id"):
        labels = set(df["condition"].astype(str))
        if {"HD", "control"}.issubset(labels):
            eligible.append(dataset)

    meta = meta[meta["dataset_id"].isin(eligible)].reset_index(drop=True)

    if "pseudobulk_id" in meta.columns and set(meta["pseudobulk_id"].astype(str)).issubset(set(x.index.astype(str))):
        x = x.loc[meta["pseudobulk_id"].astype(str)].reset_index(drop=True)
    else:
        x = x.iloc[meta.index].reset_index(drop=True)

    y = (meta["condition"].astype(str) == "HD").astype(int).reset_index(drop=True)
    return x, meta, y, eligible

# 1. Annotate single-feature screen.
single_path = TABLES / "single_feature_leakage_screen.csv"
single = pd.read_csv(single_path)
single_top = single.head(100).copy()
ann_single = annotate_features(single_top["feature"].tolist())
single_annotated = single_top.merge(ann_single, on="feature", how="left")
single_annotated.to_csv(TABLES / "top_single_feature_gene_annotations.csv", index=False)

# 2. Fit conservative 250-feature logistic model for interpretation.
x, meta, y, eligible = align_supervised_matrix()

top_features = x.var(axis=0).sort_values(ascending=False).head(250).index.tolist()
xs = x[top_features].copy()

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=0.1, class_weight="balanced", solver="lbfgs", max_iter=5000)),
])

pipe.fit(xs, y)
coef = pipe.named_steps["clf"].coef_[0]

coef_df = pd.DataFrame({
    "feature": top_features,
    "coefficient": coef,
    "abs_coefficient": np.abs(coef),
})

ann_coef = annotate_features(coef_df["feature"].tolist())
coef_df = coef_df.merge(ann_coef, on="feature", how="left")

single_small = single[["feature", "single_feature_auc_directionless", "raw_auc"]].copy()
coef_df = coef_df.merge(single_small, on="feature", how="left")
coef_df["model"] = "regularized_logistic_250_features"
coef_df["eligible_datasets"] = ";".join(eligible)
coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

coef_df.to_csv(TABLES / "primary_250_feature_logistic_coefficients_annotated.csv", index=False)

# 3. Make coefficient plot.
plot_df = coef_df.head(30).copy()
plot_df["label"] = plot_df["symbol"].where(plot_df["symbol"].fillna("") != "", plot_df["feature"])

plot_df = plot_df.sort_values("coefficient")
plt.figure(figsize=(10, 9))
plt.barh(plot_df["label"], plot_df["coefficient"])
plt.axvline(0, linestyle="--", linewidth=1)
plt.title("Top coefficients: 250-feature regularized logistic model")
plt.xlabel("standardized logistic coefficient")
plt.tight_layout()
plt.savefig(MAIN / "primary_250_feature_logistic_top_coefficients.png", dpi=200)
plt.close()

# 4. Make feature sensitivity plot.
sens_path = TABLES / "strict_supervised_feature_sensitivity.csv"
if sens_path.exists():
    sens = pd.read_csv(sens_path)
    plt.figure(figsize=(9, 5))
    for model, df in sens.groupby("model"):
        df = df.sort_values("n_features")
        plt.plot(df["n_features"], df["test_roc_auc"], marker="o", label=model)
    plt.xscale("log")
    plt.ylim(0.75, 1.0)
    plt.title("Feature-reduced sensitivity analysis")
    plt.xlabel("number of selected features")
    plt.ylabel("nested CV ROC-AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MAIN / "feature_reduced_sensitivity_auc.png", dpi=200)
    plt.close()

print("\n=== Top annotated single-feature screen ===")
print(single_annotated[["feature", "symbol", "name", "single_feature_auc_directionless", "raw_auc"]].head(25).to_string(index=False))

print("\n=== Top annotated 250-feature logistic coefficients ===")
print(coef_df[["feature", "symbol", "name", "coefficient", "abs_coefficient", "single_feature_auc_directionless"]].head(30).to_string(index=False))

print("\nWrote:")
print(TABLES / "top_single_feature_gene_annotations.csv")
print(TABLES / "primary_250_feature_logistic_coefficients_annotated.csv")
print(MAIN / "primary_250_feature_logistic_top_coefficients.png")
print(MAIN / "feature_reduced_sensitivity_auc.png")
