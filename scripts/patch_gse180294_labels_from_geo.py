from pathlib import Path
import pandas as pd
import re

TABLES = Path("reports/tables")
meta_path = TABLES / "gse180294_geo_sample_metadata.csv"

if not meta_path.exists():
    raise FileNotFoundError("Missing reports/tables/gse180294_geo_sample_metadata.csv. Run the GEO metadata parser first.")

df = pd.read_csv(meta_path)

def infer_condition(row):
    text = " | ".join(str(row.get(c, "")) for c in df.columns).lower()

    # R6/2 HD mouse model
    if re.search(r"r6/2|r6-2|r62|_hd_|condition:\s*r6/2|transgenic|mutant", text):
        return "HD_model_R6_2", "explicit_R6_2"

    # NT in this GEO metadata appears to be the non-transgenic/wild-type comparison group.
    if re.search(r"condition:\s*nt\b|\bnt\b|_wt_|wild type|wild-type|\bwt\b|control", text):
        return "control_model_NT", "explicit_NT_or_WT"

    return "unknown", "none"

pairs = df.apply(infer_condition, axis=1)
df["patched_condition"] = [p[0] for p in pairs]
df["patched_confidence"] = [p[1] for p in pairs]

summary = (
    df.groupby(["patched_condition", "patched_confidence"], dropna=False)
    .size()
    .reset_index(name="n")
)

df.to_csv(TABLES / "gse180294_geo_sample_metadata_patched.csv", index=False)
summary.to_csv(TABLES / "gse180294_geo_label_summary_patched.csv", index=False)

print("=== Patched GSE180294 label summary ===")
print(summary.to_string(index=False))

print("\n=== Preview ===")
cols = ["geo_accession", "title", "source_name", "characteristics_all", "patched_condition", "patched_confidence"]
print(df[cols].to_string(index=False))
