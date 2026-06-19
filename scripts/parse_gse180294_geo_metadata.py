from pathlib import Path
import gzip
import csv
import re
import pandas as pd

path = Path("data/raw/mouse_r62_sn/GSE180294_series_matrix.txt.gz")
out = Path("reports/tables")
out.mkdir(parents=True, exist_ok=True)

wanted = {
    "!Sample_geo_accession": "geo_accession",
    "!Sample_title": "title",
    "!Sample_source_name_ch1": "source_name",
    "!Sample_organism_ch1": "organism",
    "!Sample_characteristics_ch1": "characteristics",
    "!Sample_description": "description",
}

records = {}
characteristics_rows = []

with gzip.open(path, "rt", errors="replace") as f:
    for line in f:
        if not line.startswith("!Sample_"):
            continue
        row = next(csv.reader([line.rstrip("\n")], delimiter="\t"))
        key = row[0]
        values = [v.strip().strip('"') for v in row[1:]]

        if key == "!Sample_characteristics_ch1":
            characteristics_rows.append(values)
            continue

        if key in wanted:
            col = wanted[key]
            for i, v in enumerate(values):
                records.setdefault(i, {})[col] = v

for char_values in characteristics_rows:
    for i, v in enumerate(char_values):
        records.setdefault(i, {}).setdefault("characteristics_all", [])
        records[i]["characteristics_all"].append(v)

rows = []
for i in sorted(records):
    r = records[i]
    chars = " | ".join(r.get("characteristics_all", []))
    text = " | ".join(str(r.get(k, "")) for k in ["geo_accession", "title", "source_name", "description"]) + " | " + chars

    z = text.lower()
    inferred = "unknown"
    confidence = "none"

    if re.search(r"\br6/?2\b|\br6-?2\b|r6/2|r62|transgenic|mutant", z):
        inferred = "HD_model_R6_2"
        confidence = "keyword"
    if re.search(r"\bwt\b|wild type|wild-type|control", z):
        if inferred == "unknown":
            inferred = "control_model"
            confidence = "keyword"
        else:
            inferred = inferred + "_AND_control_keyword"
            confidence = "conflict"

    rows.append({
        "geo_accession": r.get("geo_accession", ""),
        "title": r.get("title", ""),
        "source_name": r.get("source_name", ""),
        "description": r.get("description", ""),
        "characteristics_all": chars,
        "inferred_condition": inferred,
        "inference_confidence": confidence,
        "all_text": text,
    })

df = pd.DataFrame(rows)
df.to_csv(out / "gse180294_geo_sample_metadata.csv", index=False)

summary = df.groupby(["inferred_condition", "inference_confidence"], dropna=False).size().reset_index(name="n")
summary.to_csv(out / "gse180294_geo_label_summary.csv", index=False)

print("\n=== GSE180294 GEO label summary ===")
print(summary.to_string(index=False))

print("\n=== GSE180294 metadata preview ===")
cols = ["geo_accession", "title", "source_name", "characteristics_all", "inferred_condition", "inference_confidence"]
print(df[cols].head(30).to_string(index=False))

print("\nWrote:")
print(out / "gse180294_geo_sample_metadata.csv")
print(out / "gse180294_geo_label_summary.csv")
