from pathlib import Path
import shutil
import re

src_root = Path("data/interim/extracted")
out_root = Path("data/interim/standardized_10x")
out_root.mkdir(parents=True, exist_ok=True)

matrix_files = sorted(src_root.rglob("*matrix*.mtx*"))
made = 0

for m in matrix_files:
    parent = m.parent
    name = m.name

    candidates = list(parent.glob("*barcodes*.tsv*"))
    features = list(parent.glob("*features*.tsv*")) or list(parent.glob("*genes*.tsv*"))

    if not candidates or not features:
        continue

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", m.stem)
    out = out_root / safe
    out.mkdir(parents=True, exist_ok=True)

    shutil.copy2(m, out / ("matrix.mtx.gz" if m.name.endswith(".gz") else "matrix.mtx"))
    shutil.copy2(candidates[0], out / ("barcodes.tsv.gz" if candidates[0].name.endswith(".gz") else "barcodes.tsv"))
    shutil.copy2(features[0], out / ("features.tsv.gz" if features[0].name.endswith(".gz") else "features.tsv"))

    made += 1
    print("Prepared", out)

print("Prepared 10x directories:", made)
