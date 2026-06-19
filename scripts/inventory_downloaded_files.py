from pathlib import Path
from collections import Counter

roots = [Path("data/raw"), Path("data/interim/extracted")]

files = []
for root in roots:
    if root.exists():
        files.extend([p for p in root.rglob("*") if p.is_file()])

print("Total files:", len(files))
print("\nExtension counts:")
for k, v in Counter(["".join(p.suffixes[-3:]) for p in files]).most_common(30):
    print(v, k)

patterns = [
    "*matrix*.mtx*",
    "*barcodes*.tsv*",
    "*features*.tsv*",
    "*genes*.tsv*",
    "*.h5",
    "*.h5ad",
    "*.rds",
    "*.rds.gz",
    "*.csv.gz",
    "*.txt.gz",
]

for pat in patterns:
    hits = sorted([p for root in roots if root.exists() for p in root.rglob(pat)])
    print(f"\n=== {pat}: {len(hits)} ===")
    for h in hits[:40]:
        print(h)
