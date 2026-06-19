from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def write_markdown_report(
    output: str | Path,
    metrics: pd.DataFrame | None = None,
    qc: pd.DataFrame | None = None,
    notes: list[str] | None = None,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NeuroGlia-HD Atlas Analysis Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Notes",
    ]
    for note in notes or ["Pipeline completed. Replace this section with interpreted results after review."]:
        lines.append(f"- {note}")
    if qc is not None and not qc.empty:
        lines.extend(["", "## QC summary", "", qc.head(20).to_markdown(index=False)])
    if metrics is not None and not metrics.empty:
        lines.extend(["", "## Model metrics", "", metrics.to_markdown(index=False)])
    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "These results are exploratory and not for clinical decision-making. Any candidate mechanism or biomarker should be validated in independent cohorts and, where appropriate, functional experiments.",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
