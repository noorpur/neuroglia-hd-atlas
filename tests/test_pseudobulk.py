import pandas as pd

from neuroglia_hd.features.pseudobulk import pseudobulk_from_frame


def test_pseudobulk_groups_cells_and_filters_min_cells():
    expr = pd.DataFrame(
        {"gene_a": [1, 2, 3, 4, 100], "gene_b": [2, 3, 4, 5, 100]},
        index=["c1", "c2", "c3", "c4", "c5"],
    )
    meta = pd.DataFrame(
        {
            "donor_id": ["d1", "d1", "d2", "d2", "d3"],
            "cell_type": ["astro", "astro", "micro", "micro", "astro"],
            "condition": ["control", "control", "hd", "hd", "hd"],
        },
        index=expr.index,
    )
    result = pseudobulk_from_frame(expr, meta, ["donor_id", "cell_type", "condition"], min_cells=2)
    assert result.expression.shape[0] == 2
    assert "d3|astro|hd" not in result.expression.index
    assert result.counts["n_cells"].tolist() == [2, 2]
