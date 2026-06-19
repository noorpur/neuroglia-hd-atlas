import pandas as pd

from neuroglia_hd.features.signatures import score_signatures


def test_signature_scoring_handles_missing_genes():
    expr = pd.DataFrame(
        {
            "FAN1": [1.0, 2.0, 3.0],
            "MLH1": [2.0, 2.0, 2.0],
            "GFAP": [4.0, 2.0, 0.0],
        },
        index=["s1", "s2", "s3"],
    )
    scores = score_signatures(expr, {"dna": ["FAN1", "MLH1", "MISSING"], "astro": ["GFAP"]})
    assert list(scores.columns) == ["dna", "astro"]
    assert scores.shape == (3, 2)
    assert scores.attrs["coverage"].loc[0, "n_present"] == 2
