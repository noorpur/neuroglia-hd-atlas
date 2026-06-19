# Contributing

This is a research repository. Contributions should preserve reproducibility and avoid overstating biological or clinical claims.

Before opening a pull request:

1. run `pytest -q`;
2. run `ruff check src tests`;
3. update relevant docs when changing methods;
4. do not commit raw public datasets or derived files that violate source terms;
5. keep model claims tied to generated validation artefacts.
