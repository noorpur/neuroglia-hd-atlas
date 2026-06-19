import py_compile
from pathlib import Path


def test_big_pipeline_script_compiles():
    py_compile.compile("scripts/run_research_grade_big_pipeline.py", doraise=True)


def test_big_pipeline_readme_exists():
    assert Path("BIG_PIPELINE_README.md").exists()
