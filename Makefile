.PHONY: install install-full test lint registry download pseudobulk train report clean

install:
	python -m pip install -e '.[core]'

install-full:
	python -m pip install -e '.[full]'

test:
	pytest -q

lint:
	ruff check src tests

registry:
	neurogliahd registry

download:
	neurogliahd download --config configs/default.yaml

pseudobulk:
	neurogliahd pseudobulk --config configs/default.yaml

train:
	neurogliahd train-baselines --config configs/default.yaml

report:
	neurogliahd report --config configs/default.yaml

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
