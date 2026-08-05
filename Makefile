.PHONY: install test smoke baseline aggregate

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

smoke:
	python scripts/run_burgers_cpinn.py --profile smoke --seed 0 --output-root outputs

baseline:
	python scripts/run_burgers_cpinn.py --profile baseline --seed 0 --output-root outputs

aggregate:
	python scripts/aggregate_results.py --summary outputs/experiment_summary.csv
