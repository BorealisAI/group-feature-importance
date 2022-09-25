.PHONY : conda
conda :
	conda env create -f environment.yaml --force

.PHONY : install
install :
	poetry install --no-interaction

.PHONY : update
update :
	-poetry self update
	poetry update
	-pre-commit autoupdate

.PHONY : lint
lint :
	flake8 .

.PHONY : format
format :
	black --check .

.PHONY : test
test :
	python -m pytest --cov-report html --cov-report xml --cov-report term --cov=groufi tests/

.PHONY : publish
publish :
	poetry config pypi-token.pypi ${GROUFI_PYPI_TOKEN}
	poetry publish --build
