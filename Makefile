.PHONY: clean clean-test clean-pyc clean-build docs servedocs help coverage dependency_graph tox
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys
from urllib.request import pathname2url
webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache/

lint: ## check style with flake8
	flake8 skportfolio tests

test: ## run tests quickly with the default Python
	pip install -r requirements_dev.txt
	pytest

tox: ## run tests on every Python version with tox
	tox --parallel auto

coverage: ## check code coverage quickly with the default Python
	coverage run --source skportfolio -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate mkdocs documentation, including API docs
	mkdocs build

servedocs: docs ## compile the docs watching for changes
	mkdocs serve

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python setup.py install
	pip install -r requirements_dev.txt

dependency_graph:
	mkdir -p .pyreverse
	pyreverse --colorized skportfolio --project scikit-portfolio -d .pyreverse -k
	sed -i 's/rankdir=BT/rankdir=RL/g' .pyreverse/packages_scikit-portfolio.dot
	sed -i 's/rankdir=BT/rankdir=RL/g' .pyreverse/classes_scikit-portfolio.dot
	dot -Tsvg .pyreverse/packages_scikit-portfolio.dot > docs/imgs/packages_scikit-portfolio.svg
	dot -Tsvg .pyreverse/classes_scikit-portfolio.dot > docs/imgs/classes_scikit-portfolio.svg
