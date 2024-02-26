.ONESHELL:
ENV_PREFIX=.venv/bin
FILES='.'
PYTEST_ARGS=--cov-config .coveragerc --cov-report xml --cov=alexis --cov-append --cov-report term-missing --cov-fail-under=95 --no-cov-on-fail

.PHONY: help
help:		## Show this help message.
	@echo "Usage: make <target>"
	@echo "\nTargets:"
	@fgrep "##" Makefile | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##/ -/'

.PHONY: install
install:	## Install the project dependencies.
	pip install poetry
	poetry install --with=dev,test

.PHONY: run
run:		## Run the project.
	uvicorn --factory alexis.app:create_app --reload

.PHONY: test
test:		## Run the tests.
	export ALEXIS_ENV=test
	pytest $(PYTEST_ARGS) $(FILES)


