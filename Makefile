PROJECT_ROOT := $(shell pwd)
VENV := $(PROJECT_ROOT)/.venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
# Prefer Python 3.10+ (required for torch 2.6+ and sentence_transformers); avoid conda's python for venv
VENV_PYTHON := $(shell command -v python3.12 2>/dev/null || command -v python3.11 2>/dev/null || command -v python3.10 2>/dev/null || command -v /usr/bin/python3 2>/dev/null || command -v python3 2>/dev/null)

# Convenience flag: `make chat debug=1` or `make chat DEBUG=1`
DEBUG ?= 0
ifdef debug
DEBUG := $(debug)
endif

.PHONY: venv install ingest ingest-sql query chat reingest-chinook-mysql

venv:
	@test -d $(VENV) || $(VENV_PYTHON) -m venv $(VENV)

install: venv
	@$(PY) -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" || (echo "Error: Python 3.10+ required. Install with: brew install python@3.11" && exit 1)
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt

ingest:
	@PYTHONPATH=$(PROJECT_ROOT)/src $(PY) -m chatbot.cli.ingest --collection $(collection)

ingest-sql:
	@PYTHONPATH=$(PROJECT_ROOT)/src $(PY) -m chatbot.cli.ingest_sql $(args)

query:
	@PYTHONPATH=$(PROJECT_ROOT)/src $(PY) -m chatbot.cli.query "$(q)" --collection "$(collection)" $(args)

chat:
	@PYTHONPATH=$(PROJECT_ROOT)/src $(PY) -m chatbot.cli.chat --collection "$(collection)" $(args) $(if $(filter 1 true TRUE yes YES,$(DEBUG)),--debug,)

reingest-chinook-mysql:
	@$(PY) reingest_chinook_mysql.py

eval_ablate:
	@PYTHONPATH=$(PROJECT_ROOT)/src $(PY) eval/runner.py --db-uri "$${DB_URI}" --modes "bm25,prf,qexp" --k 10

ci_gate:
	@bash scripts/ci_eval_gate.sh


