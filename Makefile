PY ?= python
export PYTHONPATH := src

.PHONY: install test smoke data run report paper all clean

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest

# Uji asap pada data sintetis (hasil BUKAN untuk paper)
smoke:
	$(PY) -m tip.pipeline --config configs/smoke.yaml --stage all

# Unduh SGCC (lihat README) lalu jalankan pipeline penuh
data:
	bash scripts/download_sgcc.sh

run:
	$(PY) -m tip.pipeline --config configs/default.yaml --stage all

report:
	$(PY) -m tip.pipeline --config configs/default.yaml --stage report

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

all: run paper

clean:
	rm -rf artifacts results paper/generated/*.tex paper/generated/*.pdf paper/generated/SUMMARY.md
	cd paper && latexmk -C
