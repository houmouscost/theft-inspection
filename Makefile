PY ?= python
export PYTHONPATH := src

.PHONY: install test smoke data run main sens report paper all clean

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

# Run penuh: data -> eksperimen utama (E1-E6) -> sensitivitas (br_iters, S2) -> laporan
run: main sens report

main:
	$(PY) -m tip.pipeline --config configs/default.yaml --stage data
	$(PY) -m tip.pipeline --config configs/default.yaml --stage experiments

# Sensitivitas yang dibaca laporan sgcc_main (report.br_sensitivity, report.level_matched)
sens:
	$(PY) -m tip.pipeline --config configs/sens_br24.yaml --stage data
	$(PY) -m tip.pipeline --config configs/sens_br24.yaml --stage experiments
	$(PY) -m tip.pipeline --config configs/sens_level_matched.yaml --stage data
	$(PY) -m tip.pipeline --config configs/sens_level_matched.yaml --stage experiments

report:
	$(PY) -m tip.pipeline --config configs/default.yaml --stage report

# latexmk (TeX Live) bila ada; jika tidak, tectonic (mengunduh paket sendiri)
paper:
	cd paper && (command -v latexmk >/dev/null && latexmk -pdf -interaction=nonstopmode main.tex || tectonic main.tex)

all: run paper

clean:
	rm -rf artifacts results paper/generated/*.tex paper/generated/*.pdf paper/generated/SUMMARY.md
	cd paper && latexmk -C
