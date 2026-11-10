PY ?= python
export PYTHONPATH := src

.PHONY: install test smoke data run main sens uci report paper all clean

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

# Run penuh: data -> eksperimen utama (E1-E9) -> sensitivitas/ekstensi (br_iters, S2, batas r, CNN, onset S3, UCI) -> laporan
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
	for c in sens_rmax05 sens_rmax07 sens_cnn sens_onset; do \
	  $(PY) -m tip.pipeline --config configs/$$c.yaml --stage data && \
	  $(PY) -m tip.pipeline --config configs/$$c.yaml --stage experiments || exit 1; done
	$(MAKE) uci

# Replikasi di luar SGCC (UCI ElectricityLoadDiagrams20112014, tanpa label)
uci:
	$(PY) scripts/fetch_uci_eld.py
	$(PY) -m tip.pipeline --config configs/uci.yaml --stage data
	$(PY) -m tip.pipeline --config configs/uci.yaml --stage experiments

report:
	$(PY) -m tip.pipeline --config configs/default.yaml --stage report

# latexmk (TeX Live) bila ada; jika tidak, tectonic (mengunduh paket sendiri)
paper:
	cd paper && (command -v latexmk >/dev/null && latexmk -pdf -interaction=nonstopmode main.tex || tectonic main.tex)

all: run paper

clean:
	rm -rf artifacts results paper/generated/*.tex paper/generated/*.pdf paper/generated/SUMMARY.md
	cd paper && latexmk -C
