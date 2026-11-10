PY ?= python
export PYTHONPATH := src

.PHONY: install test smoke data run main diag fs interval sens uci report paper all clean

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
run: main diag fs interval sens report

main:
	$(PY) -m tip.pipeline --config configs/default.yaml --stage data
	$(PY) -m tip.pipeline --config configs/default.yaml --stage experiments

# Sensitivitas yang dibaca laporan sgcc_main (report.br_sensitivity, report.level_matched)
# Diagnostik data SGCC (seleksi label, kebersihan, robust thd konsumen berbeban nol)
diag:
	$(PY) -m tip.pipeline --config configs/default.yaml --stage diag

# E4/E5 dengan estimator sampel-hingga (TV-KDE, BA cross-fit, null permutasi, celah cakupan KS); murah, aliran acak utama tak berubah
fs:
	$(PY) -m tip.pipeline --config configs/diag_e4e5.yaml --stage data
	$(PY) -m tip.pipeline --config configs/diag_e4e5.yaml --stage experiments

# Aturan interval 3-parameter (S1 dan S2); E5 ikut dijalankan pada S1 supaya aliran acak identik dengan run utama
interval:
	for c in diag_interval diag_interval_s2; do \
	  $(PY) -m tip.pipeline --config configs/$$c.yaml --stage data && \
	  $(PY) -m tip.pipeline --config configs/$$c.yaml --stage experiments || exit 1; done

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
