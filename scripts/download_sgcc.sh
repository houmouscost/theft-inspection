#!/usr/bin/env bash
# Unduh dataset SGCC (Zheng et al., IEEE TII 2018) ke data/raw/sgcc/data.csv
# Dua sumber publik yang diketahui:
#   (1) Kaggle: bensalem14/sgcc-dataset  -> butuh `pip install kaggle` + ~/.kaggle/kaggle.json
#   (2) GitHub: henryRDlab/ElectricityTheftDetection -> tiga berkas data.zip, data.z01, data.z02
# Skrip ini mencoba (1); bila gagal, mencetak langkah manual untuk (2).
set -euo pipefail
OUT="data/raw/sgcc"
mkdir -p "$OUT"

if [ -f "$OUT/data.csv" ]; then
  echo "[ok] $OUT/data.csv sudah ada"; exit 0
fi

if command -v kaggle >/dev/null 2>&1; then
  echo "[..] mengunduh via Kaggle"
  kaggle datasets download -d bensalem14/sgcc-dataset -p "$OUT" --unzip || true
  # nama berkas di Kaggle bisa 'data set.csv'
  for f in "$OUT/data set.csv" "$OUT/data.csv" "$OUT"/*.csv; do
    if [ -f "$f" ] && [ "$f" != "$OUT/data.csv" ]; then mv "$f" "$OUT/data.csv"; break; fi
  done
fi

if [ ! -f "$OUT/data.csv" ]; then
  cat <<'EOF'
[!] Unduhan otomatis tidak tersedia. Langkah manual:
  A) Kaggle  : unduh "SGCC Electricity Theft Detection" (bensalem14), simpan sebagai data/raw/sgcc/data.csv
  B) GitHub  : https://github.com/henryRDlab/ElectricityTheftDetection
               unduh data.zip, data.z01, data.z02 ke satu folder, lalu:
                 zip -s 0 data.zip --out merged.zip && unzip merged.zip -d data/raw/sgcc
               (pastikan berkas CSV bernama data.csv dengan kolom CONS_NO, FLAG, dan kolom-kolom tanggal)
Setelah itu: make run
EOF
  exit 1
fi
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/raw/sgcc/data.csv", nrows=3)
print("[ok] kolom:", df.shape[1], "contoh:", list(df.columns[:3]), "...", list(df.columns[-2:]))
PY
