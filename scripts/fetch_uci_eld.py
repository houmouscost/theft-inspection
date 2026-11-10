"""Unduh UCI ElectricityLoadDiagrams20112014 (370 pelanggan Portugal, 15 menit) -> kWh harian.

Data kedua di luar SGCC untuk uji generalisasi (ekstensi 5). TIDAK ada label pencurian: detektor
dilatih pada serangan sintetis (praktik umum literatur NTL), dan semua pencuri adalah injeksi S1.
Nilai asli dalam kW per interval 15 menit -> kWh = kW / 4. Jendela 2012-01-01..2014-12-31; pelanggan
yang belum aktif (nol sepanjang Januari 2012) dibuang supaya tidak ada "awal" palsu yang mirip onset.

    python scripts/fetch_uci_eld.py [--out data/raw/uci/uci_daily.npz]
Sitasi: A. Trindade, "ElectricityLoadDiagrams20112014", UCI Machine Learning Repository, 2015,
doi:10.24432/C58C86.
"""
from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/uci/uci_daily.npz")
    ap.add_argument("--zip", default="data/raw/uci/eld.zip")
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--end", default="2014-12-31")
    a = ap.parse_args()
    zp = Path(a.zip)
    zp.parent.mkdir(parents=True, exist_ok=True)
    if not zp.exists():
        print("[..] mengunduh", URL, flush=True)
        with urllib.request.urlopen(URL, timeout=600) as r:
            zp.write_bytes(r.read())
    with zipfile.ZipFile(zp) as z:
        with z.open("LD2011_2014.txt") as f:
            df = pd.read_csv(io.TextIOWrapper(f), sep=";", decimal=",", index_col=0, parse_dates=True)
    # stempel 00:15..24:00 -> geser 15 menit supaya tiap interval masuk ke hari kalendernya
    df.index = df.index - pd.Timedelta(minutes=15)
    daily = (df / 4.0).resample("D").sum().loc[a.start:a.end]
    active = (daily.loc[a.start:pd.Timestamp(a.start) + pd.Timedelta(days=30)] > 0).any()
    daily = daily.loc[:, active]
    X = daily.T.to_numpy(np.float32)
    np.savez_compressed(a.out, X=X, consumers=np.array(daily.columns), dates=np.array(daily.index.astype(str)))
    print(f"[ok] {a.out}: {X.shape[0]} pelanggan x {X.shape[1]} hari "
          f"(dibuang {int((~active).sum())} yang belum aktif)")


if __name__ == "__main__":
    main()
