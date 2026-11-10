"""Unduh SGCC dari GitHub (henryRDlab/ElectricityTheftDetection) dan ekstrak tanpa `zip -s 0`.

Arsip di sana adalah zip terpecah (data.z01, data.z02, data.zip) berisi SATU entri deflate
`data.csv`. `zip -s 0` + `unzip` menghasilkan CSV TERPOTONG (~33.8 ribu dari 42.372 baris) dengan
pesan galat yang mudah terlewat. Skrip ini menyambung ketiga bagian, membaca local file header,
meng-inflate aliran deflate secara langsung, lalu memeriksa ukuran dan CRC32 terhadap header.

    python scripts/fetch_sgcc_github.py [--out data/raw/sgcc/data.csv]
"""
from __future__ import annotations

import argparse
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

BASE = "https://raw.githubusercontent.com/henryRDlab/ElectricityTheftDetection/master/"
PARTS = ["data.z01", "data.z02", "data.zip"]   # urutan zip terpecah: .z01, .z02, lalu .zip


def fetch(cache: Path) -> bytes:
    cache.mkdir(parents=True, exist_ok=True)
    chunks = []
    for name in PARTS:
        f = cache / name
        if not f.exists():
            print(f"[..] mengunduh {name}", flush=True)
            with urllib.request.urlopen(BASE + name, timeout=300) as r:
                f.write_bytes(r.read())
        chunks.append(f.read_bytes())
    return b"".join(chunks)


def inflate_single_entry(raw: bytes, out: Path) -> None:
    off = 4 if raw[:4] == b"PK\x07\x08" else 0          # penanda spanning
    if raw[off:off + 4] != b"PK\x03\x04":
        raise ValueError("local file header tidak ditemukan")
    (_, _flag, method, _, _, crc, _csz, usz, nl, el) = struct.unpack("<HHHHHIIIHH", raw[off + 4:off + 30])
    name = raw[off + 30:off + 30 + nl].decode()
    if method != 8:
        raise ValueError(f"metode kompresi {method} tidak didukung")
    start = off + 30 + nl + el
    dec = zlib.decompressobj(-15)
    c, n = 0, 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fo:
        pos = start
        while pos < len(raw) and not dec.eof:
            buf = dec.decompress(raw[pos:pos + (1 << 20)])
            c, n = zlib.crc32(buf, c), n + len(buf)
            fo.write(buf)
            pos += 1 << 20
    if not dec.eof:
        raise ValueError("aliran deflate tidak lengkap")
    # bila bit 3 flag diset, crc/ukuran ada di data descriptor dan header bisa nol
    if usz and n != usz:
        raise ValueError(f"ukuran {n} != header {usz}")
    if crc and c != crc:
        raise ValueError(f"CRC {c:#x} != header {crc:#x}")
    print(f"[ok] {name} -> {out}  ({n:,} byte, CRC32 {c:#010x})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/sgcc/data.csv")
    ap.add_argument("--cache", default="data/raw/sgcc/_parts")
    a = ap.parse_args()
    inflate_single_entry(fetch(Path(a.cache)), Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
