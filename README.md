# Inspeksi Pencurian Listrik di Bawah Adaptasi Pencuri

Repositori riset dan draf paper: **kebijakan inspeksi berbantuan ML yang kokoh terhadap pencuri listrik yang beradaptasi** (permainan Stackelberg pada data SGCC).

> **Status: pipeline penuh + ekstensi untuk IEEE Trans. Smart Grid (kebijakan sadar-strategi, skenario onset, rasionalitas terbatas, CNN, batas kedalaman, replikasi UCI) sudah dijalankan dan paper (10 halaman) diisi dari hasilnya.** Tersisa: nama penulis, URL repositori, verifikasi `refs.bib` dan kuartil jurnal.
> Angka di folder `results/smoke` berasal dari data sintetis dan **tidak boleh** masuk paper.
> Semua tabel dan gambar di paper dihasilkan otomatis dari CSV; tidak ada angka yang ditulis tangan.

---

## 1. Pertanyaan riset

Detektor pencurian listrik umumnya dinilai sebagai pengklasifikasi statis (AUC/F1). Padahal utilitas hanya bisa mengunjungi sebagian kecil pelanggan (anggaran inspeksi `b`), dan pencuri adalah agen yang menyesuaikan diri terhadap detektor. Repositori ini menjawab tiga pertanyaan:

| # | Pertanyaan | Di mana dijawab |
|---|-----------|-----------------|
| Q1 | Seberapa besar energi curian yang lolos (NTL lolos) ketika pencuri *best-respond* terhadap kebijakan inspeksi yang dipublikasikan? | E2, E3 |
| Q2 | Apakah campuran "peringkat ML + lantai acak" (`eps`) memberi jaminan kokoh tanpa mengorbankan kinerja bila pencuri tidak adaptif? | E2, E3, Prop. 2 |
| Q3 | Seberapa andal sertifikat tingkat-lolos ketika adversary saat deployment lebih kuat daripada saat kalibrasi? | E5 |

Landasan teori (di `paper/main.tex`):
1. **Prop. 1 (batas mimikri)**: pencuri yang melapor `y = (1-r)x` tidak mengubah fitur *shape* sama sekali, sehingga akurasi seimbang detektor apa pun ≤ (1 + TV)/2, dengan TV jarak variasi total antara distribusi *level* jujur dan *level* pencuri yang tergeser `log(1-r)`.
2. **Prop. 2 (lantai kokoh + konsistensi)**: `p_i ≥ eps·b` untuk semua orang, dan `R(eps) ≥ (1-eps)·R(0)` untuk pencuri non-adaptif.
3. **Prop. 3 (deterensi, Becker)**: bila `eps·b ≥ 1/(1+phi)`, pencuri risk-neutral tidak mencuri sama sekali. Jadi saran ML hanya berarti pada rezim `b(1+phi) < 1`.

**Yang sengaja tidak diklaim:** arsitektur detektor baru. Kontribusinya adalah cara evaluasi yang benar dan alokasi inspeksi saat detektor bisa dikelabui.

## 2. Struktur repositori

```
configs/        default.yaml (SGCC penuh), smoke.yaml (sintetis), sens_*.yaml (sensitivitas, S2, S3, CNN), uci.yaml,
                diag_e4e5.yaml (E4/E5 sampel-hingga), diag_interval(.s2).yaml (aturan interval)
src/tip/
  data/         loader SGCC, pembersihan, split per konsumen, generator sintetis
  features.py   fitur SHAPE (invarian skala) vs LEVEL (peka skala) vs missingness
  detector.py   gradient boosting + kalibrasi Platt
  adversary.py  A1 skala konstan, A2 skala bertahap, A3 mimikri bentuk, A4-L onset eksogen + ramp L
  policies.py   keluarga kebijakan P(eps, tau) dengan anggaran tepat
  game.py       best-response pencuri + keseimbangan Stackelberg + quantal response
  stackelberg.py kebijakan sadar-strategi per sel (skor x level): pencarian lokal + MILP DOBSS
  deep.py       pembanding Wide & Deep CNN (PyTorch, CPU)
  theory.py     TV mimikri (histogram dan KDE), KS, akurasi seimbang cross-fit, null permutasi, ambang deterensi
  conformal.py  sertifikat miss-rate (split-conformal + Hoeffding) + jarak Kolmogorov antar pool miss
  diagnostics.py seleksi label (desil konsumsi, kelas fraksi-hilang), kebersihan data SGCC, AUC tanpa konsumen berbeban nol, syarat perlu deterensi
  pipeline.py   orkestrasi: data -> eksperimen -> laporan
  report.py     gambar PDF + tabel LaTeX + makro angka (numbers.tex)
  report_ext.py tabel/gambar ekstensi (E7, E9, S3, UCI, CNN, batas r)
scripts/        download_sgcc.sh, fetch_sgcc_github.py (ekstraksi zip terpecah + cek CRC)
tests/          uji unit + uji asap pipeline
paper/          main.tex (IEEEtran, 10 halaman), supplement.tex (bahan pendukung), refs.bib, generated/ (diisi pipeline)
docs/           literature_review.md (studi literatur + status rekomendasi)
```

## 3. Pemasangan

```bash
python -m venv .venv && source .venv/bin/activate
make install          # numpy, pandas, scipy, scikit-learn, matplotlib, pyyaml, pytest
make test             # 16 uji
```

Tidak butuh GPU. Detektor memakai `HistGradientBoostingClassifier` (scikit-learn).

## 4. Data

Dataset: **SGCC** (State Grid Corporation of China), 42.372 pelanggan, 1.034 hari, ±8,5% berlabel pencuri, ±25% sel kosong. Dirilis oleh Zheng dkk. (IEEE TII 2018); **wajib disitasi** bila dipakai.

Sumber publik yang dijumpai saat riset:
- GitHub: `henryRDlab/ElectricityTheftDetection` (tiga berkas `data.zip`, `data.z01`, `data.z02`, harus diekstrak bersamaan).
- Kaggle: `bensalem14/sgcc-dataset`.

```bash
bash scripts/download_sgcc.sh     # mencoba Kaggle; jika gagal, mencetak langkah manual
```

Hasil akhir harus berupa `data/raw/sgcc/data.csv` dengan kolom `CONS_NO`, `FLAG`, lalu kolom-kolom tanggal. Loader mengurutkan kolom tanggal sendiri.

> Loader SGCC sudah diuji pada berkas asli (sumber GitHub; 42.372 baris, 1.034 kolom tanggal, CRC32 cocok). **Jangan** menggabung arsip GitHub dengan `zip -s 0` + `unzip`: hasilnya CSV terpotong (~33,8 ribu baris) dengan galat yang mudah terlewat. `scripts/fetch_sgcc_github.py` meng-inflate aliran deflate langsung dan memeriksa ukuran + CRC; `download_sgcc.sh` memakainya otomatis bila Kaggle gagal.

## 5. Menjalankan

```bash
make smoke      # data sintetis, 2 seed, ~15 detik. Hanya untuk memeriksa pipeline.
make run        # SGCC penuh: main (E1-E6) + sens (br_iters=24, skenario S2) + laporan, ~25 menit di 8 core
make main       # hanya run utama configs/default.yaml
make diag       # diagnostik data SGCC (seleksi label, kebersihan, AUC tanpa beban nol), ~3 menit
make fs         # E4/E5 dengan estimator sampel-hingga, ~3 menit; aliran acak utama tidak berubah
make interval   # aturan interval 3-parameter pada S1 dan S2, ~3 menit
make sens       # hanya run sensitivitas (configs/sens_br24.yaml, configs/sens_level_matched.yaml)
make paper      # kompilasi paper/main.pdf dan paper/supplement.pdf (butuh IEEEtran, ada di TeX Live/Overleaf)
```

Per tahap:

```bash
export PYTHONPATH=src
python -m tip.pipeline --config configs/default.yaml --stage data         # cache data bersih + fitur
python -m tip.pipeline --config configs/default.yaml --stage experiments  # E1..E5 semua seed
python -m tip.pipeline --config configs/default.yaml --stage report       # gambar + tabel LaTeX
```

### Alur pipeline

```
data.csv ──► bersihkan (NaN, interpolasi, winsorisasi) ──► fitur (shape | level | missing)
   │
   ├─ split per konsumen (60/20/20, stratifikasi) × 5 seed
   │
   ├─ latih 4 varian detektor ──► E1: AUC, PR-AUC, recall@b  (label nyata, tanpa adaptasi)
   │
   ├─ populasi S1: konsumen jujur-uji, 8,5% dijadikan pencuri (beban asli diketahui)
   │     adversary A1/A2/A3 × grid potongan r  ──► matriks skor S[i, r]
   │
   ├─ E2: pencuri best-response terhadap kebijakan P(eps,tau), tiap anggaran b dan denda phi
   ├─ E3: pencuri non-adaptif (r tetap)  ──► konsistensi
   ├─ E4: akurasi seimbang empiris vs batas mimikri Prop. 1
   └─ E5: cakupan sertifikat, kalibrasi vs deployment (termasuk lintas-kelas)
                │
                └─► results/<nama>/*.csv  ──►  paper/generated/{tab_*.tex, fig_*.pdf}
```

Metrik utama: **NTL lolos** = Σ(1−p_i)·G_i / total beban sistem, dengan G_i kWh yang dicuri.

## 6. Keluaran

| Berkas | Isi |
|--------|-----|
| `results/<nama>/e1_detector.csv` | AUC, PR-AUC, recall/presisi@b per varian dan seed |
| `results/<nama>/e2_game.csv` | semua keseimbangan (mode `adaptive` dan `static`) |
| `results/<nama>/e4_mimicry.csv` | TV (histogram dan KDE), KS, akurasi seimbang naif dan cross-fit (+ galat baku), null permutasi vs `r` |
| `results/<nama>/e5_certificate.csv` | cakupan conformal/Hoeffding per detektor, pasangan kelas (`A2@0.6` = pencuri statis) dan `n_cal`; jarak KS/TV antar pool, batas Hoeffding yang digeser |
| `results/<nama>/d_selection.csv`, `d_hygiene.csv`, `d_zero_load.csv`, `d_deter_feasible.csv` | diagnostik data (`make diag`) |
| `results/<nama>/e0_data.csv` | provenans per seed: ukuran data, jumlah pencuri S1, median beban |
| `results/<nama>/e6_deterrence.csv` | sapuan deterensi (Prop. 3) ke anggaran besar |
| `results/<nama>/config_used.yaml` | config efektif run tersebut |
| `results/<nama>/e7_strategic.csv` | kebijakan sadar-strategi vs top-k/acak/oracle (out-of-sample) |
| `results/<nama>/e7_policy_cells.csv` | probabilitas inspeksi yang dipelajari per sel |
| `results/<nama>/e7_solver_check.csv` | pencarian lokal vs MILP eksak (subsampel) + batas dual |
| `results/<nama>/e9_bounded.csv` | pencuri quantal-response dan serangan transfer |
| `paper/generated/numbers.tex` | makro `\r...` untuk SETIAP angka di teks paper (lihat juga `SUMMARY.md`) |
| `paper/generated/tab_*.tex`, `fig_*.pdf` | di-`\input` otomatis oleh `main.tex` |

## 7. Hasil SGCC (ringkas; angka lengkap di `results/sgcc_main/SUMMARY.md`)

- Detektor bagus pada label nyata (E1), tetapi pencuri SGCC berlabel berlevel **lebih tinggi** daripada konsumen jujur, sehingga detektor membaca "konsumsi tinggi = mencurigakan". Under-reporting justru menurunkan skor; pencuri statis tersimulasi ditemukan lebih jarang daripada inspeksi acak (per kepala), meski energi yang dipulihkan lebih tinggi (inspeksi terkonsentrasi pada konsumen besar).
- Pencuri adaptif (E2) memilih kedalaman di batas grid (`r = 0,9`) dan praktis tak satu pun diinspeksi di bawah top-k. `eps` dan `phi` hampir tak berpengaruh, sesuai Prop. 3 karena `b(1+phi)` jauh di bawah 1.
- Adversarial training hanya membantu terhadap A2 (jejak ramp pada shape); A1 tidak bisa dipelajari (Prop. 1), dan A3 yang ditahan juga tidak.
- E6: deterensi penuh tepat saat `eps*b*(1+phi) >= 1`; top-k murni tidak pernah mendekati itu.
- S2 (pencuri berlevel seperti pencuri berlabel): top-k mengurangi NTL lolos dibanding acak dengan **membatasi** curian konsumen besar, bukan dengan mendeteksinya.
- E5: sertifikat hanya informatif untuk detektor adv-training yang dikalibrasi pada pencuri statis, dan cakupannya runtuh ke 0 begitu pencuri beradaptasi. Keruntuhan itu terukur: jarak Kolmogorov antar pool miss = 0,38; menggeser batas Hoeffding sebesar KS memulihkan cakupan tetapi membuat batas 1,000 (kosong). Pertidaksamaan celah berlaku di 160/160 baris.
- E4 (setelah perbaikan sampel-hingga): TV histogram bias ke atas (rerata 0,018; 0,028 pada r=0,1), sehingga "0 pelanggaran" versi lama terlalu mudah; dengan TV-KDE, estimator naif (maks. atas ambang) melampaui batas di 3/180 setting (semua A3, r=0,1), sedangkan uji satu-sisi pada akurasi seimbang cross-fit di 0/180. Batas pada r=0,1 adalah 0,542 (bukan 0,522-0,528 pada catatan literatur: populasi jujur memuat 5,4% konsumen berbeban nol).
- Seleksi label: laju pencuri berlabel naik dari 3,1% (desil konsumsi terendah) ke 22,2% (tertinggi); menurut fraksi-hilang tidak monoton (puncak 16,2% pada 40-60% hilang; Spearman 0,07). Hasil tidak berubah bila konsumen berbeban nol dibuang (AUC 0,756 menjadi 0,754).
- Perubahan protokol: konsumen berbeban 0 (±5% konsumen jujur) tidak lagi bisa dipilih sebagai pencuri S1; sebelumnya mereka menggelembungkan `deterred`.

### Ekstensi untuk IEEE Trans. Smart Grid (angka di `SUMMARY.md` / `numbers.tex`)

- **E7 kebijakan sadar-strategi** (`stackelberg.py`): p per sel (skor x level terlapor), dilatih pada pencuri injeksi split kalibrasi, dievaluasi pada pencuri uji yang memilih jenis serangan dan r. Di S1 kebijakan joint menurunkan NTL lolos di semua setting (hingga ±17% vs yang terbaik dari top-k/acak), terutama dengan mengurangi curian; sel level saja (tanpa ML) hampir sama baiknya. Kebijakan yang dipelajari menginspeksi pelapor "rendah tapi tidak minimal". MILP eksak tidak selesai bahkan pada 80 pencuri; pencarian lokal lebih baik dari incumbent MILP dan dalam ≤17% dari batas dual.
- **S3 onset** (`configs/sens_onset.yaml`): fitur perubahan saja tidak cukup bila dilatih pada label SGCC; dengan adversarial training pada serangan onset, inversi level hilang, pencuri adaptif mengurangi kedalaman, top-k jauh lebih baik daripada acak, dan kebijakan sadar-strategi menambah perbaikan.
- **Robustness**: CNN Wide & Deep (AUC lebih tinggi, inversi sama, gagal sama), batas r 0.5/0.7 (pencuri memilih batas), quantal response dan serangan transfer (top-k tetap tidak lebih baik daripada acak untuk A1/A3).
- **Aturan interval** (`stackelberg.fit_interval`): 3 parameter (pita level, probabilitas) memulihkan 73-124% dari manfaat kebijakan sel-level penuh di S1; di S2 hanya menolong di satu setting dan lebih buruk daripada top-k di 3 dari 4.
- **UCI** (`configs/uci.yaml`, tanpa label): A2/A4 terdeteksi bila detektor dilatih pada serangan itu; A1/A3 tak terlihat; bila pencuri bebas memilih serangan, tak ada kebijakan yang mengalahkan acak.
- **Catatan lingkungan**: PyTorch hanya untuk CNN; `tip/__init__.py` menyetel `KMP_AFFINITY=disabled` (beberapa sandbox melarang afinitas thread).

## 8. Rencana publikasi (Q1/Q2)

Kuartil berubah tiap tahun. **Verifikasi di Scimago/JCR/Scopus untuk tahun pengajuan** sebelum memutuskan, terutama bila dipakai sebagai syarat kelulusan.

Terverifikasi saat riset: *IEEE Transactions on Smart Grid* berstatus Q1 (impact factor sekitar 9,8–10, tergantung sumber). Impact factor *IJEPES* (≈5,0) dan *Sustainable Energy, Grids and Networks* (≈5,6) tercantum pada indeks pihak ketiga; kuartilnya belum saya verifikasi.

| Tingkat | Jurnal | Syarat agar realistis |
|---------|--------|-----------------------|
| Ambisius | IEEE Trans. Smart Grid; IEEE Trans. Industrial Informatics | Hasil SGCC kuat, seluruh Prop. 1–3 rapi, satu uji tambahan di luar SGCC (mis. profil beban sintetis dari data publik lain) dan analisis sensitivitas penuh |
| Realistis | Int. J. Electrical Power & Energy Systems; Sustainable Energy, Grids and Networks | Kerangka Stackelberg + empiris SGCC + sertifikat; keterbatasan dilaporkan terbuka |
| Aman | IEEE Access; Electric Power Systems Research; Energy and AI | Cukup dengan E1–E5 lengkap dan kode terbuka |

Kuartil IEEE Access, EPSR, Energy and AI, dan TII **belum diverifikasi**.

Satu celah yang menentukan: satu-satunya data berlabel pencuri yang umum dipakai adalah SGCC. Reviewer akan menanyakan generalisasi. Siapkan jawaban sebelum mengirim.

## 9. Keterbatasan dan jebakan (baca sebelum menulis klaim)

1. **Adaptasi hanya disimulasikan.** Tidak ada data publik pencuri adaptif. Skenario S1 menjadikan konsumen jujur sebagai pencuri dengan beban asli diketahui. Klaim harus berbunyi "kokoh terhadap adversary yang dimodelkan".
2. **Pencuri tipikal-level.** S1 menarik pencuri dari populasi jujur sehingga Prop. 1 ketat. Pencuri nyata bisa berlevel tinggi.
3. **Shortcut missingness.** Studi SGCC berbahasa Indonesia melaporkan rasio data hilang sebagai fitur paling diskriminatif (Alfian & Armin, J-ICOM 2026). Hasil utama di sini **tidak** memakainya; varian `gbm_level_miss` mengukur selisihnya.
4. **Tidak ada split temporal.** SGCC tidak mencatat waktu pencurian; split dilakukan per konsumen.
5. **Anggaran dalam ekspektasi.** Inspeksi Bernoulli; anggaran keras butuh dependent rounding.
6. **Label terseleksi.** Retraining dari hasil inspeksi (selective labels) tidak dimodelkan.
7. **Sertifikat bersyarat.** Sah hanya relatif terhadap kelas adversary; E5 mengukur kegagalannya.
8. **Konvergensi.** Best-response asinkron bisa tidak konvergen; kolom `converged` di `e2_game.csv` harus dilaporkan (pada uji asap ±79% konvergen). Pada SGCC: 97% konvergen dengan `br_iters=8` (sisa di adv-training vs A2), 100% dengan 24, perubahan NTL ≤0,23 poin persen (`tab_conv.tex`).

## 10. Ceklis sebelum mengirim paper

- [x] `make data && make run` selesai tanpa galat pada SGCC asli
- [x] Laporkan fraksi `converged` dan sensitivitas terhadap `br_iters`
- [x] Isi semua `\todo{}` hasil di `paper/main.tex` hanya dengan angka dari tabel hasil (via `numbers.tex`; tersisa penulis + URL)
- [x] Ablasi missingness dilaporkan di badan paper
- [ ] Verifikasi ulang entri `refs.bib` terhadap sumber asli. Sudah dicek: entri awal, DOI UCI (halaman repositori), `alfian2026` (halaman jurnal J-ICOM; tidak terindeks CrossRef), penulis Amin dkk. 2015 (Semantic Scholar). Belum: halaman akhir Border dan Sobel 1987 (CrossRef hanya memberi halaman awal); Amin dkk. 2015 dan Lin dkk. 2014 dikarakterisasi dari abstrak saja (teks penuh tidak tersedia), jadi baca sebelum mengirim; kuartil jurnal target
- [x] Batas halaman TSG (IEEE PES): pengiriman pertama maksimal 10 halaman termasuk referensi, jika lebih dikembalikan tanpa review; biaya halaman berlebih berlaku di atas 12 halaman terbit ($250 per halaman untuk pengiriman 2024 ke atas). Paper 10 halaman (sisa ~0,3 halaman); tabel/gambar pendukung ada di `paper/supplement.tex`
- [ ] Isi nama penulis dan afiliasi (blok `\todo`), tambahkan URL repositori dan lisensi; kedua blok menambah sekitar 0,1 halaman
- [x] Hapus `paper/generated/*` hasil uji asap sebelum menjalankan `make run`

## 11. Sitasi data

Z. Zheng, Y. Yang, X. Niu, H.-N. Dai, Y. Zhou, "Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids," *IEEE Trans. Industrial Informatics*, vol. 14, no. 4, pp. 1606–1615, 2018.
