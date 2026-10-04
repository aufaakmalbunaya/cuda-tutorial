LAPORAN PRAKTIKUM TUTORIAL GPU (CUDA C++) — PAKET OVERLEAF
======================================================
Nama: Aufa Akmal Bunaya (23/515767/PA/22027)
Tanggal: 4 Oktober 2026

ISI PAKET
---------
laporan_cuda.tex   : sumber LaTeX laporan (bahasa Indonesia)
figures/           : 3 plot dari DATA PENGUKURAN ASLI Tesla T4
                     (vector_speedup.png, matmul_naive_tiled.png,
                      gridstride_bw.png) — dirender matplotlib dari
                     77 pengukuran mentah (evidence/*.csv di repo GitHub)

CARA KOMPILASI DI OVERLEAF
--------------------------
1. Upload laporan_cuda.tex sebagai main file.
2. Upload folder figures/ beserta isinya (pertahankan struktur folder).
3. Pilih compiler pdfLaTeX, lalu Recompile (2x untuk TOC final).

CATATAN
-------
- File .tex memakai paket standar: babel (indonesian via \babelprovide),
  lmodern, microtype, graphicx, xcolor, booktabs, tabularx, float,
  caption, enumitem, fancyhdr, titlesec, hyperref, listings, seqsplit, soul.
- Seluruh angka pada laporan adalah hasil pengukuran nyata pada
  Tesla T4 (Kaggle, CUDA 12.8), diringkas sebagai median beberapa run.
- Kode sumber CUDA, notebook Kaggle tereksekusi, dan data mentah:
  https://github.com/aufaakmalbunaya/cuda-tutorial
