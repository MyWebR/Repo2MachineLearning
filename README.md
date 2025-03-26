# Proyek Analisis Statistik Tim Sepakbola

## Ringkasan
Proyek ini menyediakan analisis komprehensif statistik tim sepakbola untuk musim 2021-2022, menunjukkan teknik pemrosesan data, visualisasi, dan machine learning menggunakan Python.

## Hasil code dari minggu 3 dan 4 ada di dalam folder output

## Struktur Proyek

### 1. `minggu3.py` - Eksplorasi Data Tim Sepakbola
Skrip ini berfokus pada analisis data performa tim sepakbola dengan fitur utama:
- Deteksi encoding file CSV secara otomatis
- Eksplorasi dan praproses data dasar
- Analisis statistik metrik performa tim
- Visualisasi distribusi gol dan performa tim
- Menghasilkan file keluaran termasuk:
  - Analisis teks detail
  - Histogram distribusi gol
  - Scatter plot poin vs selisih gol

#### Analisis Utama:
- Total gol yang dicetak
- Rata-rata gol per tim
- Tim terbaik dalam berbagai kategori
- Perhitungan selisih gol
- Peringkat liga berdasarkan poin

### 2. `minggu4.py` - Teknik Pemrosesan Data Lanjutan
Skrip ini mendemonstrasikan beberapa teknik pemrosesan data dan machine learning lanjutan:

#### Eksperimen yang Diimplementasikan:
1. **Penanganan Nilai Hilang**
   - Teknik imputasi manual mean, median, dan modus
   - Menunjukkan strategi berbeda untuk menangani data tidak lengkap

2. **Penskalaan Fitur**
   - Implementasi manual standarisasi (normalisasi Z-skor)
   - Normalisasi min-maks manual
   - Menunjukkan cara normalisasi fitur untuk machine learning

3. **Evaluasi Klasifikasi Biner**
   - Perhitungan matriks konfusi manual
   - Komputasi metrik kunci:
     - Akurasi
     - Presisi
     - Recall
     - Skor F1

4. **Penanganan Data Tidak Seimbang**
   - Teknik undersampling manual
   - Oversampling manual dengan duplikasi data
   - Menunjukkan metode untuk menyeimbangkan distribusi kelas

5. **Validasi Silang**
   - Implementasi validasi silang k-fold manual
   - Model regresi logistik sederhana
   - Menghitung dan merata-ratakan performa model di berbagai pembagian data

## Persyaratan
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Chardet

## Penggunaan
1. Pastikan file `2021_2022_Football_Team_Stats.csv` berada di direktori yang sama
2. Jalankan skrip menggunakan Python
3. Periksa direktori `output` untuk file analisis dan visualisasi yang dihasilkan

## Tujuan Pembelajaran
- Teknik praproses data
- Analisis statistik
- Evaluasi model machine learning
- Visualisasi data olahraga

## Lisensi
Proyek sumber terbuka untuk tujuan pendidikan
