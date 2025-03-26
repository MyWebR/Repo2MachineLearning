import numpy as np
import matplotlib.pyplot as plt
import chardet
import csv
import pandas as pd
import sys
import os

# Redirect output to both console and file
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Buat direktori output untuk minggu ke-3 jika belum ada
output_dir = 'output/minggu_ke_3'
os.makedirs(output_dir, exist_ok=True)

# Buka file output
output_file_path = os.path.join(output_dir, 'analisis_tim_sepakbola.txt')
output_file = open(output_file_path, 'w', encoding='utf-8')

# Redirect output
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, output_file)

# Mendeteksi encoding file CSV secara otomatis
file_path = "2021_2022_Football_Team_Stats.csv"
with open(file_path, "rb") as f:
    detected_encoding = chardet.detect(f.read())['encoding']
    print("Encoding yang terdeteksi:", detected_encoding)

# Membaca data menggunakan pandas (lebih reliable)
try:
    # Gunakan titik koma sebagai delimiter
    df = pd.read_csv(file_path, sep=';', encoding=detected_encoding)
    
    # Cetak informasi dasar dataset
    print("\nInformasi Dataset:")
    print(df.info())

    # Menampilkan 5 tim pertama
    print("\n5 Tim pertama dalam dataset:")
    for i in range(min(5, len(df))):
        print(f"{df['Squad'].iloc[i]} ({df['Country'].iloc[i]}) - Peringkat: {df['LgRk'].iloc[i]}")
    
    # Konversi ke numpy array untuk analisis lanjutan
    data = df.to_records(index=False)

except FileNotFoundError:
    print("File tidak ditemukan, menggunakan data contoh...")
    # Data contoh jika file tidak ditemukan
    squads = ['Manchester City', 'Liverpool', 'Chelsea', 'Tottenham', 'Arsenal']
    countries = ['England', 'England', 'England', 'England', 'England']
    league_ranks = [1, 2, 3, 4, 5]
    matches_played = [38, 38, 38, 38, 38]
    wins = [29, 28, 21, 22, 21]
    draws = [6, 8, 11, 5, 3]
    losses = [3, 2, 6, 11, 14]
    goals_for = [99, 94, 76, 69, 61]
    goals_against = [26, 26, 33, 40, 48]
    points = [93, 92, 74, 71, 69]

    # Membuat structured array
    data = np.array(list(zip(squads, countries, league_ranks, matches_played, wins, draws, losses, goals_for, goals_against, points)), 
                    dtype=[('Squad', 'U30'), ('Country', 'U20'), ('LgRk', 'i4'), ('MP', 'i4'), ('W', 'i4'), 
                           ('D', 'i4'), ('L', 'i4'), ('GF', 'i4'), ('GA', 'i4'), ('Pts', 'i4')])

# Analisis Dasar
print("\nAnalisis Statistik Liga:")
print(f"Total gol dicetak: {np.sum(data['GF'])}")
print(f"Rata-rata gol per tim: {np.mean(data['GF']):.1f}")
print(f"Tim dengan gol terbanyak: {data['Squad'][np.argmax(data['GF'])]} ({np.max(data['GF'])} gol)")
print(f"Tim dengan pertahanan terbaik: {data['Squad'][np.argmin(data['GA'])]} ({np.min(data['GA'])} kebobolan)")
print(f"Tim dengan poin tertinggi: {data['Squad'][np.argmax(data['Pts'])]} ({np.max(data['Pts'])} poin)")

# Menghitung selisih gol (GD)
goal_difference = data['GF'] - data['GA']

# Menampilkan 5 tim dengan selisih gol terbaik
sorted_indices = np.argsort(goal_difference)[::-1]  # Urutkan dari terbesar ke terkecil
print("\n5 Tim dengan Selisih Gol Terbaik:")
for rank, idx in enumerate(sorted_indices[:5], 1):
    print(f"{rank}. {data['Squad'][idx]}: {goal_difference[idx]}")

# Visualisasi Data
plt.figure(figsize=(12, 5))

# Histogram Jumlah Gol
plt.subplot(1, 2, 1)
plt.hist(data['GF'], bins=7, color='blue', edgecolor='black')
plt.title('Distribusi Jumlah Gol Tim')
plt.xlabel('Jumlah Gol')
plt.ylabel('Jumlah Tim')

# Simpan plot
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'distribusi_gol.png'))
plt.close()

# Scatter Plot Poin vs Selisih Gol
plt.figure(figsize=(8, 6))
plt.scatter(goal_difference, data['Pts'], color='red')
plt.title('Hubungan Selisih Gol dan Poin')
plt.xlabel('Selisih Gol')
plt.ylabel('Poin')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'hubungan_gol_poin.png'))
plt.close()

# Peringkat Liga berdasarkan Poin
sorted_indices = np.argsort(data['Pts'])[::-1]  # Urutkan dari tertinggi ke terendah
print("\nPeringkat Liga Berdasarkan Poin:")
for rank, idx in enumerate(sorted_indices, 1):
    print(f"{rank}. {data['Squad'][idx]} ({data['Pts'][idx]} Poin)")

# Kembalikan stdout ke semula
sys.stdout = original_stdout
output_file.close()

print(f"\nHasil analisis telah disimpan di {output_file_path}")
print("Visualisasi telah disimpan di direktori 'output/minggu_ke_3':")
print("- distribusi_gol.png")
print("- hubungan_gol_poin.png")wdadcawewaedawd