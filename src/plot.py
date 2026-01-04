import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ---------------------------------
# 1) Chemin vers timings.csv
# ---------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)                  
CSV_PATH   = os.path.join(ROOT_DIR, "timings.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Fichier introuvable : {CSV_PATH}")

print("CSV utilisé :", CSV_PATH)

# ---------------------------------
# 2) Lecture du CSV
# ---------------------------------
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
df["method"] = df["method"].astype(str).str.strip()
df["n"] = df["n"].astype(int)
df["time_s"] = df["time_s"].astype(float)

# Conversion en microsecondes
df["time_us"] = df["time_s"] * 1e6

# Tailles de matrices (ticks propres)
n_values = sorted(df["n"].unique())
x_pos = list(range(len(n_values)))  # positions discrètes 0..N-1 (pour éviter chevauchement)

# Style des méthodes
styles = {
    "TRIDIAG_LU_FACTOR": dict(marker="s", linewidth=1.8),
    "LAPACKE_DGBTRF_DGBTRS": dict(marker="o", linewidth=1.8),
}

# Ordre stable 
preferred_order = ["LAPACKE_DGBTRF_DGBTRS", "TRIDIAG_LU_FACTOR"]
method_list = [m for m in preferred_order if m in df["method"].unique()] + \
              [m for m in df["method"].unique() if m not in preferred_order]

def x_from_n(series_n):
    """Convertit une liste/serie de n en positions discrètes x=0..N-1 selon n_values."""
    idx = {n: i for i, n in enumerate(n_values)}
    return [idx[int(v)] for v in series_n]

# ---------------------------------
# 3) Figures séparées 
# ---------------------------------
for method in method_list:
    sub = df[df["method"] == method].sort_values("n")
    x = x_from_n(sub["n"])

    plt.figure(figsize=(9, 5))
    plt.plot(x, sub["time_us"], label=method, **styles.get(method, {}))

    plt.xlabel("Taille de matrice n")
    plt.ylabel("Temps (µs)")
    plt.title(f"Performance - {method} (échelle linéaire)")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.xticks(x_pos, [str(n) for n in n_values], rotation=30)
    plt.tight_layout()

    plt.savefig(os.path.join(ROOT_DIR, f"{method}.png"), dpi=200)
    plt.close()

# ---------------------------------
# 4) Figure combinée 
# ---------------------------------
plt.figure(figsize=(10, 5))

for method in method_list:
    sub = df[df["method"] == method].sort_values("n")
    x = x_from_n(sub["n"])
    plt.plot(x, sub["time_us"], label=method, **styles.get(method, {}))

plt.xlabel("Taille de matrice n")
plt.ylabel("Temps (µs)")
plt.title("Comparaison des méthodes – Poisson 1D (échelle linéaire)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.xticks(x_pos, [str(n) for n in n_values], rotation=30)
plt.tight_layout()

plt.savefig(os.path.join(ROOT_DIR, "comparison.png"), dpi=200)
plt.close()

# ---------------------------------
# 5) Figure log-log (ANALYSE COMPLEXITÉ : log-log)
# ---------------------------------
plt.figure(figsize=(10, 5))

for method in method_list:
    sub = df[df["method"] == method].sort_values("n")
    plt.plot(sub["n"], sub["time_us"], label=method, **styles.get(method, {}))

plt.xscale("log")
plt.yscale("log")
plt.xlabel("n (taille matrice)")
plt.ylabel("Temps (µs)")
plt.title("Analyse de complexité (log-log)")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend()

plt.xticks(n_values, [str(n) for n in n_values], rotation=30)
plt.tight_layout()

plt.savefig(os.path.join(ROOT_DIR, "comparison_loglog.png"), dpi=200)
plt.close()

print("OK – figures générées dans le dossier principal :")
print(" - comparison.png")
print(" - comparison_loglog.png")
for m in method_list:
    print(f" - {m}.png")
