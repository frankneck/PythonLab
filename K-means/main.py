# -*- coding: utf-8 -*-
# ЛР 3.8 — Кластеризация: K-means и EM (Gaussian Mixture)
# Читает quake.dat (ARFF/CSV-подобный), строит квадратные графики,
# выбирает лучшее k по минимуму f(k)=avg_intra/avg_inter, сохраняет PNG/CSV.
#
# Выходные файлы (папка output/):
#   Гистограммы: hist_Focal_depth.png, hist_Latitude.png, hist_Longitude.png, hist_Richter.png
#   Сырые точки: raw_scatter.png
#   ---- K-means:
#   kmeans_k_scan.csv, inertia_vs_k.png, f_vs_k.png
#   kmeans_bestk_summary.csv, kmeans_bestk_scatter.png
#   ---- EM (GMM):
#   gmm_k_scan.csv, gmm_f_vs_k.png
#   gmm_bestk_summary.csv, gmm_bestk_scatter.png

import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances

# ---------- настройки отрисовки ----------
FIGSIZE = (8, 8)  # все картинки квадратные и крупные
PALETTE = np.array([
    "tab:blue","tab:orange","tab:green","tab:red","tab:purple",
    "tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"
])

# ---------- загрузка данных ----------
def load_quake(path="quake.dat"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}. Положите quake.dat рядом с main.py")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    data_started = False
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("@data"):
            data_started = True
            continue
        if line.startswith("@") or (not data_started):
            continue
        rows.append(line)
    if not rows:
        raise ValueError("В quake.dat не найден блок @data.")
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None)
    if df.shape[1] < 4:
        raise ValueError("Ожидались 4 признака в каждой строке.")
    df.columns = ["Focal_depth", "Latitude", "Longitude", "Richter"]
    return df

# ---------- метрика f(k) ----------
def avg_intra_inter(X, labels):
    D = pairwise_distances(X, X, metric="euclidean")
    uniq = np.unique(labels)
    intra_vals, inter_vals = [], []
    for c in uniq:
        idx = np.where(labels == c)[0]
        idx_out = np.where(labels != c)[0]
        # внутрикластерное среднее попарное расстояние (без диагонали)
        if len(idx) > 1:
            d_in = D[np.ix_(idx, idx)]
            m = len(idx)
            intra_vals.append(d_in.sum()/(m*(m-1)))
        else:
            intra_vals.append(0.0)
        # среднее расстояние до остальных кластеров
        if len(idx) > 0 and len(idx_out) > 0:
            d_out = D[np.ix_(idx, idx_out)]
            inter_vals.append(d_out.mean())
    avg_intra = float(np.mean(intra_vals)) if intra_vals else float("nan")
    avg_inter = float(np.mean(inter_vals)) if inter_vals else float("nan")
    f_ratio = (avg_intra/avg_inter) if (avg_inter and avg_inter > 0) else float("nan")
    return avg_intra, avg_inter, f_ratio

def choose_best_k_by_f(scan_df):
    # k >= 2, минимальный f; при равенстве — меньшая inertia (если есть), затем меньшее k
    cand = scan_df[scan_df["k"] >= 2].copy()
    if "inertia" in cand.columns:
        cand = cand.sort_values(by=["f_ratio", "inertia", "k"], ascending=[True, True, True])
    else:
        cand = cand.sort_values(by=["f_ratio", "k"], ascending=[True, True])
    return int(cand.iloc[0]["k"])

# ---------- графики ----------
def plot_histograms(df, outdir):
    for col in df.columns:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.hist(df[col].values, bins=30, edgecolor="black")
        ax.set_title(f"Гистограмма: {col}")
        ax.set_xlabel(col); ax.set_ylabel("Частота")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"hist_{col}.png"))
        plt.close(fig)

def plot_raw_scatter(X, outpath):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(X[:,0], X[:,1], s=8)
    ax.set_title("Данные: широта–долгота (raw)")
    ax.set_xlabel("Latitude"); ax.set_ylabel("Longitude")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_line_square(xs, ys, title, xlabel, ylabel, outpath):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(xs, ys, marker="o")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_clusters_right_legend(X, labels, centers, outpath, title, label_prefix, center_label):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    # резервируем правое поле под легенду (ось не сжимается)
    fig.subplots_adjust(right=0.80)

    uniq = np.unique(labels)
    handles = []
    for lab in uniq:
        mask = labels == lab
        color = PALETTE[lab % len(PALETTE)]
        sc = ax.scatter(X[mask,0], X[mask,1], s=12, c=color, label=f"{label_prefix} {lab+1}")
        handles.append(sc)

    # центры/средние
    ctr = ax.scatter(centers[:,0], centers[:,1], s=180, marker="X",
                     edgecolor="black", linewidths=1.2, c="none", label=center_label)
    handles.append(ctr)

    ax.set_title(title)
    ax.set_xlabel("Latitude"); ax.set_ylabel("Longitude")
    ax.set_aspect("equal", adjustable="box")

    fig.legend(handles=handles,
               labels=[h.get_label() for h in handles],
               loc="center left", bbox_to_anchor=(0.82, 0.5), frameon=True)
    fig.savefig(outpath)  # без bbox_inches="tight" — геометрия осей не меняется
    plt.close(fig)

# ---------- основной сценарий ----------
def main():
    outdir = "output"
    os.makedirs(outdir, exist_ok=True)

    df = load_quake("quake.dat")
    X = df[["Latitude", "Longitude"]].to_numpy()

    # 1) распределения и исходная карта
    plot_histograms(df, outdir)
    plot_raw_scatter(X, os.path.join(outdir, "raw_scatter.png"))

    # 2) K-means: сканирование k=1..10
    rows_km = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, init="random", n_init=10, max_iter=10000, random_state=42)
        labels = km.fit_predict(X)
        inertia = float(km.inertia_)
        ai, ae, fr = avg_intra_inter(X, labels)
        rows_km.append({"k": k, "inertia": inertia, "avg_intra": ai, "avg_inter": ae, "f_ratio": fr})
    scan_km = pd.DataFrame(rows_km)
    scan_km.to_csv(os.path.join(outdir, "kmeans_k_scan.csv"), index=False)

    # графики выбора k
    plot_line_square(scan_km["k"], scan_km["inertia"],
                     "K-means: inertia vs k (1..10)", "k", "Inertia",
                     os.path.join(outdir, "inertia_vs_k.png"))
    plot_line_square(scan_km["k"], scan_km["f_ratio"],
                     "K-means: f(k)=intra/inter vs k (1..10)", "k", "f(k)",
                     os.path.join(outdir, "f_vs_k.png"))

    # выбор лучшего k по f(k) (тай-брейк inertia, затем k)
    best_k_km = choose_best_k_by_f(scan_km)
    print(f"[K-means] Лучшее k по f(k): {best_k_km}")

    # финальная кластеризация K-means
    km_best = KMeans(n_clusters=best_k_km, init="random", n_init=10, max_iter=10000, random_state=42)
    labels_best = km_best.fit_predict(X)
    centers_km = km_best.cluster_centers_

    # сводка
    sizes = pd.Series(labels_best).value_counts().sort_index()
    summary_km = pd.concat([
        sizes.rename("size"),
        pd.DataFrame(centers_km, columns=["Latitude_center", "Longitude_center"])
    ], axis=1)
    summary_km.index = summary_km.index + 1
    summary_km.to_csv(os.path.join(outdir, "kmeans_bestk_summary.csv"))

    # картинка K-means
    plot_clusters_right_legend(
        X, labels_best, centers_km,
        os.path.join(outdir, "kmeans_bestk_scatter.png"),
        title=f"K-means: кластеры при k={best_k_km}",
        label_prefix="Кластер",
        center_label="Центры"
    )

    # 3) EM (GMM): сканирование k=1..10 по f(k)
    rows_em = []
    for k in range(1, 11):
        gmm = GaussianMixture(n_components=k, covariance_type="full", max_iter=1000,
                              random_state=42, n_init=5)
        gmm.fit(X)
        labels_em = gmm.predict(X)  # жёсткие метки для расчёта f(k)
        ai, ae, fr = avg_intra_inter(X, labels_em)
        rows_em.append({"k": k, "avg_intra": ai, "avg_inter": ae, "f_ratio": fr})
    scan_em = pd.DataFrame(rows_em)
    scan_em.to_csv(os.path.join(outdir, "gmm_k_scan.csv"), index=False)

    # график f(k) для EM
    plot_line_square(scan_em["k"], scan_em["f_ratio"],
                     "EM (GMM): f(k)=intra/inter vs k (1..10)", "k", "f(k)",
                     os.path.join(outdir, "gmm_f_vs_k.png"))

    # выбор лучшего k для EM
    best_k_em = choose_best_k_by_f(scan_em)
    print(f"[EM] Лучшее k по f(k): {best_k_em}")

    # финальная EM-модель
    gmm_best = GaussianMixture(n_components=best_k_em, covariance_type="full", max_iter=1000,
                               random_state=42, n_init=5)
    gmm_best.fit(X)
    labels_em_best = gmm_best.predict(X)
    means_em = gmm_best.means_

    # сводка по EM
    sizes_em = pd.Series(labels_em_best).value_counts().sort_index()
    summary_em = pd.concat([
        sizes_em.rename("size"),
        pd.DataFrame(means_em, columns=["Latitude_mean", "Longitude_mean"])
    ], axis=1)
    summary_em.index = summary_em.index + 1
    summary_em.to_csv(os.path.join(outdir, "gmm_bestk_summary.csv"))

    # картинка EM
    plot_clusters_right_legend(
        X, labels_em_best, means_em,
        os.path.join(outdir, "gmm_bestk_scatter.png"),
        title=f"EM (GMM): компоненты при k={best_k_em}",
        label_prefix="Компонента",
        center_label="Средние (μ)"
    )

if __name__ == "__main__":
    main()
