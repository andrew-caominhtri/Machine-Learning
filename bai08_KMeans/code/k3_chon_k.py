"""Bước 3: chọn số cụm K bằng khuỷu tay và chỉ số silhouette."""

import matplotlib.pyplot as plt


from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

plt.switch_backend("Agg")


def save_figure(fig, name):
    figure_dir = Path(__file__).resolve().parents[1] / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


df = pd.read_csv(DATA_DIR / "khach_hang.csv")
Xs = StandardScaler().fit_transform(df[["thu_nhap", "diem_chi_tieu"]])

print("  K    inertia   giam so voi K-1   silhouette")
inertia = []
silhouettes = []
truoc = None
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
    giam = "" if truoc is None else f"{truoc - km.inertia_:8.1f}"
    # Silhouette cần ít nhất hai cụm
    sil = "" if k == 1 else f"{silhouette_score(Xs, km.labels_):.3f}"
    print(f"  {k:2d}   {km.inertia_:7.1f}   {giam:>15s}   {sil:>10s}")
    inertia.append(km.inertia_)
    if k > 1:
        silhouettes.append(silhouette_score(Xs, km.labels_))
    truoc = km.inertia_


fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, ks, diem, ten in [(axes[0], range(1, 11), inertia, "Inertia"),
                           (axes[1], range(2, 11), silhouettes, "Silhouette")]:
    ax.plot(ks, diem, "o-")
    ax.scatter(5, diem[5 - ks.start], s=130, facecolors="none", edgecolors="red")
    ax.set(xlabel="Số cụm K", ylabel=ten, xticks=list(ks))
save_figure(fig, "hinh_4_chon_k")
