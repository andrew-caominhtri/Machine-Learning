"""Bước 2: phân khúc khách hàng bằng K-Means."""

import matplotlib.pyplot as plt


from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
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
print("Kich thuoc bang:", df.shape)
print(df.describe().round(1).to_string())
print()

X = df[["thu_nhap", "diem_chi_tieu"]]
sc = StandardScaler().fit(X)
Xs = sc.transform(X)

km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(Xs)
df["cum"] = km.labels_

# Tâm cụm đang ở thang đã chuẩn hoá, đổi ngược về đơn vị gốc để đọc
tam = pd.DataFrame(sc.inverse_transform(km.cluster_centers_),
                    columns=["thu_nhap", "diem_chi_tieu"])
tam["so_khach"] = np.bincount(km.labels_)
tam["tuoi_tb"] = df.groupby("cum")["tuoi"].mean()
print("Tam cum, don vi goc:")
print(tam.round(1).to_string())
print()
print(f"Inertia: {km.inertia_:.2f}")
print()

# Khách mới: chuẩn hoá bằng đúng bộ chuẩn hoá cũ rồi tìm tâm gần nhất
moi = pd.DataFrame({"thu_nhap": [29.0, 9.0],
                        "diem_chi_tieu": [85, 75]})
cum_moi = km.predict(sc.transform(moi))
for i in range(len(moi)):
    print(f"Khach moi {i + 1}: thu nhap {moi.iloc[i, 0]},"
            f" chi tieu {moi.iloc[i, 1]} -> cum {cum_moi[i]}")


fig, ax = plt.subplots()
ax.scatter(*X.to_numpy().T, alpha=0.7)
ax.set(xlabel="Thu nhập (triệu đồng)", ylabel="Điểm chi tiêu", title="Dữ liệu khách hàng")
save_figure(fig, "hinh_2_du_lieu_khach_hang")

fig, ax = plt.subplots()
for i, goc in enumerate(tam["thu_nhap"].sort_values().index, 1):
    ax.scatter(*X.loc[km.labels_ == goc].to_numpy().T, label=f"Cụm {i}")
ax.scatter(tam["thu_nhap"], tam["diem_chi_tieu"], marker="X", c="black", s=90)
ax.set(xlabel="Thu nhập (triệu đồng)", ylabel="Điểm chi tiêu", title="Năm cụm khách hàng")
ax.legend()
save_figure(fig, "hinh_3_nam_cum_khach_hang")
