"""Bước 4: ba cái bẫy của K-Means."""

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
X = df[["thu_nhap", "diem_chi_tieu"]]
Xs = StandardScaler().fit_transform(X)

# Bẫy 1: thu nhập ghi bằng nghìn đồng, không chuẩn hoá
X_nghin = X.copy()
X_nghin["thu_nhap"] = X_nghin["thu_nhap"] * 1000
km_tho = KMeans(n_clusters=5, n_init=10, random_state=0).fit(X_nghin)
tam_tho = km_tho.cluster_centers_.copy()
tam_tho[:, 0] = tam_tho[:, 0] / 1000             # đổi lại triệu đồng để dễ đọc
print("Bay 1: khong chuan hoa, tam cum (thu nhap trieu, chi tieu):")
for t in tam_tho[np.argsort(tam_tho[:, 0])]:
    print(f"  ({t[0]:5.1f}, {t[1]:5.1f})")
print()

# Bẫy 2: mỗi lần chỉ khởi tạo ngẫu nhiên một lần
ket = []
km_te = None
for rs in range(20):
    km = KMeans(n_clusters=5, init="random", n_init=1,
                random_state=rs).fit(Xs)
    if km_te is None or km.inertia_ > km_te.inertia_:
        km_te = km
    ket.append(round(km.inertia_, 1))
print("Bay 2: inertia cua 20 lan khoi tao ngau nhien:")
ket = sorted(ket)
print(" ", ket[:10])
print(" ", ket[10:])
tot = KMeans(n_clusters=5, n_init=10, random_state=0).fit(Xs)
print(f"  Voi n_init=10: inertia {tot.inertia_:.1f}")
print()

# Bẫy 3: cụm không tròn, dữ liệu hai vầng trăng
tr = pd.read_csv(DATA_DIR / "hai_vang_trang.csv")
km_tr = KMeans(n_clusters=2, n_init=10, random_state=0)
nhan = km_tr.fit_predict(tr[["x1", "x2"]])
khop = (nhan == tr["nhan"]).mean()
# Số thứ tự cụm có thể ngược với nhãn thật, nên lấy cách khớp tốt hơn
print(f"Bay 3: hai vang trang, ty le khop nhan that "
        f"{max(khop, 1 - khop):.4f}")


def ve_cum(ax, diem, nhan, ten, tam=None):
    ax.scatter(*diem.to_numpy().T, c=nhan, cmap="tab10", s=18)
    if tam is not None:
        ax.scatter(*tam.T, marker="X", s=85, c="black")
    ax.set(title=ten, xlabel=diem.columns[0], ylabel=diem.columns[1])


sc = StandardScaler().fit(X)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
ve_cum(axes[0], X, km_tho.labels_, "Không chuẩn hoá", tam_tho)
ve_cum(axes[1], X, tot.labels_, "Có chuẩn hoá", sc.inverse_transform(tot.cluster_centers_))
save_figure(fig, "hinh_5_thang_do")

fig, ax = plt.subplots()
ve_cum(ax, X, km_te.labels_, "Lần khởi tạo có inertia cao nhất",
       sc.inverse_transform(km_te.cluster_centers_))
save_figure(fig, "hinh_6_khoi_tao_te")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
ve_cum(axes[0], tr[["x1", "x2"]], tr["nhan"], "Nhãn thật")
ve_cum(axes[1], tr[["x1", "x2"]], nhan, "Cụm K-Means", km_tr.cluster_centers_)
save_figure(fig, "hinh_7_hai_vang_trang")
