"""Bước 1: tự viết K-Means bằng numpy, chạy trên bảy điểm của slide.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

from pathlib import Path

import matplotlib.pyplot as plt


import numpy as np
from sklearn.cluster import KMeans

plt.switch_backend("Agg")


def save_figure(fig, name):
    figure_dir = Path(__file__).resolve().parents[1] / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


P = np.array([[1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0],
                [3.5, 5.0], [4.5, 5.0], [3.5, 4.5]])
TEN = ["A", "B", "C", "D", "E", "F", "G"]


lich_su = []


def k_means(X, tam, so_vong_toi_da=10):
    """K-Means don gian: gan diem vao tam gan nhat roi doi tam."""
    for vong in range(1, so_vong_toi_da + 1):
        # Bước gán: khoảng cách từ mỗi điểm tới mỗi tâm
        kc = np.sqrt(((X[:, None, :] - tam[None, :, :]) ** 2).sum(axis=2))
        nhan = kc.argmin(axis=1)
        lich_su.append((tam.copy(), nhan.copy()))
        print(f"Vong {vong}: tam = {np.round(tam, 2).tolist()}")
        for k in range(len(tam)):
            ds = [TEN[i] for i in range(len(X)) if nhan[i] == k]
            print(f"  cum {k + 1}: {', '.join(ds)}")
        # Bước dời tâm: trung bình các điểm trong mỗi cụm
        tam_moi = np.array([X[nhan == k].mean(axis=0)
                                    for k in range(len(tam))])
        if np.allclose(tam_moi, tam):
            print("  Tam khong doi nua, dung.")
            return tam, nhan
        tam = tam_moi
    return tam, nhan


# Khởi tạo cố ý tệ: hai tâm là A và B, nằm sát nhau
tam, nhan = k_means(P, P[[0, 1]].copy())
print()

# Tính inertia: tổng bình phương khoảng cách tới tâm cụm của mình
inertia = ((P - tam[nhan]) ** 2).sum()
print(f"Inertia tu tinh: {inertia:.4f}")

# So với scikit-learn, cho cùng tâm ban đầu
km = KMeans(n_clusters=2, init=P[[0, 1]], n_init=1).fit(P)
print(f"Inertia cua scikit-learn: {km.inertia_:.4f}")
print("Tam cua scikit-learn:", np.round(km.cluster_centers_, 2).tolist())


fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharex=True, sharey=True)
for i, (ax, (tam_vong, nhan_vong)) in enumerate(zip(axes, lich_su), 1):
    ax.scatter(*P.T, c=nhan_vong, cmap="coolwarm")
    ax.scatter(*tam_vong.T, marker="X", color="black", s=80)
    for ten, (a, b) in zip(TEN, P):
        ax.annotate(ten, (a, b), xytext=(5, 4), textcoords="offset points")
    ax.set(title=f"Vòng {i}", xlabel="x1")
axes[0].set_ylabel("x2")
save_figure(fig, "hinh_1_cac_vong_kmeans")
