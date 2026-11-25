"""Bước 5: nén màu một bức ảnh bằng K-Means."""

from pathlib import Path

import matplotlib.pyplot as plt


import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

plt.switch_backend("Agg")


def save_figure(fig, name):
    figure_dir = Path(__file__).resolve().parents[1] / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


anh = load_sample_image("china.jpg")                     # ảnh có sẵn trong scikit-learn
cao, rong, _ = anh.shape
diem = anh.reshape(-1, 3) / 255.0                # mỗi dòng là một điểm ảnh RGB
so_mau = len(np.unique(anh.reshape(-1, 3), axis=0))
print(f"Anh {cao} x {rong}, co {cao * rong} diem anh, {so_mau} mau")
print()

# Học trên 5000 điểm ảnh rút ngẫu nhiên cho nhanh, rồi gán cho mọi điểm
rng = np.random.default_rng(0)
mau = diem[rng.choice(len(diem), 5000, replace=False)]

print("  K   so bit moi diem   nho hon   sai lech trung binh")
anh_nen = {}
for k in [4, 16, 64]:
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(mau)
    nhan = km.predict(diem)
    moi = km.cluster_centers_[nhan]
    anh_nen[k] = moi.reshape(anh.shape)
    bit = int(np.ceil(np.log2(k)))
    sai = np.abs(moi - diem).mean() * 255
    print(f"  {k:2d}        {bit}            {24 / bit:4.1f} lan"
            f"      {sai:5.2f} muc sang")


fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, img, ten in zip(axes, [anh, anh_nen[16], anh_nen[4]], ["Ảnh gốc", "16 màu", "4 màu"]):
    ax.imshow(img)
    ax.set_title(ten)
    ax.axis("off")
save_figure(fig, "hinh_8_nen_mau_anh")
