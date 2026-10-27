# -*- coding: utf-8 -*-
"""Bước 4: đặt KNN cạnh hai mô hình đã học trên dữ liệu có biên cong."""

import pandas as pd
from pathlib import Path

# Đường dẫn theo vị trí file, chạy được từ thư mục gốc hoặc code/.
THU_MUC_GOC = Path(__file__).resolve().parent.parent
THU_MUC_HINH = THU_MUC_GOC / "figures"
THU_MUC_HINH.mkdir(parents=True, exist_ok=True)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv(THU_MUC_GOC / "data" / "hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

cac_mo_hinh = [
    ("Hoi quy logistic", LogisticRegression()),
    ("Naive Bayes     ", GaussianNB()),
    ("KNN, k = 13     ", KNeighborsClassifier(n_neighbors=13)),
]

print("Accuracy tren tap kiem tra, bo hai vang trang")
for ten, mh in cac_mo_hinh:
    mh.fit(X_tr, y_tr)
    print(f" {ten}: {accuracy_score(y_te, mh.predict(X_te)):.4f}")

# ==================== VẼ HÌNH THEO LAB04 ====================
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

# Hình 5: PDF chỉ minh hoạ biên của LogisticRegression và KNN.

x1, x2 = np.meshgrid(np.linspace(X.x1.min() - 0.3, X.x1.max() + 0.3, 220),
                     np.linspace(X.x2.min() - 0.3, X.x2.max() + 0.3, 180))
luoi = pd.DataFrame({"x1": x1.ravel(), "x2": x2.ravel()})

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
for ax, (ten, model) in zip(axes, [cac_mo_hinh[0], cac_mo_hinh[2]]):

    du_doan = model.predict(luoi).reshape(x1.shape)
    ax.contourf(x1, x2, du_doan, levels=[-0.5, 0.5, 1.5],
                colors=["#f7dada", "#d9e8f7"], alpha=0.65)
    ax.contour(x1, x2, du_doan, levels=[0.5], colors="#374151", linewidths=1)
    for nhan, mau, dau in [(0, "#db6c67", "^"), (1, "#4779a7", "o")]:
        nhom = y == nhan
        ax.scatter(X.loc[nhom, "x1"], X.loc[nhom, "x2"],
                   c=mau, marker=dau, s=15, alpha=0.75, label=f"Lớp {nhan}")
    ax.set(title=ten, xlabel="x1")

axes[0].set_ylabel("x2")
axes[-1].legend(fontsize=8)
fig.suptitle("So sánh biên quyết định trên hai vầng trăng")
fig.tight_layout()
fig.savefig(THU_MUC_HINH / "hinh_5_so_sanh_bien.png", dpi=180)

# Hình 6: mô phỏng minh hoạ lời nguyền số chiều, không dùng CSV.
# PDF không nêu seed/cách lấy mốc; ở đây dùng điểm gốc và 500 điểm
# phân bố đều trong [0, 1]^d, lấy trung bình tỷ số qua 20 lần lặp.
# Vì quy trình mô phỏng được ghi rõ, số đo có thể khác hình trong PDF.
rng = np.random.default_rng(17)
so_chieu = [1, 2, 5, 10, 20, 50, 100, 200, 500]
ty_so = []
for d in so_chieu:
    lap = []
    for _ in range(20):
        cac_diem = rng.random((500, d))
        khoang_cach = np.linalg.norm(cac_diem, axis=1)
        lap.append(khoang_cach.min() / khoang_cach.max())
    ty_so.append(np.mean(lap))
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(so_chieu, ty_so, marker="o", color="#4779a7")
ax.set(xscale="log", ylim=(0, 1), xlabel="Số đặc trưng (thang loga)",
       ylabel="Khoảng cách gần nhất / xa nhất",
       title="Nhiều chiều thì ai cũng xa gần như nhau")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(THU_MUC_HINH / "hinh_6_loi_nguyen_so_chieu.png", dpi=180)
plt.show()
