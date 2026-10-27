# -*- coding: utf-8 -*-
"""Bước 3: chọn số láng giềng k bằng kiểm định chéo."""

import pandas as pd
from pathlib import Path

# Đường dẫn theo vị trí file, chạy được từ thư mục gốc hoặc code/.
THU_MUC_GOC = Path(__file__).resolve().parent.parent
THU_MUC_HINH = THU_MUC_GOC / "figures"
THU_MUC_HINH.mkdir(parents=True, exist_ok=True)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv(THU_MUC_GOC / "data" / "hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Hai cột x1 và x2 có thang đo gần nhau, độ lệch chuẩn chỉ chênh khoảng
# 1.6 lần, nên ở bộ này không chuẩn hoá cũng được.
print("Accuracy tren tap hoc va tap kiem tra theo k")
print("   k    tap hoc   tap kiem tra")
for k in [1, 5, 13, 31, 101]:
    mh = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
    a_hoc = accuracy_score(y_tr, mh.predict(X_tr))
    a_kt = accuracy_score(y_te, mh.predict(X_te))
    print(f" {k:3d}    {a_hoc:.4f}      {a_kt:.4f}")
print()

# Chọn k chỉ dựa trên tập học, không nhìn tập kiểm tra
print("Kiem dinh cheo 5 phan tren tap hoc:")
diem = {}
for k in range(1, 52, 2):
    diem[k] = cross_val_score(KNeighborsClassifier(n_neighbors=k),
                              X_tr, y_tr, cv=5).mean()
k_tot = max(diem, key=diem.get)
for k in [1, 5, 9, 13, 17, 25, 51]:
    dau = " <- tot nhat" if k == k_tot else ""
    print(f" k = {k:2d} diem trung binh {diem[k]:.4f}{dau}")
print()

# Chọn xong mới mở tập kiểm tra, và chỉ mở một lần
mh = KNeighborsClassifier(n_neighbors=k_tot).fit(X_tr, y_tr)
print(f"k tot nhat = {k_tot}")
acc = accuracy_score(y_te, mh.predict(X_te))
print(f"Accuracy tren tap kiem tra = {acc:.4f}")

# ==================== VẼ HÌNH THEO LAB04 ====================
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

# Hình 3: ba giá trị k minh hoạ học vẹt, biên mềm và quá đơn giản.

x1, x2 = np.meshgrid(np.linspace(X.x1.min() - 0.3, X.x1.max() + 0.3, 220),
                     np.linspace(X.x2.min() - 0.3, X.x2.max() + 0.3, 180))
luoi = pd.DataFrame({"x1": x1.ravel(), "x2": x2.ravel()})

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
for ax, k in zip(axes, [1, 13, 101]):
    model = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
    ten = f"k = {k}"

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
axes[-1].legend(loc="upper right", fontsize=8)
fig.suptitle("Biên quyết định của KNN theo số láng giềng")
fig.tight_layout()
fig.savefig(THU_MUC_HINH / "hinh_3_bien_theo_k.png", dpi=180)

# Hình 4: chỉ minh hoạ chẩn đoán; KHÔNG dùng đường test để chọn k.
# k_tot phía trên vẫn được chọn bằng cross-validation chỉ trên tập học.
cac_k = list(range(1, 102))
diem_hoc, diem_kiem_tra = [], []
for k in cac_k:
    model = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
    diem_hoc.append(accuracy_score(y_tr, model.predict(X_tr)))
    diem_kiem_tra.append(accuracy_score(y_te, model.predict(X_te)))
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(cac_k, diem_hoc, label="Tập học", color="#4779a7")
ax.plot(cac_k, diem_kiem_tra, label="Tập kiểm tra", color="#db6c67")
ax.axvline(k_tot, linestyle="--", color="#409976", label=f"k chọn bằng CV = {k_tot}")
ax.set(title="Accuracy theo số láng giềng", xlabel="Số láng giềng k", ylabel="Accuracy",
       ylim=(0, 1.03))
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
fig.savefig(THU_MUC_HINH / "hinh_4_accuracy_theo_k.png", dpi=180)
plt.show()
