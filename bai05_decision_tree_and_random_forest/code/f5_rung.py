# -*- coding: utf-8 -*-
"""Bước 5: trồng rừng ngẫu nhiên và so với một cây."""


from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import matplotlib
matplotlib.use("Agg")  # Chỉ lưu ảnh, không mở cửa sổ.
import matplotlib.pyplot as plt

df = pd.read_csv("../data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Bốc mẫu có hoàn lại: rút 450 lần từ 450 khách
rng = np.random.default_rng(0)
n = len(X_tr)
chi_so = rng.integers(0, n, n)
khac = len(np.unique(chi_so))
print(f"Boc {n} lan, co {khac} khach khac nhau ({khac / n:.0%})")
print()

rung = RandomForestClassifier(n_estimators=200, random_state=0)
rung.fit(X_tr, y_tr)
print("Rung 200 cay")
print(f" tap hoc       : {accuracy_score(y_tr, rung.predict(X_tr)):.4f}")
print(f" tap kiem tra : {accuracy_score(y_te, rung.predict(X_te)):.4f}")
print()

print("So cay   accuracy tap kiem tra")
for m in [1, 10, 50, 200, 400]:
    r = RandomForestClassifier(n_estimators=m, random_state=0)
    r.fit(X_tr, y_tr)
    print(f" {m:4d}    {accuracy_score(y_te, r.predict(X_te)):.4f}")
print()

# Bảng tổng kết ba mô hình trên cùng tập kiểm tra
ba_mo_hinh = {
    "Cay moc tu do": DecisionTreeClassifier(random_state=0),
    "Cay sau 5": DecisionTreeClassifier(max_depth=5, random_state=0),
    "Rung 200 cay": rung,
}
print("Mo hinh          tap hoc   tap kiem tra")
for ten, mh in ba_mo_hinh.items():
    mh.fit(X_tr, y_tr)
    a = accuracy_score(y_tr, mh.predict(X_tr))
    b = accuracy_score(y_te, mh.predict(X_te))
    print(f" {ten:14s} {a:.4f}     {b:.4f}")


# ==================== VẼ VÀ CHỈ LƯU HÌNH ====================

THU_MUC_HINH = Path("../figures")
THU_MUC_HINH.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

# Hình 5: điểm số đúng từ các rừng đã chấm ở vòng lặp phía trên.
# Tính riêng dữ liệu vẽ, giữ nguyên vòng lặp code gốc trong PDF.
so_cay_hinh = [1, 10, 50, 200, 400]
accuracy_hinh = []
for so_cay in so_cay_hinh:
    model_hinh = RandomForestClassifier(n_estimators=so_cay, random_state=0)
    model_hinh.fit(X_tr, y_tr)
    accuracy_hinh.append(accuracy_score(y_te, model_hinh.predict(X_te)))
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(so_cay_hinh, accuracy_hinh, marker="o", color="#3678a8", label="Rừng ngẫu nhiên")
diem_cay = accuracy_score(y_te, ba_mo_hinh["Cay moc tu do"].predict(X_te))
ax.axhline(diem_cay, color="#d97732", linestyle="--", label=f"Một cây tự do: {diem_cay:.4f}")
ax.set(title="Accuracy theo số cây trong rừng", xlabel="Số cây", ylabel="Accuracy trên tập kiểm tra", ylim=(0, 1.03))
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
fig.savefig(THU_MUC_HINH / "hinh_5_accuracy_theo_so_cay.png", dpi=180)
plt.close(fig)

# Hình 6: cây và rừng 200 cây, chỉ dùng hai cột để vẽ biên.

# Biên 2D: fit RIÊNG mô hình trên hai cột, không coi là biên của mô hình 4 cột.
cot_2d = ["thu_nhap", "ty_le_no"]
x1, x2 = np.meshgrid(
    np.linspace(X.thu_nhap.min() - 1, X.thu_nhap.max() + 1, 230),
    np.linspace(max(0, X.ty_le_no.min() - 0.05), X.ty_le_no.max() + 0.05, 180))
luoi = pd.DataFrame({"thu_nhap": x1.ravel(), "ty_le_no": x2.ravel()})

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
for ax, (ten, model) in zip(axes, [
        ("Cây mọc tự do", DecisionTreeClassifier(random_state=0)),
        ("Rừng 200 cây", RandomForestClassifier(n_estimators=200, random_state=0))]):

    model.fit(X_tr[cot_2d], y_tr)
    zz = model.predict(luoi).reshape(x1.shape)
    ax.contourf(x1, x2, zz, levels=[-0.5, 0.5, 1.5], colors=["#fae4d1", "#d9eafa"], alpha=0.7)
    ax.contour(x1, x2, zz, levels=[0.5], colors="#475569", linewidths=0.8)
    for nhan, ten_lop, mau, dau in [(0, "Trễ hạn", "#d97732", "^"),
                                    (1, "Đúng hạn", "#3678a8", "o")]:
        nhom = y == nhan
        ax.scatter(X.loc[nhom, "thu_nhap"], X.loc[nhom, "ty_le_no"],
                   c=mau, marker=dau, s=13, alpha=0.6, label=ten_lop)
    ax.set(title=ten, xlabel="Thu nhập (triệu đồng)")

axes[0].set_ylabel("Tỷ lệ nợ")
axes[-1].legend(fontsize=9)
fig.suptitle("So sánh cây và rừng trên hai đặc trưng")
fig.tight_layout()
fig.savefig(THU_MUC_HINH / "hinh_6_bien_cay_va_rung.png", dpi=180)
plt.close(fig)
print("Da luu hinh 5 va 6 tai:", THU_MUC_HINH)
