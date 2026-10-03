# ‐*‐ coding: utf‐8 ‐*‐
"""Bước 6: thêm điểm giữa kỳ làm biến thứ hai."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/sinh_vien.csv")
y = df["qua_mon"]

print("========== Bước 6 ==========")
# Chỉ đổi đúng một chỗ: danh sách cột đưa vào X
for ten, cot in [("Mot bien", ["gio_on"]),
                 ("Hai bien", ["gio_on", "diem_giua_ky"])]:
    X = df[cot]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)
    mo_hinh = LogisticRegression().fit(X_train, y_train)
    do_chinh_xac = accuracy_score(y_test, mo_hinh.predict(X_test))
    print(f"{ten}: accuracy tren tap kiem tra = {do_chinh_xac:.4f}")
print()

# Xem kỹ hệ số của mô hình hai biến
X = df[["gio_on", "diem_giua_ky"]]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh = LogisticRegression().fit(X_train, y_train)

print("He so cua tung bien:")
for ten_cot, he_so in zip(X.columns, mo_hinh.coef_[0]):
    print(f" {ten_cot:14s} {he_so:+.4f}")
print(f" {'he so chan':14s} {mo_hinh.intercept_[0]:+.4f}")
print()
print("Ca hai he so deu duong, nghia la on nhieu hon va")
print("diem giua ky cao hon deu lam tang xac suat qua mon.")

# ===== VẼ HÌNH 7: biên quyết định của mô hình hai biến =====
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

w1, w2 = mo_hinh.coef_[0]
b = mo_hinh.intercept_[0]
xx, yy = np.meshgrid(np.linspace(0, 30, 300), np.linspace(0, 10, 300))
luoi = pd.DataFrame({"gio_on": xx.ravel(), "diem_giua_ky": yy.ravel()})
zz = mo_hinh.predict(luoi).reshape(xx.shape)

plt.figure(figsize=(9, 6))
plt.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], alpha=0.25)

qua = df["qua_mon"] == 1
rot = df["qua_mon"] == 0
plt.scatter(df.loc[qua, "gio_on"], df.loc[qua, "diem_giua_ky"],
            s=28, edgecolors="white", linewidths=0.5, label="Qua môn")
plt.scatter(df.loc[rot, "gio_on"], df.loc[rot, "diem_giua_ky"],
            s=28, edgecolors="white", linewidths=0.5, label="Rớt môn")

x_bien = np.linspace(0, 30, 300)
y_bien = -(w1 * x_bien + b) / w2
hop_le = (y_bien >= 0) & (y_bien <= 10)
plt.plot(x_bien[hop_le], y_bien[hop_le], linewidth=2)

plt.xlim(0, 30); plt.ylim(0, 10)
plt.xlabel("Số giờ ôn tập"); plt.ylabel("Điểm giữa kỳ")
plt.title("Biên quyết định của mô hình hai biến")
plt.grid(alpha=0.2); plt.legend(); plt.tight_layout()
plt.savefig(OUTPUT_DIR / "c6_hinh7_bien_quyet_dinh_hai_bien.png", dpi=150)
plt.close()
