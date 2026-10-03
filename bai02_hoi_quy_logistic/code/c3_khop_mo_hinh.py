# ‐*‐ coding: utf‐8 ‐*‐
"""Bước 3: để scikit‐learn tìm w và b từ dữ liệu thật."""

import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("../data/sinh_vien.csv")

# X phải là bảng hai chiều nên viết hai cặp ngoặc vuông.
# y là dãy một chiều nên chỉ một cặp.
X = df[["gio_on"]]
y = df["qua_mon"]

mo_hinh = LogisticRegression()
mo_hinh.fit(X, y)

w = float(mo_hinh.coef_[0][0])
b = float(mo_hinh.intercept_[0])

print("========== Bước3 ==========")
print(f"He so goc w = {w:.6f}")
print(f"He so chan b = {b:.6f}")
print()

# Số giờ ôn mà mô hình phân vân đúng năm mươi năm mươi
print(f"So gio on ung voi xac suat 0.5: {-b / w:.2f} gio")
print()

can_moi = pd.DataFrame({"gio_on": [5.0, 10.0, 13.0, 20.0, 28.0]})
xac_suat = mo_hinh.predict_proba(can_moi)[:, 1]
nhan = mo_hinh.predict(can_moi)

print("Du doan cho nam ban moi:")
print(" So gio on Xac suat qua Nhan mo hinh dua ra")
for gio, p, n in zip(can_moi["gio_on"], xac_suat, nhan):
    print(f" {gio:5.1f} {p:.4f} {n}")

# ===== VẼ HÌNH 3: đường cong logistic khớp trên dữ liệu =====
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

x_ve = np.linspace(0, 30, 400)
X_ve = pd.DataFrame({"gio_on": x_ve})
p_ve = mo_hinh.predict_proba(X_ve)[:, 1]
moc_05 = -b / w

plt.figure(figsize=(9, 5))
qua = y.to_numpy() == 1
rot = y.to_numpy() == 0
plt.scatter(df.loc[qua, "gio_on"], y[qua], s=22, label="Qua môn")
plt.scatter(df.loc[rot, "gio_on"], y[rot], s=22, label="Rớt môn")
plt.plot(x_ve, p_ve, linewidth=2.2, label="Xác suất qua môn do mô hình đoán")
plt.axhline(0.5, linestyle=":", linewidth=1)
plt.axvline(moc_05, linestyle="--", linewidth=1.5)
plt.text(moc_05 + 0.4, 0.08, f"{moc_05:.1f} giờ", rotation=90, va="bottom")
plt.xlim(0, 30); plt.ylim(-0.05, 1.05)
plt.xlabel("Số giờ ôn tập"); plt.ylabel("Xác suất qua môn")
plt.title("Đường cong logistic khớp trên 120 sinh viên")
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(OUTPUT_DIR / "c3_hinh3_duong_cong_logistic.png", dpi=150)
plt.close()
