# ‐*‐ coding: utf‐8 ‐*‐
"""Bước 2: đo xem một đường thẳng dự đoán sai bao nhiêu.

Chỉ lấy 5 căn cho dễ nhìn. Thử hai đường thẳng khác nhau rồi so sai số
trung bình bình phương của từng đường.
"""

import pandas as pd
df = pd.read_csv('../data/gia_nha.csv')

nho = df.iloc[[0, 14, 29, 44, 59]]

x = nho["dien_tich"].to_numpy()
y = nho["gia"].to_numpy()

def mse(w, b):
    y_su_doan = w * x + b
    sai_so = y - y_su_doan
    return (sai_so ** 2).mean()

print("======= Bước 2 =======")
print("Nam can duoc chon:")
for i in range(5):
    print(f" {x[i]:6.1f} m2 - {y[i]:5.2f} ty")
print()

for w, b in [(0.05, 1.5), (0.08, 0.5)]:
    print(f"Duong thang y = {w} * x + {b}")
    for i in range(5):
        du_doan = w * x[i] + b
        sai_so = y[i] - du_doan
        print(f"x = {x[i]:6.1f} that = {y[i]:5.2f}"
            f" du doan = {du_doan:5.2f} sai so = {sai_so:+6.2f}")
    print(f"mse = {mse(w, b):.4f}")
    print()
    
print("Duong nao có MSE nho hon thi duong do khop du lieu tot hon.")

# ===== Vẽ đồ thị tương ứng Hình 3 trong tài liệu =====
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

thu_muc_outputs = Path(__file__).resolve().parent.parent / "figures"
thu_muc_outputs.mkdir(parents=True, exist_ok=True)

w_ve, b_ve = 0.08, 0.5
x_line = np.linspace(x.min() - 5, x.max() + 5, 200)
y_line = w_ve * x_line + b_ve
y_pred_ve = w_ve * x + b_ve
sai_so_ve = y - y_pred_ve

plt.figure(figsize=(8, 5))
plt.scatter(x, y, label="Giá thật", zorder=3)
plt.plot(x_line, y_line, label="y = 0.08x + 0.5")

for xi, yi, ypi, ei in zip(x, y, y_pred_ve, sai_so_ve):
    plt.vlines(xi, min(yi, ypi), max(yi, ypi), linestyles="--", alpha=0.8)
    plt.text(xi + 0.8, (yi + ypi) / 2, f"{ei:+.2f}", fontsize=8)

plt.xlabel("Diện tích (m²)")
plt.ylabel("Giá (tỷ đồng)")
plt.title("Sai số của từng căn là đoạn thẳng đứng")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

duong_dan_hinh = thu_muc_outputs / "b2_sai_so_5_can.png"
plt.savefig(duong_dan_hinh, dpi=150)
plt.close()
print("Da luu hinh:", duong_dan_hinh)
