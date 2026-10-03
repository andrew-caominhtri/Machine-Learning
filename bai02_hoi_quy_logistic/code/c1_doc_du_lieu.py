# ‐*‐ coding: utf‐8 ‐*‐
"""Bước 1: đọc bộ dữ liệu sinh viên và xem qua một lượt.

Chú thích trong tệp viết tiếng Việt có dấu, nhưng phần in ra màn hình cố ý
viết không dấu. Cửa sổ lệnh Windows thường không hiện được chữ có dấu.
"""

import pandas as pd

df = pd.read_csv("../data/sinh_vien.csv")

print("========== Bước1 ==========")
print("Kich thuoc bang (so dong, so cot):", df.shape)
print()

print("Nam dong dau tien:")
print(df.head())
print()

# Cột qua_mon chỉ có hai giá trị, đếm số lần xuất hiện từng giá trị
print("So sinh vien theo ket qua:")
print(df["qua_mon"].value_counts())
print()

print("Ty le qua mon:", round(df["qua_mon"].mean(), 4))
print()

# So sánh số giờ ôn trung bình của hai nhóm
print("So gio on trung binh theo nhom:")
print(df.groupby("qua_mon")["gio_on"].mean().round(2))

# ===== VẼ HÌNH 1: hồi quy tuyến tính không phù hợp với nhãn 0/1 =====
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

x = df["gio_on"].to_numpy()
y = df["qua_mon"].to_numpy()
w_linear, b_linear = np.polyfit(x, y, 1)
x_ve = np.linspace(0, 30, 300)
y_ve = w_linear * x_ve + b_linear

plt.figure(figsize=(9, 5))
plt.scatter(x[y == 1], y[y == 1], s=24, label="Qua môn (y = 1)")
plt.scatter(x[y == 0], y[y == 0], s=24, label="Rớt môn (y = 0)")
plt.plot(x_ve, y_ve, "--", linewidth=2, label="Đường thẳng hồi quy tuyến tính")
plt.axhspan(1, 1.5, alpha=0.12)
plt.axhspan(-0.5, 0, alpha=0.12)
plt.text(24, 1.30, "vùng vô nghĩa\n(lớn hơn 1)", ha="center")
plt.text(5, -0.35, "vùng vô nghĩa (nhỏ hơn 0)", ha="center")
plt.xlim(0, 30); plt.ylim(-0.5, 1.5)
plt.xlabel("Số giờ ôn tập")
plt.ylabel("Nhãn / giá trị dự đoán")
plt.title("Đường thẳng chạy ra ngoài khoảng 0 tới 1")
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(OUTPUT_DIR / "c1_hinh1_duong_thang_ngoai_khoang_0_1.png", dpi=150)
plt.close()
