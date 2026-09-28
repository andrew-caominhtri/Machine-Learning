# ‐*‐ coding: utf‐8 ‐*‐
"""Bước 1: đọc tệp dữ liệu và nhìn qua một lượt.

Chú thích trong tệp viết tiếng Việt có dấu, nhưng phần in ra màn hình cố ý
viết không dấu. Cửa sổ lệnh Windows thường không hiện được chữ có dấu.
"""

import pandas as pd
df = pd.read_csv("../data/gia_nha.csv")

print("======= Bước 1 =======")
print("Kich thuoc bang (so dong, so cot):", df.shape)
print()

print("Nam dong dau tien:")
print(df.head())
print()

print("Ten cac cot:", list(df.columns))
print()

print("Thong ke nhanh cot dien_tich va cot gia:")
print(df[["dien_tich", "gia"]].describe().round(2))
print()

print("So o bi thieu trong tung cot:")
print(df.isna().sum())

# ===== Vẽ đồ thị tương ứng Hình 1 trong tài liệu =====
from pathlib import Path
import matplotlib.pyplot as plt

thu_muc_outputs = Path(__file__).resolve().parent.parent / "figures"
thu_muc_outputs.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.scatter(df["dien_tich"], df["gia"], label="60 căn nhà")
plt.xlabel("Diện tích (m²)")
plt.ylabel("Giá (tỷ đồng)")
plt.title("Giá nhà theo diện tích")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

duong_dan_hinh = thu_muc_outputs / "b1_gia_nha_theo_dien_tich.png"
plt.savefig(duong_dan_hinh, dpi=150)
plt.close()
print("Da luu hinh:", duong_dan_hinh)
