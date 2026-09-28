
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