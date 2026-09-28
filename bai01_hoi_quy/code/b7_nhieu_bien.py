# ‐*‐ coding: utf‐8 ‐*‐
"""Bước 7: dùng thêm số phòng và tuổi nhà, xem mô hình có khá hơn không."""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/gia_nha.csv")
y = df["gia"]

 # Chỉ đổi đúng một chỗ: danh sách cột đưa vào X
X_mot = df[["dien_tich"]]
X_ba = df[["dien_tich", "so_phong", "tuoi_nha"]]

ket_qua_r2 = {}
for ten, X in [("Mot bien ", X_mot), ("Ba bien ", X_ba)]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    mo_hinh = LinearRegression()
    mo_hinh.fit(X_train, y_train)
    r2 = r2_score(y_test, mo_hinh.predict(X_test))
    print(f"{ten}: R2 tren tap kiem tra = {r2:.4f}")
    ket_qua_r2[ten.strip()] = r2

print("=========== Bước 7 ===========")
print()

# Xem kỹ các hệ số của mô hình ba biến
X_train, X_test, y_train, y_test = train_test_split(
    X_ba, y, test_size=0.2, random_state=42
)
mo_hinh = LinearRegression()
mo_hinh.fit(X_train, y_train)

print("He so cua tung bien:")
for ten_cot, he_so in zip(X_ba.columns, mo_hinh.coef_):
    print(f" {ten_cot:12s} {he_so:+.4f}")
print(f" {'he so chan':12s} {mo_hinh.intercept_:+.4f}")

# ===== Minh họa so sánh mô hình một biến và ba biến =====
# Tài liệu giải thích rằng mô hình 3 biến không thể vẽ trực tiếp trong không
# gian 4 chiều, nên ta vẽ đúng đại lượng mà tệp này đang so sánh: R².
from pathlib import Path
import matplotlib.pyplot as plt

thu_muc_outputs = Path(__file__).resolve().parent.parent / "figures"
thu_muc_outputs.mkdir(parents=True, exist_ok=True)

ten_mo_hinh = list(ket_qua_r2.keys())
gia_tri_r2 = list(ket_qua_r2.values())

plt.figure(figsize=(7, 5))
cot = plt.bar(ten_mo_hinh, gia_tri_r2)
plt.ylabel("R² trên tập kiểm tra")
plt.title("So sánh mô hình một biến và ba biến")
plt.ylim(0, 1.0)
plt.grid(axis="y", alpha=0.3)

for c, value in zip(cot, gia_tri_r2):
    plt.text(c.get_x() + c.get_width()/2, value + 0.01,
             f"{value:.4f}", ha="center")

plt.tight_layout()
duong_dan_hinh = thu_muc_outputs / "b7_so_sanh_r2_mot_bien_ba_bien.png"
plt.savefig(duong_dan_hinh, dpi=150)
plt.close()
print("Da luu hinh:", duong_dan_hinh)
