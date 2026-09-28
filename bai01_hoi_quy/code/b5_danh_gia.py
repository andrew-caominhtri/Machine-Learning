# ‐*‐ coding: utf‐8 ‐*‐
"""Bước 5: chia dữ liệu và chấm điểm mô hình bằng các độ đo chuẩn."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]

# 80 phần trăm để học, 20 phần trăm để kiểm tra.
# random_state cố định để lần nào chia cũng ra đúng một cách chia.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("======= Bước 5 =======")
print("So can de hoc :", len(X_train))
print("So can de kiem tra:", len(X_test))
print()

mo_hinh = LinearRegression()
mo_hinh.fit(X_train, y_train) # chỉ học trên tập học

y_pred = mo_hinh.predict(X_test) # chấm điểm trên tập kiểm tra

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE = {mae:.4f} ty dong")
print(f"MSE = {mse:.4f}")
print(f"RMSE = {rmse:.4f} ty dong")
print(f"R2 = {r2:.4f}")
print()

print("Mot vai can trong tap kiem tra:")
for dt, that, du_doan in list(zip(X_test["dien_tich"], y_test, y_pred))[:5]:
    print(f" {dt:6.1f} m2 that = {that:5.2f} du doan = {du_doan:5.2f}"
          f" sai so = {that - du_doan:+5.2f}")

# ===== Vẽ đồ thị đánh giá tương ứng Hình 8 và Hình 9 =====
from pathlib import Path
import matplotlib.pyplot as plt

thu_muc_outputs = Path(__file__).resolve().parent.parent / "figures"
thu_muc_outputs.mkdir(parents=True, exist_ok=True)

# Hình 8: Giá thật và giá dự đoán
muc_nho = min(float(y_test.min()), float(y_pred.min()))
muc_lon = max(float(y_test.max()), float(y_pred.max()))

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred)
plt.plot([muc_nho, muc_lon], [muc_nho, muc_lon], "--",
         label="Dự đoán đúng hoàn toàn")
plt.xlabel("Giá thật (tỷ đồng)")
plt.ylabel("Giá dự đoán (tỷ đồng)")
plt.title(f"Giá thật và giá dự đoán - R² = {r2:.4f}")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
duong_dan_hinh_8 = thu_muc_outputs / "b5_hinh8_gia_that_va_du_doan.png"
plt.savefig(duong_dan_hinh_8, dpi=150)
plt.close()

# Hình 9: Phần dư theo diện tích
phan_du = y_test.to_numpy() - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(X_test["dien_tich"], phan_du)
plt.axhline(0, linestyle="--")
plt.xlabel("Diện tích (m²)")
plt.ylabel("Phần dư (tỷ đồng)")
plt.title("Phần dư rải đều hai phía đường 0 là dấu hiệu tốt")
plt.grid(alpha=0.3)
plt.tight_layout()
duong_dan_hinh_9 = thu_muc_outputs / "b5_hinh9_phan_du.png"
plt.savefig(duong_dan_hinh_9, dpi=150)
plt.close()

print("Da luu hinh:", duong_dan_hinh_8)
print("Da luu hinh:", duong_dan_hinh_9)
