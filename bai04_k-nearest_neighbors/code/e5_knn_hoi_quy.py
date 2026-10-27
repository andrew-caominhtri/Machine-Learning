# -*- coding: utf-8 -*-
"""Bước 5: KNN cho bài toán hồi quy, trên dữ liệu giá nhà của bài 1."""

import pandas as pd
from pathlib import Path

# Đường dẫn theo vị trí file, chạy được từ thư mục gốc hoặc code/.
THU_MUC_GOC = Path(__file__).resolve().parent.parent
THU_MUC_HINH = THU_MUC_GOC / "figures"
THU_MUC_HINH.mkdir(parents=True, exist_ok=True)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv(THU_MUC_GOC / "data" / "gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]

# Cùng cách chia với bài 1 để so sánh công bằng
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                         random_state=42)

knn = KNeighborsRegressor(n_neighbors=5).fit(X_tr, y_tr)
duong = LinearRegression().fit(X_tr, y_tr)

print("MAE tren tap kiem tra (ty dong)")
mae_knn = mean_absolute_error(y_te, knn.predict(X_te))
mae_duong = mean_absolute_error(y_te, duong.predict(X_te))
print(f" KNN, k = 5 : {mae_knn:.4f}")
print(f" Duong thang : {mae_duong:.4f}")
print()

can_moi = pd.DataFrame({"dien_tich": [60.0, 100.0, 150.0, 250.0]})
print("Du doan cho bon can moi (ty dong)")
print(" dien tich      KNN   duong thang")
for dt, a, b in zip(can_moi["dien_tich"], knn.predict(can_moi),
                    duong.predict(can_moi)):
    print(f" {dt:6.0f} m2    {a:6.2f}   {b:6.2f}")
print()

# Năm láng giềng của căn 250 m2 chính là những căn rộng nhất đã thấy
_, chi_so = knn.kneighbors(pd.DataFrame({"dien_tich": [250.0]}))
lg = X_tr.iloc[chi_so[0]]["dien_tich"].tolist()
print("Nam lang gieng cua can 250 m2:", ", ".join(f"{v:.1f}" for v in lg))

# ==================== VẼ HÌNH THEO LAB04 ====================
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

# Hình 7: kéo miền dự đoán tới 250 m² để thấy giới hạn ngoại suy.
dien_tich = np.linspace(min(30, X.dien_tich.min()), max(250, X.dien_tich.max()), 700)
luoi = pd.DataFrame({"dien_tich": dien_tich})
fig, ax = plt.subplots(figsize=(10, 5))
ax.axvspan(X_tr.dien_tich.min(), X_tr.dien_tich.max(), color="#d9e8f7",
           alpha=0.5, label="Khoảng diện tích trong tập học")
ax.scatter(X.dien_tich, y, s=25, color="#4779a7", alpha=0.65, label="Dữ liệu giá nhà")
ax.plot(dien_tich, knn.predict(luoi), color="#db6c67", linewidth=2, label="KNN, k = 5")
ax.plot(dien_tich, duong.predict(luoi), color="#409976", linewidth=2,
        label="Hồi quy tuyến tính")
ax.set(title="KNN không ngoại suy xu hướng ngoài vùng dữ liệu",
       xlabel="Diện tích (m²)", ylabel="Giá (tỷ đồng)")
ax.grid(alpha=0.2)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(THU_MUC_HINH / "hinh_7_knn_hoi_quy.png", dpi=180)
plt.show()
