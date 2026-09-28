# ‐*‐ coding: utf‐8 ‐*‐
"""Bước 4: làm lại đúng việc đó bằng scikit‐learn, chỉ mất ba dòng."""

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("../data/gia_nha.csv")
#
 # X phải là bảng hai chiều, y là dãy một chiều.
 # Chú ý X viết hai cặp ngoặc vuông, còn y chỉ một cặp.
X = df[["dien_tich"]]
y = df["gia"]

mo_hinh = LinearRegression()
mo_hinh.fit(X, y)

print("======= Bước 4 =======")
print("He so goc w =", round(float(mo_hinh.coef_[0]), 6))
print("He so chan b =", round(float(mo_hinh.intercept_), 6))
print()

# So sánh với kết quả tính tay ở bước 3
print("Ket qua tinh tay o buoc 3: w = 0.078367, b = 0.401752")
print()

can_moi = pd.DataFrame({"dien_tich": [80.0, 100.0]})
du_doan = mo_hinh.predict(can_moi)
for dt, gia in zip(can_moi["dien_tich"], du_doan):
    print(f"Can {dt:.0f} m2 ‐> du doan {gia:.3f} ty dong")

# ===== Vẽ đường hồi quy học bằng scikit-learn =====
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

thu_muc_outputs = Path(__file__).resolve().parent.parent / "figures"
thu_muc_outputs.mkdir(parents=True, exist_ok=True)

x_ve = np.linspace(df["dien_tich"].min(), df["dien_tich"].max(), 200)
X_ve = pd.DataFrame({"dien_tich": x_ve})
y_ve = mo_hinh.predict(X_ve)

plt.figure(figsize=(8, 5))
plt.scatter(df["dien_tich"], y, label="Dữ liệu")
plt.plot(x_ve, y_ve, label="Đường hồi quy scikit-learn")
plt.scatter(can_moi["dien_tich"], du_doan, marker="*", s=150,
            label="Các căn cần dự đoán", zorder=4)
plt.xlabel("Diện tích (m²)")
plt.ylabel("Giá (tỷ đồng)")
plt.title("Hồi quy tuyến tính bằng scikit-learn")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

duong_dan_hinh = thu_muc_outputs / "b4_sklearn_duong_hoi_quy.png"
plt.savefig(duong_dan_hinh, dpi=150)
plt.close()
print("Da luu hinh:", duong_dan_hinh)
