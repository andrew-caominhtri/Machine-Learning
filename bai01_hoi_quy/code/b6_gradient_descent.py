# ‐*‐ coding: utf‐8 ‐*‐
"""Bước 6: tự đi tìm w và b bằng cách dò từng bước nhỏ.

Đây là cách máy học khi bài toán không có công thức đóng. Với hồi quy tuyến
tính ta đã có công thức rồi, nên đoạn này chỉ để thấy cơ chế hoạt động.
"""

import numpy as np
import pandas as pd

df = pd.read_csv("../data/gia_nha.csv")
x_goc = df["dien_tich"].to_numpy()
y = df["gia"].to_numpy()

# Đưa diện tích về thang đo nhỏ quanh số 0. Thiếu bước này thuật toán sẽ vỡ.
x_tb = x_goc.mean()
x_do_lech = x_goc.std()
x = (x_goc - x_tb) / x_do_lech

w, b = 0.0, 0.0     # bắt đầu từ một đường thẳng nằm ngang đi qua gốc
toc_do_hoc = 0.1    # mỗi vòng lặp đi một bước dài bằng 0.1 lần độ dốc
so_vong = 200
n = len(x)

lich_su_mse = [(((w * x + b) - y) ** 2).mean()]
cac_moc = {0: (w, b)}

print("Vong w b MSE")
for vong in range(1, so_vong + 1):
    y_du_doan = w * x + b
    # Chú ý: đây là dự đoán trừ giá thật, tức sai số của Mục 4 đảo dấu.
    # Hai công thức độ dốc bên dưới viết theo đúng chiều này, đừng đổi dấu.
    chenh_lech = y_du_doan - y

    # Độ dốc của MSE theo w và theo b
    grad_w = (2 / n) * (chenh_lech * x).sum()
    grad_b = (2 / n) * chenh_lech.sum()

    # Đi ngược hướng dốc thì sai số giảm
    w -= toc_do_hoc * grad_w
    b -= toc_do_hoc * grad_b

    mse_hien_tai = (((w * x + b) - y) ** 2).mean()
    lich_su_mse.append(mse_hien_tai)
    if vong in (2, 5, 15, 60):
        cac_moc[vong] = (w, b)

    if vong in (1, 2, 5, 10, 25, 50, 100, 200):
        # Tính lại MSE bằng w và b VỪA cập nhật, để ba con số trên cùng một
        # dòng bảng đều thuộc về cùng một thời điểm. Nếu dùng lại chenh_lech
        # ở trên thì MSE sẽ là của cặp w, b cũ và bảng bị lệch một vòng.
        mse = (((w * x + b) - y) ** 2).mean()
        print(f"{vong:4d} {w:7.4f} {b:7.4f} {mse:8.4f}")

print("========== Bước 6 ==========")
print()
# Đổi w và b về lại thang đo mét vuông ban đầu
w_goc = w / x_do_lech
b_goc = b - w * x_tb / x_do_lech
print(f"Sau khi doi ve thang do met vuong:")
print(f" w = {w_goc:.6f}")

print(f" b = {b_goc:.6f}")
print("So voi cong thuc o buoc 3: w = 0.078367, b = 0.401752")

# ===== Vẽ đồ thị tương ứng Hình 10 và Hình 11 =====
from pathlib import Path
import matplotlib.pyplot as plt

thu_muc_outputs = Path(__file__).resolve().parent.parent / "figures"
thu_muc_outputs.mkdir(parents=True, exist_ok=True)

# Hình 10: MSE giảm theo vòng lặp, trục tung loga
plt.figure(figsize=(8, 5))
plt.plot(range(0, so_vong + 1), lich_su_mse)
plt.yscale("log")
plt.xlabel("Vòng lặp")
plt.ylabel("MSE (thang loga)")
plt.title("Sai số giảm nhanh rồi đứng yên")
plt.grid(alpha=0.3)
plt.tight_layout()
duong_dan_hinh_10 = thu_muc_outputs / "b6_hinh10_mse_gradient_descent.png"
plt.savefig(duong_dan_hinh_10, dpi=150)
plt.close()

# Hình 11: Các đường thẳng dịch dần về vị trí đúng
x_goc_ve = np.linspace(x_goc.min(), x_goc.max(), 200)
x_chuan_hoa_ve = (x_goc_ve - x_tb) / x_do_lech

plt.figure(figsize=(8, 5))
plt.scatter(x_goc, y, alpha=0.65, label="Dữ liệu")
for moc in (0, 2, 5, 15, 60):
    w_moc, b_moc = cac_moc[moc]
    y_moc = w_moc * x_chuan_hoa_ve + b_moc
    nhan = "Lúc bắt đầu" if moc == 0 else f"Sau {moc} vòng"
    plt.plot(x_goc_ve, y_moc, label=nhan)

plt.xlabel("Diện tích (m²)")
plt.ylabel("Giá (tỷ đồng)")
plt.title("Đường thẳng tự dịch dần về đúng chỗ")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
duong_dan_hinh_11 = thu_muc_outputs / "b6_hinh11_cac_buoc_gradient_descent.png"
plt.savefig(duong_dan_hinh_11, dpi=150)
plt.close()

print("Da luu hinh:", duong_dan_hinh_10)
print("Da luu hinh:", duong_dan_hinh_11)
