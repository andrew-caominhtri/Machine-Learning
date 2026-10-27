# -*- coding: utf-8 -*-
"""Bước 1: tự tính khoảng cách và tự tìm láng giềng, chưa dùng thư viện.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Đường dẫn theo vị trí file, chạy được từ thư mục gốc hoặc code/.
THU_MUC_GOC = Path(__file__).resolve().parent.parent
THU_MUC_HINH = THU_MUC_GOC / "figures"
THU_MUC_HINH.mkdir(parents=True, exist_ok=True)

# Ba bạn: A và C học gần như y hệt nhau, B học khác hẳn
ba_ban = {
    "A": np.array([5.0, 3.0, 4000.0]),
    "B": np.array([25.0, 9.0, 4100.0]),
    "C": np.array([6.0, 3.5, 7000.0]),
}


def kc(u, v):
    """Khoang cach Euclid giua hai diem."""
    return np.sqrt(((u - v) ** 2).sum())


print("Khoang cach KHI CHUA chuan hoa")
print(f" d(A, B) = {kc(ba_ban['A'], ba_ban['B']):8.1f}")
print(f" d(A, C) = {kc(ba_ban['A'], ba_ban['C']):8.1f}")
print()

# Chuẩn hoá bằng trung bình và độ lệch chuẩn của cả bộ dữ liệu
df = pd.read_csv(THU_MUC_GOC / "data" / "sinh_vien_mo_rong.csv")
X = df[["gio_on", "diem_giua_ky", "chi_tieu"]].to_numpy(dtype=float)
tb = X.mean(axis=0)
dl = X.std(axis=0)
print("Do lech chuan cua ba cot:")
print(f" gio_on {dl[0]:.2f}    diem_giua_ky {dl[1]:.2f}"
      f"   chi_tieu {dl[2]:.0f}")
print()

chuan = {ten: (v - tb) / dl for ten, v in ba_ban.items()}
print("Khoang cach SAU KHI chuan hoa")
print(f" d(A, B) = {kc(chuan['A'], chuan['B']):.2f}")
print(f" d(A, C) = {kc(chuan['A'], chuan['C']):.2f}")
print()

# Tìm 5 láng giềng gần nhất của một bạn mới, chỉ dùng hai cột học tập
X2 = df[["gio_on", "diem_giua_ky"]].to_numpy(dtype=float)
y = df["qua_mon"].to_numpy()
tb2, dl2 = X2.mean(axis=0), X2.std(axis=0)
ban_moi = np.array([12.0, 6.5])
X2c = (X2 - tb2) / dl2
moi_c = (ban_moi - tb2) / dl2
khoang = np.sqrt(((X2c - moi_c) ** 2).sum(axis=1))
gan = np.argsort(khoang)[:5]

print("Ban moi: on 12 gio, diem giua ky 6.5")
print("Nam lang gieng gan nhat:")
for i in gan:
    ket = "qua" if y[i] == 1 else "rot"
    print(f" gio {X2[i, 0]:5.1f} diem {X2[i, 1]:4.1f}       "
          f"khoang cach {khoang[i]:.3f} -> {ket}")
so_qua = int(y[gan].sum())
print(f"Bo phieu: {so_qua} qua, {5 - so_qua} rot")
print("Du doan:", "qua mon" if so_qua >= 3 else "rot mon")

# ==================== VẼ HÌNH THEO LAB04 ====================
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

# Hình 1: các láng giềng được chọn theo khoảng cách ĐÃ chuẩn hoá ở trên.
fig, ax = plt.subplots(figsize=(9, 5))
for nhan, ten, mau, dau in [(1, "Qua môn", "#4779a7", "o"),
                            (0, "Rớt môn", "#db6c67", "^")]:
    nhom = y == nhan
    ax.scatter(X2[nhom, 0], X2[nhom, 1], c=mau, marker=dau,
               s=25, alpha=0.7, label=ten)
ax.scatter(X2[gan, 0], X2[gan, 1], s=170, facecolors="none",
           edgecolors="#b58300", linewidths=1.8, label="5 láng giềng gần nhất")
ax.scatter(*ban_moi, s=200, marker="*", c="#148547", edgecolors="white",
           linewidths=0.6, zorder=5, label="Bạn mới")
ax.set(title="Hỏi năm người hàng xóm gần nhất", xlabel="Số giờ ôn tập", ylabel="Điểm giữa kỳ")
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(THU_MUC_HINH / "hinh_1_lang_gieng.png", dpi=180)
plt.show()
