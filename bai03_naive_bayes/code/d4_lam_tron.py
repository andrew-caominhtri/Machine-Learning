# ‐*‐ coding: utf‐8 ‐*‐
"""Bước 4: tự cài Naive Bayes trên một kho nhỏ để thấy vấn đề số không.

Đoạn này cố ý KHÔNG dùng scikit‐learn, vì thư viện đã làm trơn sẵn nên
sinh viên không nhìn thấy chỗ hỏng.
"""

from collections import Counter

# Kho tin nhắn tí hon, viết thẳng vào mã cho dễ theo dõi
kho = [
    ("trung thuong ngay", 1),
    ("khuyen mai giam gia", 1),
    ("trung thuong click", 1),
    ("hop nhom chieu mai", 0),
    ("bai tap nhom nop mai", 0),
    ("thay gui tai lieu", 0),
]

print("==================== BƯỚC 4 ====================")

tu_dien = sorted({t for cau, _ in kho for t in cau.split()})
V = len(tu_dien)
print(f"Tu dien co {V} tu:")

# In thành nhiều dòng, mỗi dòng 9 từ, để không tràn ra ngoài khung
for i in range(0, V, 9):
    print(" " + " ".join(tu_dien[i:i + 9]))
print()

# Đếm số lần mỗi từ xuất hiện trong từng lớp
dem = {0: Counter(), 1: Counter()}
tong = {0: 0, 1: 0}
for cau, nhan in kho:
    for t in cau.split():
        dem[nhan][t] += 1
        tong[nhan] += 1

print("So lan xuat hien cua tung tu")
print(f" {'tu':12s} {'rac':>5s} {'thuong':>7s}")
for t in tu_dien:
    print(f" {t:12s} {dem[1][t]:5d} {dem[0][t]:7d}")
print()


def xac_suat(tu, nhan, alpha):
    """P(tu | nhan) co lam tron Laplace voi he so alpha."""
    return (dem[nhan][tu] + alpha) / (tong[nhan] + alpha * V)

def diem(cau, nhan, alpha):
    """Tich cua xac suat tien nghiem va cac xac suat tu."""
    p = 0.5     # hai lop can bang nen tien nghiem bang nhau
    for t in cau.split():
        if t in tu_dien:
            p *= xac_suat(t, nhan, alpha)
    return p

cau_thu = "trung thuong bai tap"
print(f"Tin nhan thu: \"{cau_thu}\"")
print("Tu 'bai' chi xuat hien o lop thuong, chua bao gio o lop rac.")
print()

for alpha in [0.0, 1.0]:
    d_rac = diem(cau_thu, 1, alpha)
    d_thuong = diem(cau_thu, 0, alpha)
    ket = "rac" if d_rac > d_thuong else "thuong"
    print(f"alpha = {alpha}")
    print(f" diem lop rac = {d_rac:.10f}")
    print(f" diem lop thuong = {d_thuong:.10f}")
    print(f" ket luan = {ket}")
    print()

print("Khong lam tron, CA HAI diem deu bang 0 nen mo hinh te liet.")
print("Lop rac bang 0 vi tu 'bai' chua tung xuat hien o lop rac.")
print("Lop thuong bang 0 vi tu 'trung' chua tung xuat hien o lop thuong.")
print("Co lam tron, diem lop rac lon hon va mo hinh ket luan dung.")

# ==================== VẼ HÌNH THEO LAB03 ====================
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# File nằm trong code/ thì ảnh được lưu tại figures/ cùng cấp với code/.
thu_muc_hinh = Path(__file__).resolve().parent.parent / "figures"
thu_muc_hinh.mkdir(parents=True, exist_ok=True)

# Hình 3: xác suất từ 'bai' (chưa gặp ở lớp rác) theo alpha.
# Dùng số đếm trong kho nhỏ của file này nên thang xác suất có thể khác PDF.
cac_alpha = np.linspace(0, 2, 201)
cac_xac_suat = [xac_suat("bai", 1, a) for a in cac_alpha]
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(cac_alpha, cac_xac_suat, color="#3578a8", linewidth=2)
ax.scatter([0], [0], color="#d97732", zorder=3)
ax.annotate("alpha = 0: xác suất bằng 0", xy=(0, 0), xytext=(0.35, max(cac_xac_suat) * 0.25),
            arrowprops={"arrowstyle": "->"})
ax.set(title="Làm trơn kéo xác suất của từ chưa gặp ra khỏi số 0",
       xlabel="Tham số làm trơn alpha", ylabel='P("bai" | thư rác)')
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(thu_muc_hinh / "hinh_3_lam_tron_laplace.png", dpi=180)
plt.show()
