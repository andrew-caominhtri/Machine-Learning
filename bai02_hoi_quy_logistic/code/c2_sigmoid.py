# ‐*‐ coding: utf‐8 ‐*‐
"""Bước 2: làm quen hàm sigmoid bằng tay trước khi gọi thư viện."""

import numpy as np

def sigmoid(z):
    """Ep mot so thuc bat ky ve khoang tu 0 toi 1."""
    return 1 / (1 + np.exp(-z))

print("========== Bước2 ==========")
print("Bang gia tri cua ham sigmoid")
print(" z sigmoid(z)")
for z in [-6, -4, -2, -1, 0, 1, 2, 4, 6]:
    print(f"{z:3d} {sigmoid(z):.4f}")
print()

# Hai tính chất cần tự kiểm chứng
print("Kiem tra sigmoid(‐z) = 1 ‐ sigmoid(z):")
for z in [1.0, 2.5, 3.7]:
    trai = sigmoid(-z)
    phai = 1 - sigmoid(z)
    print(f" z = {z}: {trai:.6f} va {phai:.6f}")
print()

# Thử với một mô hình giả định w = 0.4 và b = -5
w, b = 0.4, -5.0
print(f"Mo hinh gia dinh: w = {w}, b = {b}")
print(" So gio on z = w*x + b Xac suat qua mon")
for gio in [5, 10, 12.5, 15, 20, 25]:
    z = w * gio + b
    print(f" {gio:5.1f} {z:6.2f} {sigmoid(z):.4f}")

# ===== VẼ HÌNH 2: đồ thị hàm sigmoid =====
from pathlib import Path
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

z_ve = np.linspace(-8, 8, 400)
plt.figure(figsize=(8, 5))
plt.plot(z_ve, sigmoid(z_ve), linewidth=2)
plt.axhline(0.5, linestyle=":", linewidth=1)
plt.axvline(0, linestyle=":", linewidth=1)
plt.scatter([0], [0.5], zorder=3)
plt.annotate("σ(0) = 0.5", xy=(0, 0.5), xytext=(1.0, 0.42),
             arrowprops=dict(arrowstyle="->"))
plt.text(-7.2, 0.08, "tiến tới 0")
plt.text(5.2, 0.92, "tiến tới 1")
plt.xlim(-8, 8); plt.ylim(0, 1.05)
plt.xlabel("z"); plt.ylabel("σ(z)")
plt.title("Đồ thị hàm sigmoid")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(OUTPUT_DIR / "c2_hinh2_sigmoid.png", dpi=150)
plt.close()
