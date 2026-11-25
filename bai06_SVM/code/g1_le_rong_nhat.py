"""Bước 1: tìm đường có lề rộng nhất và các vectơ hỗ trợ.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

from pathlib import Path

import matplotlib.pyplot as plt


import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

plt.switch_backend("Agg")


def save_figure(fig, name):
    figure_dir = Path(__file__).resolve().parents[1] / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# 24 điểm, hai cụm tách hẳn nhau
X, y = make_blobs(n_samples=24, centers=[[2, 2], [6, 6]],
                cluster_std=0.9, random_state=0)
y = 1 - y                # lớp 1 là cụm dưới bên trái, giống hình trên slide

# C rất lớn nghĩa là không cho điểm nào nằm trong lề
svm = SVC(kernel="linear", C=1000)
svm.fit(X, y)

w = svm.coef_[0]
b = svm.intercept_[0]
print(f"w1 = {w[0]:.3f}, w2 = {w[1]:.3f}, b = {b:.3f}")
do_dai_w = np.sqrt(w[0] ** 2 + w[1] ** 2)
print(f"Do rong cua le = 2 / {do_dai_w:.4f} = {2 / do_dai_w:.4f}")
print()

# Các vectơ hỗ trợ nằm đúng trên hai mép lề, nơi f(x) bằng 1 hoặc -1
print("So vecto ho tro:", len(svm.support_))
for i in svm.support_:
    f = svm.decision_function(X[i:i + 1])[0]
    print(f"  diem {i:2d}: x1 = {X[i, 0]:.2f}, x2 = {X[i, 1]:.2f},"
            f" lop {y[i]}, f(x) = {f:+.3f}")
print()

# Những điểm khác nằm ngoài lề, nên |f(x)| lớn hơn 1
f_tat_ca = svm.decision_function(X)
ngoai = np.setdiff1d(np.arange(len(X)), svm.support_)
nho_nhat = np.abs(f_tat_ca[ngoai]).min()
print(f"|f(x)| nho nhat cua cac diem khac: {nho_nhat:.3f}")
print()

# Học lại chỉ với các vectơ hỗ trợ
chi_sv = SVC(kernel="linear", C=1000)
chi_sv.fit(X[svm.support_], y[svm.support_])
w2 = chi_sv.coef_[0]
print("Hoc lai chi voi vecto ho tro:")
b2 = chi_sv.intercept_[0]
print(f"  w1 = {w2[0]:.3f}, w2 = {w2[1]:.3f}, b = {b2:.3f}")
print()

# Dự đoán cho hai điểm mới
moi = np.array([[3.0, 3.0], [5.0, 4.5]])
for diem, f, lop in zip(moi, svm.decision_function(moi),
                        svm.predict(moi)):
    print(f"Diem ({diem[0]}, {diem[1]}): f(x) = {f:+.3f}, doan lop {lop}")


fig, ax = plt.subplots()
mien = np.linspace(X[:, 0].min() - 0.4, X[:, 0].max() + 0.4, 200)
ax.scatter(*X.T, c=y, cmap="coolwarm", edgecolors="white")
for muc in [-1, 0, 1]:
    ax.plot(mien, (muc - w[0] * mien - b) / w[1],
            color="black", linestyle="-" if muc == 0 else "--")
ax.scatter(*X[svm.support_].T, s=150, facecolors="none", edgecolors="orange")
ax.set(xlabel="x1", ylabel="x2", ylim=(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5),
       title="Ranh giới, lề và vectơ hỗ trợ")
save_figure(fig, "hinh_2_le_va_vecto_ho_tro")

fig, ax = plt.subplots()
ax.scatter(*X.T, c=y, cmap="coolwarm", edgecolors="white")
bien_duoi = -(X[y == 1] @ w).min()
bien_tren = -(X[y == 0] @ w).max()
for ten, he_so, mau in [("A", b - 0.55 * (b - bien_duoi), "#228833"),
                        ("B", b + 0.55 * (bien_tren - b), "#aa3377"),
                        ("C", b, "#222222")]:
    ax.plot(mien, (-w[0] * mien - he_so) / w[1], label=ten, color=mau)
ax.set(xlabel="x1", ylabel="x2", ylim=(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5),
       title="Ba đường thẳng phân tách hai lớp")
ax.legend()
save_figure(fig, "hinh_1_ba_duong_phan_tach")
