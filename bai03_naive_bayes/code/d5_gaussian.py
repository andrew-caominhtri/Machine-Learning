# ‐*‐ coding: utf‐8 ‐*‐
"""Bước 5: Naive Bayes cho số liệu liên tục, đặt cạnh hồi quy logistic."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("../data/sinh_vien.csv")
X = df[["gio_on", "diem_giua_ky"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print("==================== BƯỚC 5 ====================")

# Hai mô hình, cùng dữ liệu, cùng cách chia, nên so sánh mới công bằng
for ten, mo_hinh in [("Naive Bayes ", GaussianNB()),
                      ("Hoi quy logistic", LogisticRegression())]:
    mo_hinh.fit(X_train, y_train)
    acc = accuracy_score(y_test, mo_hinh.predict(X_test))
    print(f"{ten}: accuracy = {acc:.4f}")
    print()

gnb = GaussianNB().fit(X_train, y_train)

# theta_ la trung binh, var_ la phuong sai, moi dong mot lop
print("Nhung gi Naive Bayes hoc duoc")
print(f" {'':16s} {'gio_on':>10s} {'diem_giua_ky':>14s}")
for i, ten_lop in enumerate(["Lop 0 (rot) ", "Lop 1 (qua) "]):
    tb = gnb.theta_[i]
    dl = gnb.var_[i] ** 0.5
    print(f" {ten_lop} trung binh {tb[0]:10.2f} {tb[1]:14.2f}")
    print(f" {'':12s} do lech {dl[0]:10.2f} {dl[1]:14.2f}")
print()

ty_le = ", ".join(f"{float(v):.4f}" for v in gnb.class_prior_)
print(f"Ty le tung lop trong tap hoc: {ty_le}")
print()

# Dự đoán cho vài sinh viên giả định
ban_moi = pd.DataFrame({"gio_on": [5, 13, 20, 28],
                        "diem_giua_ky": [4.0, 6.0, 7.0, 9.0]})
p = gnb.predict_proba(ban_moi)[:, 1]
print("Du doan cho bon ban moi:")
print(f" {'gio on':>8s} {'diem GK':>9s} {'P(qua mon)':>12s}")
for (_, r), pi in zip(ban_moi.iterrows(), p):
    print(f" {r.gio_on:8.0f} {r.diem_giua_ky:9.1f} {pi:12.4f}")

# ==================== VẼ HÌNH THEO LAB03 ====================
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# File nằm trong code/ thì ảnh được lưu tại figures/ cùng cấp với code/.
thu_muc_hinh = Path(__file__).resolve().parent.parent / "figures"
thu_muc_hinh.mkdir(parents=True, exist_ok=True)

# Hình 4: phân phối số giờ ôn của từng lớp, ước lượng từ tập học.
fig, ax = plt.subplots(figsize=(10, 5))
xx = np.linspace(df.gio_on.min() - 5, df.gio_on.max() + 5, 500)
for i, (ten, mau) in enumerate([("Rớt môn", "#d97732"), ("Qua môn", "#3578a8")]):
    mu, sigma = gnb.theta_[i, 0], np.sqrt(gnb.var_[i, 0])
    mat_do = np.exp(-0.5 * ((xx - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ax.plot(xx, mat_do, color=mau,
            label=f"{ten}: trung bình {mu:.1f}, độ lệch {sigma:.1f}")
    gio = X_train.loc[y_train == gnb.classes_[i], "gio_on"]
    ax.plot(gio, np.full(len(gio), -0.0015 * (i + 1)), "|", color=mau, alpha=0.5)
ax.set(title="Naive Bayes mô tả từng lớp bằng một đường chuông",
       xlabel="Số giờ ôn tập", ylabel="Mật độ")
ax.legend()
fig.tight_layout()
fig.savefig(thu_muc_hinh / "hinh_4_phan_phoi_gaussian.png", dpi=180)

# Hình 5: xác suất qua môn và biên P(qua môn) = 0.5 trên cùng tập học.
xx, yy = np.meshgrid(
    np.linspace(max(0, X.gio_on.min() - 1), X.gio_on.max() + 1, 250),
    np.linspace(max(0, X.diem_giua_ky.min() - 0.5), max(10, X.diem_giua_ky.max()), 250))
luoi = pd.DataFrame({"gio_on": xx.ravel(), "diem_giua_ky": yy.ravel()})
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
for ax, (ten, model) in zip(axes, [("Hồi quy logistic", LogisticRegression().fit(X_train, y_train)),
                                  ("Naive Bayes", gnb)]):
    cot_qua = list(model.classes_).index(1)
    zz = model.predict_proba(luoi)[:, cot_qua].reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[0, 0.5, 1], colors=["#fce6d3", "#dcebf7"], alpha=0.6)
    ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=1.5)
    for nhan, ten_lop, mau in [(0, "Rớt môn", "#d97732"), (1, "Qua môn", "#3578a8")]:
        nhom = df.loc[y == nhan]
        ax.scatter(nhom.gio_on, nhom.diem_giua_ky, s=22, c=mau, label=ten_lop, alpha=0.7)
    ax.set(title=ten, xlabel="Số giờ ôn tập")
    ax.legend()
axes[0].set_ylabel("Điểm giữa kỳ")
fig.suptitle("Biên quyết định của hai mô hình")
fig.tight_layout()
fig.savefig(thu_muc_hinh / "hinh_5_bien_quyet_dinh.png", dpi=180)
plt.show()
