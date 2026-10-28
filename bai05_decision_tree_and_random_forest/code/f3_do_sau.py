# -*- coding: utf-8 -*-
"""Bước 3: để cây mọc tự do, rồi chọn độ sâu bằng kiểm định chéo."""


from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

import matplotlib
matplotlib.use("Agg")  # Chỉ lưu ảnh, không mở cửa sổ.
import matplotlib.pyplot as plt

df = pd.read_csv("../data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Không đặt giới hạn nào, cây chia tới khi mọi lá đều thuần
tu_do = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
print("Cay moc tu do")
print(f" do sau {tu_do.get_depth()}, so la {tu_do.get_n_leaves()}")
a_hoc = accuracy_score(y_tr, tu_do.predict(X_tr))
a_kt = accuracy_score(y_te, tu_do.predict(X_te))
print(f" tap hoc {a_hoc:.4f}, tap kiem tra {a_kt:.4f}")
print()

# Chọn độ sâu chỉ dựa trên tập học
print("Do sau   tap hoc   kiem dinh cheo")
diem = {}
for d in range(1, 16):
    mh = DecisionTreeClassifier(max_depth=d, random_state=0)
    diem[d] = cross_val_score(mh, X_tr, y_tr, cv=5).mean()
    hoc = accuracy_score(y_tr, mh.fit(X_tr, y_tr).predict(X_tr))
    print(f" {d:2d}      {hoc:.4f}    {diem[d]:.4f}")
d_tot = max(diem, key=diem.get)
print()

# Chọn xong mới mở tập kiểm tra, và chỉ mở một lần
tot = DecisionTreeClassifier(max_depth=d_tot, random_state=0)
tot.fit(X_tr, y_tr)
print(f"Do sau tot nhat = {d_tot}, cay co {tot.get_n_leaves()} la")
acc = accuracy_score(y_te, tot.predict(X_te))
print(f"Accuracy tren tap kiem tra = {acc:.4f}")


# ==================== VẼ VÀ CHỈ LƯU HÌNH ====================

THU_MUC_HINH = Path("../figures")
THU_MUC_HINH.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

# Hình 3: so sánh cây tự do và cây sâu tối đa 3 trên hai cột.

# Biên 2D: fit RIÊNG mô hình trên hai cột, không coi là biên của mô hình 4 cột.
cot_2d = ["thu_nhap", "ty_le_no"]
x1, x2 = np.meshgrid(
    np.linspace(X.thu_nhap.min() - 1, X.thu_nhap.max() + 1, 230),
    np.linspace(max(0, X.ty_le_no.min() - 0.05), X.ty_le_no.max() + 0.05, 180))
luoi = pd.DataFrame({"thu_nhap": x1.ravel(), "ty_le_no": x2.ravel()})

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
for ax, (ten, model) in zip(axes, [
        ("Cây mọc tự do", DecisionTreeClassifier(random_state=0)),
        ("Cây sâu tối đa 3", DecisionTreeClassifier(max_depth=3, random_state=0))]):

    model.fit(X_tr[cot_2d], y_tr)
    zz = model.predict(luoi).reshape(x1.shape)
    ax.contourf(x1, x2, zz, levels=[-0.5, 0.5, 1.5], colors=["#fae4d1", "#d9eafa"], alpha=0.7)
    ax.contour(x1, x2, zz, levels=[0.5], colors="#475569", linewidths=0.8)
    for nhan, ten_lop, mau, dau in [(0, "Trễ hạn", "#d97732", "^"),
                                    (1, "Đúng hạn", "#3678a8", "o")]:
        nhom = y == nhan
        ax.scatter(X.loc[nhom, "thu_nhap"], X.loc[nhom, "ty_le_no"],
                   c=mau, marker=dau, s=13, alpha=0.6, label=ten_lop)
    ax.set(title=ten, xlabel="Thu nhập (triệu đồng)")

axes[0].set_ylabel("Tỷ lệ nợ")
axes[-1].legend(fontsize=9)
fig.suptitle("Biên quyết định trên hai đặc trưng")
fig.tight_layout()
fig.savefig(THU_MUC_HINH / "hinh_3_bien_theo_do_sau.png", dpi=180)
plt.close(fig)

# Hình 4: điểm kiểm định chéo lấy từ vòng chọn độ sâu, chỉ dùng tập học.
do_sau = list(diem)
diem_hoc = []
for d in do_sau:
    model = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_tr, y_tr)
    diem_hoc.append(accuracy_score(y_tr, model.predict(X_tr)))
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(do_sau, diem_hoc, marker="o", label="Tập học", color="#3678a8")
ax.plot(do_sau, [diem[d] for d in do_sau], marker="s", label="Kiểm định chéo 5 phần", color="#d97732")
ax.axvline(d_tot, color="#439775", linestyle="--", label=f"Độ sâu chọn bằng CV: {d_tot}")
ax.set(title="Ảnh hưởng của độ sâu tối đa", xlabel="Độ sâu tối đa", ylabel="Accuracy", ylim=(0, 1.03))
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
fig.savefig(THU_MUC_HINH / "hinh_4_accuracy_theo_do_sau.png", dpi=180)
plt.close(fig)
print("Da luu hinh 3 va 4 tai:", THU_MUC_HINH)
