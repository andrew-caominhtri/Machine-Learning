# -*- coding: utf-8 -*-
"""Bước 2: đo hậu quả của việc quên chuẩn hoá trên tập kiểm tra."""

import pandas as pd
from pathlib import Path

# Đường dẫn theo vị trí file, chạy được từ thư mục gốc hoặc code/.
THU_MUC_GOC = Path(__file__).resolve().parent.parent
THU_MUC_HINH = THU_MUC_GOC / "figures"
THU_MUC_HINH.mkdir(parents=True, exist_ok=True)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(THU_MUC_GOC / "data" / "sinh_vien_mo_rong.csv")
y = df["qua_mon"]

X3 = df[["gio_on", "diem_giua_ky", "chi_tieu"]]
X_tr, X_te, y_tr, y_te = train_test_split(
    X3, y, test_size=0.25, random_state=17, stratify=y
)


def cham(mo_hinh, cot):
    mo_hinh.fit(X_tr[cot], y_tr)
    return accuracy_score(y_te, mo_hinh.predict(X_te[cot]))


def knn_chuan():
    """KNN co chuan hoa, gop lam mot khoi bang pipeline."""
    return make_pipeline(StandardScaler(), KNeighborsClassifier(7))


ba_cot = ["gio_on", "diem_giua_ky", "chi_tieu"]
hai_cot = ["gio_on", "diem_giua_ky"]

a = cham(KNeighborsClassifier(n_neighbors=7), ba_cot)
b = cham(knn_chuan(), ba_cot)
c = cham(knn_chuan(), hai_cot)

print(f"Doan bua ca lop qua              : {y_te.mean():.4f}")
print(f"Ba cot, khong chuan hoa      : {a:.4f}")
print(f"Ba cot, co chuan hoa         : {b:.4f}")
print(f"Bo cot chi_tieu, co chuan hoa: {c:.4f}")

# ==================== VẼ HÌNH THEO LAB04 ====================
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

# Hình 2: các chiều cao lấy trực tiếp từ kết quả chấm ở trên.
fig, ax = plt.subplots(figsize=(10, 5))
nhan_cot = ["Ba cột\nkhông chuẩn hoá", "Ba cột\ncó chuẩn hoá",
            "Bỏ cột chi tiêu\ncó chuẩn hoá"]
cot = ax.bar(nhan_cot, [a, b, c], width=0.55, color=["#db6c67", "#4779a7", "#409976"])
ax.bar_label(cot, fmt="%.4f", padding=5)
ax.axhline(y_te.mean(), color="#9a7100", linestyle="--",
           label=f"Đoán cả lớp qua: {y_te.mean():.4f}")
ax.set(ylim=(0, 1.12), ylabel="Accuracy trên tập kiểm tra",
       title="Ảnh hưởng của chuẩn hoá và lựa chọn đặc trưng")
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(THU_MUC_HINH / "hinh_2_chuan_hoa.png", dpi=180)
plt.show()
