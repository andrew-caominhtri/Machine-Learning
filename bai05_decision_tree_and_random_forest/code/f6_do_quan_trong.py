# -*- coding: utf-8 -*-
"""Bước 6: đọc độ quan trọng của đặc trưng, và thấy giới hạn của nó."""


from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")  # Chỉ lưu ảnh, không mở cửa sổ.
import matplotlib.pyplot as plt

df = pd.read_csv("../data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

rung = RandomForestClassifier(n_estimators=200, random_state=0)
rung.fit(X_tr, y_tr)
qt = pd.Series(rung.feature_importances_, index=X.columns)
print("Do quan trong, bon cot that")
print(qt.sort_values(ascending=False).round(3).to_string())
print("Tong:", round(qt.sum(), 3))
print()

# Thêm một cột toàn số ngẫu nhiên, chẳng liên quan gì tới việc trả nợ
rng = np.random.default_rng(0)
X_tr_nn = X_tr.copy()
X_tr_nn["so_ngau_nhien"] = rng.random(len(X_tr))
rung_nn = RandomForestClassifier(n_estimators=200, random_state=0)
rung_nn.fit(X_tr_nn, y_tr)
qt_nn = pd.Series(rung_nn.feature_importances_, index=X_tr_nn.columns)
print("Do quan trong, them mot cot ngau nhien")
print(qt_nn.sort_values(ascending=False).round(3).to_string())


# ==================== VẼ VÀ CHỈ LƯU HÌNH ====================

THU_MUC_HINH = Path("../figures")
THU_MUC_HINH.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

# Hình 7: chỉ bốn đặc trưng thật, đúng hình minh hoạ trong PDF.
# Cột ngẫu nhiên vẫn được in kết quả ở phần tính toán, không thêm hình ngoài đề.
nhan_cot = {"thu_nhap": "Thu nhập", "ty_le_no": "Tỷ lệ nợ",
            "co_viec_on_dinh": "Có việc ổn định", "so_nguoi_phu_thuoc": "Số người phụ thuộc"}
sap_xep = qt.sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
cac_cot = ax.barh([nhan_cot.get(c, c) for c in sap_xep.index], sap_xep.values, color="#3678a8")
ax.bar_label(cac_cot, fmt="%.3f", padding=4)
ax.set(title="Độ quan trọng của đặc trưng theo rừng 200 cây", xlabel="Độ quan trọng (giảm độ hỗn tạp)",
       xlim=(0, max(sap_xep.max() * 1.2, 0.1)))
ax.grid(axis="x", alpha=0.2)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(THU_MUC_HINH / "hinh_7_do_quan_trong.png", dpi=180)
plt.close(fig)
print("Da luu:", THU_MUC_HINH / "hinh_7_do_quan_trong.png")
