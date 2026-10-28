# -*- coding: utf-8 -*-
"""Bước 2: học một cây nhỏ năm lá trên dữ liệu khách vay rồi đọc nó."""


from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

import matplotlib
matplotlib.use("Agg")  # Chỉ lưu ảnh, không mở cửa sổ.
import matplotlib.pyplot as plt

df = pd.read_csv("../data/khach_vay.csv")
print("Kich thuoc bang:", df.shape)
print(df.head())
print()
print("Ty le khach tra dung han:", round(df["tra_dung_han"].mean(), 2))
print()

X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Giới hạn cây chỉ được có 5 lá để còn đọc được bằng mắt
cay = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0)
cay.fit(X_tr, y_tr)

print(f"Do sau {cay.get_depth()}, so la {cay.get_n_leaves()}")
print(export_text(cay, feature_names=list(X.columns),
                  show_weights=True, decimals=3))

a_hoc = accuracy_score(y_tr, cay.predict(X_tr))
a_kt = accuracy_score(y_te, cay.predict(X_te))
print(f"Accuracy tap hoc      : {a_hoc:.4f}")
print(f"Accuracy tap kiem tra : {a_kt:.4f}")
print()

# Hai khách mới, đi theo cây để biết mỗi người rơi vào lá nào
moi = pd.DataFrame({
    "thu_nhap": [20.0, 10.0],
    "ty_le_no": [0.4, 0.3],
    "co_viec_on_dinh": [1, 0],
    "so_nguoi_phu_thuoc": [2, 1],
})
du_doan = cay.predict(moi)
xac_suat = cay.predict_proba(moi)[:, 1]
for i in range(len(moi)):
    ket = "dung han" if du_doan[i] == 1 else "tre han"
    print(f"Khach {i + 1}: {ket}, xac suat dung han {xac_suat[i]:.2f}")


# ==================== VẼ VÀ CHỈ LƯU HÌNH ====================

THU_MUC_HINH = Path("../figures")
THU_MUC_HINH.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

# Hình 2: vẽ chính cây đã học ở trên, không dựng lại bằng ngưỡng nhập tay.
fig, ax = plt.subplots(figsize=(16, 8))
plot_tree(cay, feature_names=list(X.columns),
          class_names=["Trễ hạn" if c == 0 else "Đúng hạn" for c in cay.classes_],
          filled=True, rounded=True, precision=3, fontsize=9, ax=ax)
ax.set_title("Cây quyết định tối đa năm lá", fontsize=15)
fig.tight_layout()
fig.savefig(THU_MUC_HINH / "hinh_2_cay_nam_la.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Da luu:", THU_MUC_HINH / "hinh_2_cay_nam_la.png")
