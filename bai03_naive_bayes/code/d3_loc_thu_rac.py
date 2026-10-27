# ‐*‐ coding: utf‐8 ‐*‐
"""Bước 3: huấn luyện bộ lọc thư rác và chấm điểm nó."""

import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("../data/tin_nhan.csv")
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df["noi_dung"])
y = df["la_rac"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print("==================== BƯỚC 3 ====================")

print("So tin de hoc :", X_train.shape[0])
print("So tin de kiem tra:", X_test.shape[0])
print()

mo_hinh = MultinomialNB(alpha=1.0)
mo_hinh.fit(X_train, y_train)
y_pred = mo_hinh.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("Ma tran nham lan")
print(f" TN = {tn:2d} thu thuong duoc giu dung")
print(f" FP = {fp:2d} thu thuong bi xoa nham")
print(f" FN = {fn:2d} thu rac lot luoi")
print(f" TP = {tp:2d} thu rac bi chan dung")
print()

print(f"Accuracy = {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision = {precision_score(y_test, y_pred):.4f}")
print(f"Recall = {recall_score(y_test, y_pred):.4f}")
print()

# Thử bộ lọc trên vài tin nhắn tự gõ
tin_moi = [
    "trung thuong khuyen mai nhanh tay click ngay",
    "hop nhom bao cao chieu mai phong 305",
    "ban gui giup em tai lieu bai tap nhe"
]
X_moi = vec.transform(tin_moi)
p_rac = mo_hinh.predict_proba(X_moi)[:, 1]

print("Thu tren ba tin nhan tu go:")
for tin, p in zip(tin_moi, p_rac):
    nhan = "RAC" if p >= 0.5 else "thuong"
    print(f" [{nhan:6s}] p = {p:.4f} <‐ {tin}")

# ==================== VẼ HÌNH THEO LAB03 ====================
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# File nằm trong code/ thì ảnh được lưu tại figures/ cùng cấp với code/.
thu_muc_hinh = Path(__file__).resolve().parent.parent / "figures"
thu_muc_hinh.mkdir(parents=True, exist_ok=True)

# Hình 2: dùng kết quả dự đoán đã tính, không nhập cứng các ô trong PDF.
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, labels=[0, 1], display_labels=["Thường", "Rác"],
    cmap="Blues", colorbar=False, ax=ax)
ax.set(title="Ma trận nhầm lẫn của bộ lọc", xlabel="Nhãn dự đoán", ylabel="Nhãn thật")
fig.tight_layout()
fig.savefig(thu_muc_hinh / "hinh_2_ma_tran_nham_lan.png", dpi=180)

# Hình 6 (mục 6.4): so sánh MultinomialNB và LogisticRegression trên tin nhắn.
# Cố định tập kiểm tra; lấy trung bình 20 lần chọn tập học con có phân tầng.
# PDF không nêu đầy đủ cách lặp thí nghiệm nên số đo có thể khác hình mẫu.
so_mau = sorted({n for n in [10, 20, 40, 60, 100, 150, 200, X_train.shape[0]]
                 if 2 <= n <= X_train.shape[0]})
ket_qua = {"Naive Bayes": [], "Hồi quy logistic": []}
for n in so_mau:
    if n == X_train.shape[0]:
        cac_tap_con = [np.arange(n)]
    else:
        chia = StratifiedShuffleSplit(n_splits=20, train_size=n, random_state=17)
        cac_tap_con = [idx for idx, _ in chia.split(X_train, y_train)]
    diem_so = {ten: [] for ten in ket_qua}
    for idx in cac_tap_con:
        for ten, model in [("Naive Bayes", MultinomialNB(alpha=1.0)),
                           ("Hồi quy logistic", LogisticRegression(max_iter=1000))]:
            model.fit(X_train[idx], y_train.iloc[idx])
            diem_so[ten].append(accuracy_score(y_test, model.predict(X_test)))
    for ten in ket_qua:
        ket_qua[ten].append(np.mean(diem_so[ten]))
fig, ax = plt.subplots(figsize=(9, 5))
for ten, gia_tri in ket_qua.items():
    ax.plot(so_mau, gia_tri, marker="o", label=ten)
ax.set(title="So sánh hai mô hình khi thay đổi số tin nhắn học",
       xlabel="Số tin nhắn dùng để huấn luyện", ylabel="Accuracy trung bình trên tập kiểm tra")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(thu_muc_hinh / "hinh_6_accuracy_theo_so_mau.png", dpi=180)
plt.show()
