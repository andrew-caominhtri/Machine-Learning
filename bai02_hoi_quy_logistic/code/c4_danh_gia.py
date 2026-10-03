# ‐*‐ coding: utf‐8 ‐*‐
"""Bước 4: chia dữ liệu rồi chấm điểm mô hình bằng bốn thước đo."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

# stratify = y  # Giữ nguyên tỷ lệ qua và rớt ở cả hai phần sau khi chia
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)

print("========== Bước 4 ==========")
print("So sinh vien de hoc :", len(X_train))
print("So sinh vien de kiem tra:", len(X_test))
print()

mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)
y_pred = mo_hinh.predict(X_test)

# Thứ tự bốn ô do scikit‐learn quy định là TN, FP, FN, TP
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("Ma tran nham lan")
print(f" TN = {tn:2d} doan rot, that su rot")
print(f" FP = {fp:2d} doan qua, that ra rot")
print(f" FN = {fn:2d} doan rot, that ra qua")
print(f" TP = {tp:2d} doan qua, that su qua")
print()

print(f"Accuracy = {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision = {precision_score(y_test, y_pred):.4f}")
print(f"Recall = {recall_score(y_test, y_pred):.4f}")
print(f"F1 = {f1_score(y_test, y_pred):.4f}")
print()

# Tự tính lại bằng tay để đối chiếu với thư viện
print("Tu tinh lai bang cong thuc:")
n = len(y_test)
print(f" Accuracy = ({tp} + {tn}) / {n} = {(tp + tn) / n:.4f}")
print(f" Precision = {tp} / ({tp} + {fp}) = {tp / (tp + fp):.4f}")
print(f" Recall = {tp} / ({tp} + {fn}) = {tp / (tp + fn):.4f}")

# ===== VẼ HÌNH 4: ma trận nhầm lẫn =====
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ma_tran = np.array([[tn, fp], [fn, tp]])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(ma_tran)
ax.set_xticks([0, 1], labels=["Đoán rớt", "Đoán qua"])
ax.set_yticks([0, 1], labels=["Thật: rớt", "Thật: qua"])
nhan_o = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{ma_tran[i, j]}\n{nhan_o[i][j]}",
                ha="center", va="center", fontsize=14)
ax.set_title("Ma trận nhầm lẫn trên tập kiểm tra")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "c4_hinh4_ma_tran_nham_lan.png", dpi=150)
plt.close()
