# ‐*‐ coding: utf‐8 ‐*‐
"""Bước 5: tự áp ngưỡng và xem precision với recall đổi ra sao."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)
mo_hinh = LogisticRegression().fit(X_train, y_train)

# predict_proba trả về hai cột, cột 0 cho lớp 0 và cột 1 cho lớp 1
p = mo_hinh.predict_proba(X_test)[:, 1]

print("========== Bước 5 ==========")
print("Nguong So ban bi doan la qua Precision Recall")
for nguong in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    y_pred = (p >= nguong).astype(int)
    pre = precision_score(y_test, y_pred, zero_division=1)
    rec = recall_score(y_test, y_pred, zero_division=0)
    print(f" {nguong:.1f} {y_pred.sum():3d}"
          f" {pre:.4f} {rec:.4f}")
print()

print("Doc bang tren tu duoi len:")
print(" Nguong cang cao thi mo hinh cang kho tinh,")
print(" precision tang nhung recall giam. Nguong thap thi nguoc lai.")

# ===== VẼ HÌNH 5: xác suất từng sinh viên và ngưỡng 0.5 =====
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

thu_tu = np.argsort(p)
p_sap_xep = p[thu_tu]
y_sap_xep = y_test.to_numpy()[thu_tu]
vi_tri = np.arange(len(p_sap_xep))

plt.figure(figsize=(9, 6))
# Hai nhóm được vẽ riêng để màu thanh thể hiện nhãn thật
for nhan_that, ten_nhom in [(1, "Thật sự qua môn"), (0, "Thật sự rớt môn")]:
    idx = np.where(y_sap_xep == nhan_that)[0]
    plt.barh(idx, p_sap_xep[idx], label=ten_nhom)
plt.axvline(0.5, linestyle="--", linewidth=1.5, label="ngưỡng 0.5")
plt.xlim(0, 1)
plt.xlabel("Xác suất qua môn do mô hình đoán")
plt.ylabel("Sinh viên trong tập kiểm tra")
plt.title("Xác suất dự đoán và ngưỡng quyết định 0.5")
plt.legend(); plt.tight_layout()
plt.savefig(OUTPUT_DIR / "c5_hinh5_xac_suat_va_nguong_05.png", dpi=150)
plt.close()

# ===== VẼ HÌNH 6: precision và recall theo ngưỡng =====
nguong_ve = np.arange(0.05, 0.951, 0.01)
precision_ve, recall_ve = [], []
for t in nguong_ve:
    y_t = (p >= t).astype(int)
    precision_ve.append(precision_score(y_test, y_t, zero_division=1))
    recall_ve.append(recall_score(y_test, y_t, zero_division=0))

plt.figure(figsize=(9, 5))
plt.plot(nguong_ve, precision_ve, linewidth=2, label="Precision (độ chính xác dương)")
plt.plot(nguong_ve, recall_ve, "--", linewidth=2, label="Recall (độ bao phủ)")
plt.axvline(0.5, linestyle=":", linewidth=1.5, label="ngưỡng mặc định")
plt.xlim(0.05, 0.95); plt.ylim(0, 1.05)
plt.xlabel("Ngưỡng quyết định"); plt.ylabel("Giá trị")
plt.title("Hạ ngưỡng thì bao phủ tăng nhưng chính xác giảm")
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(OUTPUT_DIR / "c5_hinh6_precision_recall_theo_nguong.png", dpi=150)
plt.close()
