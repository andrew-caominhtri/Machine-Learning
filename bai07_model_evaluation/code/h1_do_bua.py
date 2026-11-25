"""Bước 1: thấy accuracy đánh lừa trên dữ liệu lệch lớp.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import matplotlib.pyplot as plt
import numpy as np


from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

plt.switch_backend("Agg")


def save_figure(fig, name):
    figure_dir = Path(__file__).resolve().parents[1] / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


df = pd.read_csv(DATA_DIR / "sinh_vien_bo_hoc.csv")
print("Kich thuoc bang:", df.shape)
print(df.head())
print()
print("So sinh vien moi lop:")
print(df["bo_hoc"].value_counts().to_string())
print(f"Ty le bo hoc: {df['bo_hoc'].mean():.3f}")
print()

X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
# stratify=y giữ đúng tỷ lệ bỏ học ở cả tập học lẫn tập kiểm tra
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
print(f"Tap kiem tra: {len(y_te)} ban, {y_te.sum()} ban bo hoc")
print()

# Mô hình lười: ai cũng đoán là lớp đông nhất
luoi = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
du_luoi = luoi.predict(X_te)

lr = make_pipeline(StandardScaler(), LogisticRegression())
lr.fit(X_tr, y_tr)
du_lr = lr.predict(X_te)

for ten, du in [("Mo hinh luoi", du_luoi), ("Hoi quy logistic", du_lr)]:
    acc = accuracy_score(y_te, du)
    tim_ra = ((du == 1) & (y_te == 1)).sum()
    print(f"{ten:17s} accuracy {acc:.4f}, "
            f"tim ra {tim_ra} tren {y_te.sum()} ban bo hoc")


fig, ax = plt.subplots()
vi_tri = np.arange(2)
du_doan = [du_luoi, du_lr]
ax.bar(vi_tri - 0.18, [accuracy_score(y_te, d) for d in du_doan], 0.36, label="Accuracy")
ax.bar(vi_tri + 0.18, [recall_score(y_te, d) for d in du_doan], 0.36, label="Recall bỏ học")
ax.set(xticks=vi_tri, xticklabels=["Mô hình lười", "Logistic"], ylim=(0, 1.05))
ax.legend()
save_figure(fig, "hinh_1_accuracy_va_recall")
