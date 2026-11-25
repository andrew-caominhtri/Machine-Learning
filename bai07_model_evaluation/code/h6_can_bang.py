"""Bước 6: kiểm định chéo với nhiều thước đo, và cân bằng lớp khi học."""

import matplotlib.pyplot as plt


from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
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
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]

# Chia không phân tầng: số bạn bỏ học trong tập kiểm tra dao động
print("Khong stratify, so ban bo hoc trong tap kiem tra:")
dem = []
for rs in range(10):
    _, _, _, y_te_thu = train_test_split(X, y, test_size=0.25,
                                        random_state=rs)
    dem.append(int(y_te_thu.sum()))
print(" ", dem)
print()

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

lr = make_pipeline(StandardScaler(), LogisticRegression())
print("Kiem dinh cheo 5 phan tren tap hoc:")
for tm in ["accuracy", "precision", "recall", "f1"]:
    d = cross_val_score(lr, X_tr, y_tr, cv=5, scoring=tm)
    print(f"  {tm:10s} {d.mean():.4f}")
print()

# Cân bằng lớp: phạt nặng hơn khi đoán sai lớp hiếm
cb = make_pipeline(StandardScaler(),
                    LogisticRegression(class_weight="balanced"))
print("Tap kiem tra   precision  recall    F1")
for ten, mh in [("Binh thuong", lr), ("Can bang lop", cb)]:
    du = mh.fit(X_tr, y_tr).predict(X_te)
    print(f"  {ten:12s}  {precision_score(y_te, du):.4f}"
            f"    {recall_score(y_te, du):.4f}  {f1_score(y_te, du):.4f}")


fig, ax = plt.subplots()
for i, (ten, mh) in enumerate([("Bình thường", lr), ("Cân bằng lớp", cb)]):
    du_doan = mh.predict(X_te)
    diem = [ham(y_te, du_doan) for ham in [precision_score, recall_score, f1_score]]
    ax.bar([j + (i - 0.5) * 0.36 for j in range(3)], diem, 0.36, label=ten)
ax.set(xticks=range(3), xticklabels=["Precision", "Recall", "F1"], ylim=(0, 1),
       title="Ảnh hưởng của cân bằng lớp")
ax.legend()
save_figure(fig, "can_bang_lop")
