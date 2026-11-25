"""Bước 3: nâng chiều bằng tay, rồi để kernel làm việc đó."""

import matplotlib.pyplot as plt


from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

plt.switch_backend("Agg")


def save_figure(fig, name):
    figure_dir = Path(__file__).resolve().parents[1] / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def bien_quyet_dinh(ax, model, X, y, title):
    diem = np.asarray(X)
    a, b = np.meshgrid(
        np.linspace(diem[:, 0].min() - 0.2, diem[:, 0].max() + 0.2, 180),
        np.linspace(diem[:, 1].min() - 0.2, diem[:, 1].max() + 0.2, 180),
    )
    luoi = np.c_[a.ravel(), b.ravel()]
    if hasattr(X, "columns"):
        luoi = pd.DataFrame(luoi, columns=X.columns)
    score = (model.decision_function(luoi) if hasattr(model, "decision_function")
             else model.predict_proba(luoi)[:, 1] - 0.5).reshape(a.shape)
    ax.contourf(a, b, score >= 0, levels=[-0.5, 0.5, 1.5], cmap="coolwarm", alpha=0.18)
    ax.contour(a, b, score, levels=[0], colors="black", linewidths=1.2)
    if hasattr(model, "decision_function") and hasattr(model, "support_"):
        ax.contour(a, b, score, levels=[-1, 1], colors="gray", linestyles="--")
    ax.scatter(*diem.T, c=y, cmap="coolwarm", s=18, edgecolors="white", linewidths=0.3)
    ax.set(title=title, xlabel="x1", ylabel="x2")


# Phần 1: dữ liệu một chiều, lớp 1 nằm giữa, lớp 0 ở hai đầu
x = np.array([-4.0, -3.4, -2.8, -2.2, -1.2, -0.7, -0.2,
                0.3, 0.8, 1.3, 2.3, 2.9, 3.5, 4.0])
y = (np.abs(x) < 1.8).astype(int)

# Thử mọi ngưỡng: đoán lớp 1 khi x nhỏ hơn hoặc bằng ngưỡng
tot = max(((x <= t).astype(int) == y).mean() for t in x)
print(f"Mot nguong tren truc x, tot nhat dung {tot:.4f}")

# Thêm cột x binh phuong rồi dùng SVM tuyến tính
X_moi = np.c_[x, x ** 2]
svm = SVC(kernel="linear", C=1000).fit(X_moi, y)
print(f"Them cot x^2, SVM tuyen tinh dung {svm.score(X_moi, y):.4f}")
print()

# Phần 2: hai vầng trăng, để kernel tự nâng chiều
df = pd.read_csv(DATA_DIR / "hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

mo_hinh = {
    "SVM kernel tuyen tinh": SVC(kernel="linear"),
    "SVM kernel da thuc bac 3": SVC(kernel="poly", degree=3),
    "SVM kernel RBF": SVC(kernel="rbf"),
    "Hoi quy logistic": LogisticRegression(),
    "KNN, k = 13": KNeighborsClassifier(n_neighbors=13),
}
print("Hai vang trang, accuracy tren tap kiem tra")
for ten, mh in mo_hinh.items():
    mh.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, mh.predict(X_te))
    print(f"  {ten:26s} {acc:.4f}")
print()

rbf = mo_hinh["SVM kernel RBF"]
print("So vecto ho tro cua kernel RBF:", len(rbf.support_),
        "tren", len(X_tr), "diem hoc")


fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(x, np.zeros_like(x), c=(np.abs(x) < 1.8), cmap="coolwarm")
axes[0].set(xlabel="x", yticks=[], title="Dữ liệu một chiều")
bien_quyet_dinh(axes[1], svm, X_moi, (np.abs(x) < 1.8), "Thêm chiều x²")
axes[1].set(xlabel="x", ylabel="x²")
save_figure(fig, "hinh_4_nang_chieu")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
for ax, ten in zip(axes, ["SVM kernel tuyen tinh", "SVM kernel RBF"]):
    bien_quyet_dinh(ax, mo_hinh[ten], X, y, ten)
save_figure(fig, "hinh_5_kernel_tuyen_tinh_va_rbf")
