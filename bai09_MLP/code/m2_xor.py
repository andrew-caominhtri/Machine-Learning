"""Bước 2: một nơ-ron không giải được XOR, một lớp ẩn thì được."""

import matplotlib.pyplot as plt
import numpy as np


from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

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


df = pd.read_csv(DATA_DIR / "xor.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

lr = LogisticRegression().fit(X_tr, y_tr)
print(f"Hoi quy logistic: {accuracy_score(y_te, lr.predict(X_te)):.4f}")
print()

# Dữ liệu nhỏ nên dùng solver="lbfgs", học nhanh và ổn định hơn
print("So no-ron an   so tham so   tap hoc   tap kiem tra")
for so in [1, 2, 4, 8]:
    mlp = MLPClassifier(hidden_layer_sizes=(so,), solver="lbfgs",
                        max_iter=5000, random_state=0)
    mlp.fit(X_tr, y_tr)
    # coefs_ là danh sách ma trận trọng số, intercepts_ là hệ số chặn
    so_ts = sum(w.size for w in mlp.coefs_)
    so_ts += sum(b.size for b in mlp.intercepts_)
    a_hoc = accuracy_score(y_tr, mlp.predict(X_tr))
    a_kt = accuracy_score(y_te, mlp.predict(X_te))
    print(f"     {so:2d}           {so_ts:3d}       {a_hoc:.4f}"
            f"    {a_kt:.4f}")


fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
bien_quyet_dinh(axes[0], lr, X, y, "Hồi quy logistic")
bien_quyet_dinh(axes[1], mlp, X, y, "MLP có 8 nơ-ron ẩn")
save_figure(fig, "hinh_2_ranh_gioi_xor")
