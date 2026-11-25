"""Bước 3: mạng to thì học vẹt, điều chuẩn bằng alpha thì đỡ."""

import matplotlib.pyplot as plt
import numpy as np


from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
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


df = pd.read_csv(DATA_DIR / "hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)


def mang(lop_an, alpha):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=lop_an, alpha=alpha,
                        solver="lbfgs", max_iter=20000, random_state=0),
    )


print("Ba mang              tap hoc   tap kiem tra")
cac_mang = []
for ten, lop_an, alpha in [("(2,), alpha 1e-4", (2,), 1e-4),
                            ("(100, 100), 1e-5", (100, 100), 1e-5),
                            ("(100, 100), 1.0", (100, 100), 1.0)]:
    m = mang(lop_an, alpha).fit(X_tr, y_tr)
    cac_mang.append((ten, m))
    print(f"  {ten:18s} {accuracy_score(y_tr, m.predict(X_tr)):.4f}"
            f"    {accuracy_score(y_te, m.predict(X_te)):.4f}")
print()

# Chọn alpha bằng kiểm định chéo, chỉ trên tập học
print("alpha      kiem dinh cheo")
diem = {}
for alpha in [1e-5, 1e-3, 0.1, 1.0, 3.0, 10.0]:
    diem[alpha] = cross_val_score(mang((100, 100), alpha),
                                X_tr, y_tr, cv=5).mean()
    print(f"  {alpha:<8}  {diem[alpha]:.4f}")
a_tot = max(diem, key=diem.get)
m = mang((100, 100), a_tot).fit(X_tr, y_tr)
print(f"alpha tot nhat = {a_tot}, tap kiem tra "
        f"{accuracy_score(y_te, m.predict(X_te)):.4f}")


fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
for ax, (ten, mh) in zip(axes, cac_mang):
    bien_quyet_dinh(ax, mh, X, y, ten)
save_figure(fig, "hinh_3_kich_thuoc_va_dieu_chuan")
