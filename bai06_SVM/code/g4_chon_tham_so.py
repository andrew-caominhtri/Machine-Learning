"""Bước 4: ảnh hưởng của gamma, rồi dò lưới C và gamma."""

import matplotlib.pyplot as plt
import numpy as np


from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
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


df = pd.read_csv(DATA_DIR / "hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print("gamma   tap hoc   tap kiem tra   so vecto HT")
for g in [0.1, 1, 10, 100]:
    mh = SVC(kernel="rbf", gamma=g).fit(X_tr, y_tr)
    a_hoc = accuracy_score(y_tr, mh.predict(X_tr))
    a_kt = accuracy_score(y_te, mh.predict(X_te))
    print(f"  {g:5}  {a_hoc:.4f}    {a_kt:.4f}         "
            f"{len(mh.support_)}")
print()

# Dò lưới: thử mọi cặp bằng kiểm định chéo 5 phần, chỉ trên tập học
luoi = {"C": [0.1, 1, 10, 100], "gamma": [0.01, 0.1, 1, 10]}
tim = GridSearchCV(SVC(kernel="rbf"), luoi, cv=5)
tim.fit(X_tr, y_tr)

bang = pd.DataFrame(tim.cv_results_["params"])
bang["diem"] = tim.cv_results_["mean_test_score"]
print("Diem kiem dinh cheo, hang la C, cot la gamma")
print(bang.pivot(index="C", columns="gamma",
                values="diem").round(4).to_string())
print()

print("Cap tot nhat:", tim.best_params_)
print(f"Diem kiem dinh cheo: {tim.best_score_:.4f}")
# GridSearchCV đã tự học lại với cặp tốt nhất trên toàn tập học
print(f"Accuracy tren tap kiem tra: {tim.score(X_te, y_te):.4f}")


fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
for ax, gamma in zip(axes, [0.1, 1, 100]):
    mh = SVC(gamma=gamma).fit(X_tr, y_tr)
    bien_quyet_dinh(ax, mh, X, y, f"gamma = {gamma}")
save_figure(fig, "hinh_6_gamma")

diem_cv = bang.pivot(index="C", columns="gamma", values="diem")
fig, ax = plt.subplots()
anh = ax.imshow(diem_cv, cmap="Blues", aspect="auto")
ax.set(xticks=range(len(diem_cv.columns)), xticklabels=diem_cv.columns,
       yticks=range(len(diem_cv.index)), yticklabels=diem_cv.index,
       xlabel="gamma", ylabel="C", title="Accuracy kiểm định chéo")
for (i, j), diem in np.ndenumerate(diem_cv.to_numpy()):
    tot_nhat = (diem_cv.index[i] == tim.best_params_["C"]
                and diem_cv.columns[j] == tim.best_params_["gamma"])
    ax.text(j, i, f"{diem:.4f}", ha="center", va="center",
            color="white" if anh.norm(diem) > 0.55 else "black",
            bbox=dict(edgecolor="orange", facecolor="none") if tot_nhat else None)
fig.colorbar(anh, ax=ax)
save_figure(fig, "hinh_7_do_luoi")
