"""Bước 2: lề mềm, tham số C, và vì sao SVM phải chuẩn hoá."""

import matplotlib.pyplot as plt


from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
    if hasattr(model[-1], "support_"):
        ax.contour(a, b, score, levels=[-1, 1], colors="gray", linestyles="--")
    ax.scatter(*diem.T, c=y, cmap="coolwarm", s=18, edgecolors="white", linewidths=0.3)
    ax.set(title=title, xlabel="x1", ylabel="x2")


df = pd.read_csv(DATA_DIR / "sinh_vien.csv")
X = df[["gio_on", "diem_giua_ky"]]
y = df["qua_mon"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print("Tap hoc co", len(X_tr), "ban")
print("    C    so vecto HT   do rong le   tap hoc   tap kiem tra")
for C in [0.01, 0.1, 1, 10, 100]:
    mh = make_pipeline(StandardScaler(),
                        SVC(kernel="linear", C=C))
    mh.fit(X_tr, y_tr)
    svm = mh[-1]                             # bước cuối của pipeline là SVM
    le = 2 / np.linalg.norm(svm.coef_[0])
    a_hoc = accuracy_score(y_tr, mh.predict(X_tr))
    a_kt = accuracy_score(y_te, mh.predict(X_te))
    print(f"  {C:6}   {len(svm.support_):6d}       {le:6.2f}"
            f"      {a_hoc:.4f}    {a_kt:.4f}")
print()

# Bộ sinh viên mở rộng có thêm cột chi tiêu, tính bằng nghìn đồng
df3 = pd.read_csv(DATA_DIR / "sinh_vien_mo_rong.csv")
X3 = df3[["gio_on", "diem_giua_ky", "chi_tieu"]]
y3 = df3["qua_mon"]
X3_tr, X3_te, y3_tr, y3_te = train_test_split(
    X3, y3, test_size=0.25, random_state=17, stratify=y3
)
tho = SVC().fit(X3_tr, y3_tr)
chuan = make_pipeline(StandardScaler(), SVC()).fit(X3_tr, y3_tr)
print("Ba cot, kernel RBF mac dinh")
print(f"  Doan bua ca lop qua : {y3_te.mean():.4f}")
print(f"  Khong chuan hoa     : "
        f"{accuracy_score(y3_te, tho.predict(X3_te)):.4f}")
print(f"  Co chuan hoa        : "
        f"{accuracy_score(y3_te, chuan.predict(X3_te)):.4f}")


fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
for ax, C in zip(axes, [0.01, 100]):
    mh = make_pipeline(StandardScaler(), SVC(kernel="linear", C=C)).fit(X_tr, y_tr)
    bien_quyet_dinh(ax, mh, X_tr, y_tr, f"C = {C}")
    ax.set(xlabel="Giờ ôn", ylabel="Điểm giữa kỳ")
    ax.scatter(*X_tr.iloc[mh[-1].support_].to_numpy().T, s=70,
               facecolors="none", edgecolors="orange")
save_figure(fig, "hinh_3_tham_so_c")
