"""Bước 5: đường ROC, AUC, và so các mô hình đã học trong môn."""

import matplotlib.pyplot as plt


from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

lr = make_pipeline(StandardScaler(), LogisticRegression())
xs = lr.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]

# Mỗi ngưỡng cho một điểm của đường ROC
fpr, tpr, nguong = roc_curve(y_te, xs)
print("Duong ROC co", len(fpr), "diem. Vai diem dau:")
print("  bao dong nham   recall")
for a, b in list(zip(fpr, tpr))[:6]:
    print(f"      {a:.4f}      {b:.4f}")
print(f"AUC cua logistic = {roc_auc_score(y_te, xs):.4f}")
print()

# So các mô hình đã học, cùng tập học, cùng tập kiểm tra
mo_hinh = {
    "Logistic": lr,
    "Naive Bayes": GaussianNB(),
    "KNN, k = 15": make_pipeline(StandardScaler(),
                                KNeighborsClassifier(15)),
    "Cay sau 4": DecisionTreeClassifier(max_depth=4, random_state=0),
    "Rung 200 cay": RandomForestClassifier(200, random_state=0),
    "SVM RBF": make_pipeline(StandardScaler(),
                            SVC(probability=True, random_state=0)),
}
print("Mo hinh          AUC      recall    F1   (nguong 0.5)")
for ten, mh in mo_hinh.items():
    mh.fit(X_tr, y_tr)
    p = mh.predict_proba(X_te)[:, 1]
    du = (p >= 0.5).astype(int)
    print(f"  {ten:14s} {roc_auc_score(y_te, p):.4f}   "
            f"{recall_score(y_te, du):.4f}  {f1_score(y_te, du):.4f}")


fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"Logistic (AUC = {roc_auc_score(y_te, xs):.4f})")
ax.plot([0, 1], [0, 1], "--", color="gray")
for t in [0.5, 0.2]:
    tn, fp, fn, tp = confusion_matrix(y_te, xs >= t).ravel()
    a, b = fp / (fp + tn), tp / (tp + fn)
    ax.scatter(a, b)
    ax.annotate(f"Ngưỡng {t}", (a, b), xytext=(8, -10), textcoords="offset points")
ax.set(xlabel="Tỷ lệ dương tính giả", ylabel="Tỷ lệ dương tính thật", title="Đường ROC")
ax.legend(loc="lower right")
save_figure(fig, "hinh_4_roc_auc")
