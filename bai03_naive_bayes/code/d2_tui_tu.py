# ‐*‐ coding: utf‐8 ‐*‐
"""Bước 2: đọc kho tin nhắn và biến văn bản thành bảng số."""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("../data/tin_nhan.csv")

print("==================== BƯỚC 2 ====================")

print("Kich thuoc bang:", df.shape)
print()
print("So tin theo nhan (1 la rac, 0 la thuong):")
print(df["la_rac"].value_counts())
print()
print("Ty le tin rac:", round(df["la_rac"].mean(), 4))
print()

print("Ba tin nhan dau tien:")
for i in range(3):
    print(f" [{df.la_rac[i]}] {df.noi_dung[i]}")
print()

# token_pattern mặc định của scikit‐learn bỏ mất từ một ký tự,
# nên ta chỉ định tách từ bằng khoảng trắng cho đơn giản.
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df["noi_dung"])

print("Kich thuoc bang dem tu:", X.shape)
print("So tu trong tu dien:", len(vec.vocabulary_))
print()

# Đếm tổng số lần xuất hiện của từng từ, tách theo hai lớp
dem = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
dem["la_rac"] = df["la_rac"]
tong = dem.groupby("la_rac").sum().T
tong.columns = ["thuong", "rac"]

print("Muoi tu xuat hien nhieu nhat trong tin rac:")
print(tong.sort_values("rac", ascending=False).head(10))
print()
print("Muoi tu xuat hien nhieu nhat trong tin thuong:")
print(tong.sort_values("thuong", ascending=False).head(10))

# ==================== VẼ HÌNH THEO LAB03 ====================
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# File nằm trong code/ thì ảnh được lưu tại figures/ cùng cấp với code/.
thu_muc_hinh = Path(__file__).resolve().parent.parent / "figures"
thu_muc_hinh.mkdir(parents=True, exist_ok=True)

# Hình 1: log P(từ | rác) - log P(từ | thường), làm trơn alpha = 1.
# Dùng toàn bộ kho để mô tả từ vựng ở mục 3 của tài liệu.
xac_suat_tu = (tong + 1).div(tong.sum(axis=0) + len(tong), axis=1)
chenh_lech = np.log(xac_suat_tu["rac"]) - np.log(xac_suat_tu["thuong"])
sap_xep = chenh_lech.sort_values()
chon = pd.concat([sap_xep.head(8), sap_xep.tail(8)]).loc[lambda s: ~s.index.duplicated()]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(chon.index, chon.values,
        color=["#d97732" if v > 0 else "#3578a8" for v in chon])
ax.axvline(0, color="black", linewidth=1)
ax.set(title="Từ nào tố cáo thư rác mạnh nhất",
       xlabel="Nghiêng về thư thường ← Hiệu log xác suất → Nghiêng về thư rác")
ax.grid(axis="x", alpha=0.2)
fig.tight_layout()
fig.savefig(thu_muc_hinh / "hinh_1_chenh_lech_tu.png", dpi=180)
plt.show()
