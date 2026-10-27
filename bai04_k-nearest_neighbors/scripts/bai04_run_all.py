from pathlib import Path
import subprocess
import sys

# Thư mục gốc của project:
# project/
# ├── code/
# ├── data/
# ├── outputs/
# └── scripts/
PROJECT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_DIR / "code"

# Các file cần chạy theo thứ tự
SCRIPTS = [
    "e1_khoang_cach.py",
    "e2_chuan_hoa.py",
    "e3_chon_k.py",
    "e4_so_sanh.py",
    "e5_knn_hoi_quy.py"
]

print("=" * 60)
print("BẮT ĐẦU CHẠY TOÀN BỘ BÀI LAB")
print("=" * 60)

for script_name in SCRIPTS:
    script_path = CODE_DIR / script_name

    if not script_path.exists():
        print(f"\n[KHÔNG TÌM THẤY] {script_path}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Đang chạy: {script_name}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=CODE_DIR
    )

    if result.returncode != 0:
        print(f"\n[LỖI] {script_name} chạy không thành công.")
        print("Đã dừng quá trình chạy toàn bộ.")
        sys.exit(result.returncode)

    print(f"[HOÀN THÀNH] {script_name}")

print("\n" + "=" * 60)
print("ĐÃ CHẠY XONG TOÀN BỘ BÀI LAB")
print("=" * 60)
print(f"Kết quả/biểu đồ được lưu tại: {PROJECT_DIR / 'figures'}")
