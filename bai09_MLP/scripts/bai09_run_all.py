"""Chạy các bước theo thứ tự trong PDF và lưu ảnh kết quả terminal."""

import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_DIR / "code"
OUTPUT_DIR = PROJECT_DIR / "outputs"
SCRIPTS = [
    "m1_lan_truyen_tien.py",
    "m2_xor.py",
    "m3_do_lon_mang.py",
    "m4_chuan_hoa.py",
    "m5_chu_so.py",
]


def save_output(index, script_name, output, return_code):
    lines = output.expandtabs(4).rstrip().splitlines()
    status = f"[LỖI {return_code}]" if return_code else "[HOÀN THÀNH]"
    header = f"{script_name}  {status}"
    columns = max(64, len(header), *(len(line) for line in lines))
    fig = plt.figure(figsize=(columns * 0.095 + 0.6, (len(lines) + 3) * 0.23),
                     facecolor="#17212e")
    fig.text(0.025, 0.96, "\n".join([header, "", *lines]), va="top",
             color="#edf3fa", fontfamily="DejaVu Sans Mono", fontsize=10,
             linespacing=1.4, parse_math=False)
    fig.savefig(OUTPUT_DIR / f"B{index}.png", dpi=150)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(variable, "1")

    for index, script_name in enumerate(SCRIPTS, start=1):
        script_path = CODE_DIR / script_name
        print(f"\n{'=' * 60}\nĐang chạy {script_name}\n{'=' * 60}", flush=True)
        if not script_path.is_file():
            print(f"[LỖI] Không tìm thấy {script_path}", file=sys.stderr)
            return 1

        result = subprocess.run(
            [sys.executable, "-u", str(script_path)],
            cwd=CODE_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        print(result.stdout, end="", flush=True)
        save_output(index, script_name, result.stdout, result.returncode)
        if result.returncode:
            print(f"[LỖI] {script_name} trả về mã {result.returncode}.", file=sys.stderr)
            return result.returncode

    print(f"\nHình vẽ được lưu ở {PROJECT_DIR / 'figures'}")
    print(f"Ảnh kết quả terminal được lưu ở {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
