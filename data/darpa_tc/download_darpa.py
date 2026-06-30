#!/usr/bin/env python
"""
download_darpa.py
=================
Tải + giải nén DARPA TC E3 vào data/darpa_tc/<dataset>/ cho pretrain.

Tự động hoàn toàn:
  - groundtruth (UUID node độc hại) từ repo THREATRACE  -> LUÔN tải được.

Bán tự động (Google Drive cần file-ID):
  - JSON provenance E3 nằm trong 1 Drive FOLDER chung:
        https://drive.google.com/drive/folders/1QlbUFWAGq3Hpl8wVdzOdIoZLFxkII4EK
    Lấy file-ID từng file (mở folder -> chuột phải file -> "Get link" -> phần sau
    /d/ hoặc id=) rồi điền vào DRIVE_IDS bên dưới, HOẶC tải .tar.gz thủ công đặt vào
    data/darpa_tc/<dataset>/ và chạy script (nó sẽ tự giải nén).

Cách dùng:
    python data/darpa_tc/download_darpa.py cadets            # tải groundtruth + (nếu có ID) JSON, giải nén
    python data/darpa_tc/download_darpa.py cadets --gt-only   # chỉ groundtruth
    python data/darpa_tc/download_darpa.py cadets --id <fid> --name ta1-cadets-e3-official-2.json.tar.gz
    python data/darpa_tc/download_darpa.py cadets --folder    # gdown CẢ folder E3 (RẤT NẶNG, mọi dataset)
"""

import os
import sys
import glob
import gzip
import shutil
import tarfile
import argparse
import subprocess
import urllib.request

DARPA_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder Drive chứa toàn bộ E3 (từ DARPA TC README-E3)
E3_DRIVE_FOLDER = "1QlbUFWAGq3Hpl8wVdzOdIoZLFxkII4EK"

# file-ID Google Drive cho từng dataset — ĐIỀN sau khi lấy từ folder Drive ở trên.
# Để trống -> script chỉ tải groundtruth + giải nén .tar.gz có sẵn tại chỗ.
DRIVE_IDS = {
    "cadets": {
        # "ta1-cadets-e3-official.json.tar.gz":   "<gdrive_id>",
        # "ta1-cadets-e3-official-1.json.tar.gz": "<gdrive_id>",
        # "ta1-cadets-e3-official-2.json.tar.gz": "<gdrive_id>",
    },
    "theia": {},
    "trace": {},
}

GT_URL = "https://raw.githubusercontent.com/threaTrace-detector/threaTrace/master/groundtruth/{ds}.txt"


def _ensure_gdown():
    try:
        import gdown  # noqa
        return True
    except ImportError:
        print("[setup] Cài gdown ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
            return True
        except Exception as e:
            print(f"[setup] Không cài được gdown: {e}")
            return False


def fetch_groundtruth(ds: str, out_dir: str):
    url = GT_URL.format(ds=ds)
    dst = os.path.join(out_dir, f"{ds}.txt")
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"[gt] đã có {dst}")
        return
    try:
        print(f"[gt] tải {url}")
        urllib.request.urlretrieve(url, dst)
        n = sum(1 for _ in open(dst, encoding="utf-8", errors="ignore"))
        print(f"[gt] -> {dst} ({n} dòng)")
    except Exception as e:
        print(f"[gt] LỖI tải groundtruth: {e}\n     Tải tay: {url} -> {dst}")


def gdown_file(file_id: str, dst: str):
    import gdown
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"[drive] đã có {os.path.basename(dst)}")
        return dst
    print(f"[drive] tải id={file_id} -> {os.path.basename(dst)}")
    gdown.download(id=file_id, output=dst, quiet=False)
    return dst


def gdown_folder(out_dir: str):
    import gdown
    print(f"[drive] gdown CẢ folder E3 (rất nặng) -> {out_dir}")
    gdown.download_folder(id=E3_DRIVE_FOLDER, output=out_dir, quiet=False, use_cookies=False)


def extract_all(out_dir: str):
    """Giải nén mọi .tar.gz / .gz trong out_dir thành .json (idempotent)."""
    for fp in glob.glob(os.path.join(out_dir, "*.tar.gz")) + glob.glob(os.path.join(out_dir, "*.tgz")):
        print(f"[extract] tar {os.path.basename(fp)}")
        with tarfile.open(fp, "r:gz") as t:
            t.extractall(out_dir)
    for fp in glob.glob(os.path.join(out_dir, "*.json.gz")):
        out = fp[:-3]
        if os.path.exists(out):
            continue
        print(f"[extract] gz {os.path.basename(fp)}")
        with gzip.open(fp, "rb") as fi, open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo)


def verify(out_dir: str, ds: str):
    jsons = [f for f in glob.glob(os.path.join(out_dir, "*.json*"))
             if not f.endswith((".gz", ".tar.gz", ".tgz", ".pkl", ".txt"))]
    gt = os.path.join(out_dir, f"{ds}.txt")
    print("\n=== VERIFY ===")
    print(f"  JSON files : {len(jsons)}")
    for j in jsons[:5]:
        print(f"    - {os.path.basename(j)}")
    if jsons:
        # đếm nhanh event records trong file đầu (sample 200k dòng)
        n_evt = 0
        with open(jsons[0], encoding="utf-8", errors="ignore") as fh:
            for i, line in enumerate(fh):
                if i > 200000:
                    break
                if '"Event"' in line or "EVENT_" in line:
                    n_evt += 1
        print(f"  ~event records (200k dòng đầu của {os.path.basename(jsons[0])}): {n_evt}")
    print(f"  groundtruth: {'OK ' + gt if os.path.exists(gt) else 'THIẾU'}")
    if jsons and os.path.exists(gt):
        print("  => SẴN SÀNG. Chạy: python Benign_Agent/pretrain_on_generated.py")
    else:
        print("  => CHƯA đủ. Xem hướng dẫn Drive trong docstring/README.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="cadets", choices=["cadets", "theia", "trace"])
    ap.add_argument("--gt-only", action="store_true", help="chỉ tải groundtruth")
    ap.add_argument("--folder", action="store_true", help="gdown CẢ folder E3 (rất nặng)")
    ap.add_argument("--id", help="file-ID Drive 1 file")
    ap.add_argument("--name", help="tên file đích cho --id (vd ...json.tar.gz)")
    args = ap.parse_args()

    ds = args.dataset
    out_dir = os.path.join(DARPA_DIR, ds)
    os.makedirs(out_dir, exist_ok=True)

    fetch_groundtruth(ds, out_dir)
    if args.gt_only:
        verify(out_dir, ds)
        return

    # JSON provenance
    if args.id and args.name:
        if _ensure_gdown():
            gdown_file(args.id, os.path.join(out_dir, args.name))
    elif args.folder:
        if _ensure_gdown():
            gdown_folder(out_dir)
    else:
        ids = DRIVE_IDS.get(ds, {})
        if ids and _ensure_gdown():
            for name, fid in ids.items():
                if fid and not fid.startswith("<"):
                    gdown_file(fid, os.path.join(out_dir, name))
        else:
            print(f"\n[drive] Chưa có file-ID cho '{ds}' trong DRIVE_IDS, và không local .tar.gz.")
            print(f"        Lấy ID từ folder: https://drive.google.com/drive/folders/{E3_DRIVE_FOLDER}")
            print(f"        rồi: điền DRIVE_IDS, hoặc --id <fid> --name <file>, hoặc đặt .tar.gz vào {out_dir}")

    extract_all(out_dir)
    verify(out_dir, ds)


if __name__ == "__main__":
    main()
