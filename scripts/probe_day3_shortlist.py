"""Ngay 3 FIX — probe shortlist board-ID (CPU-only, khong GPU).

Muc tieu BUG-DAY3-02: board-ID + align ~45ms/anh (plan Phan 5). Do duoc:
  * ORB detect @scale 0.25 nf=1200 ~5ms
  * knnMatch BF tren 10 template ~120ms  <- nghen chinh
=> Khong the match du 10 template trong 45ms. Giai phap: shortlist bang
"van tay" anh nho (thumbnail chuan hoa sang) truoc, chi ORB-match top-K.

Probe do tren TOAN BO 98 anh bench (36 std + 36 cam + 16 train/val + 4 GOOD
+ 6 full-parity):
  1. Do chinh xac shortlist top-1/top-2/top-3 cua nhieu kieu van tay.
  2. Chi phi thoi gian shortlist.

Khong ghi file nao; chi in stdout.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

_spec = importlib.util.spec_from_file_location(
    "probe_b", str(PROJ / "scripts" / "probe_day3_branchB.py"))
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

from configs.resource_paths import LOCAL_TEMPLATES  # noqa: E402

TH = (48, 96)  # (h, w) thumbnail


def fingerprint(gray: np.ndarray, mode: str) -> np.ndarray:
    """Van tay anh nho bat bien voi gain/bias (va mot phan vignette)."""
    t = cv2.resize(gray, (TH[1], TH[0]), interpolation=cv2.INTER_AREA)
    f = t.astype(np.float32)
    if mode == "znorm":
        f = (f - f.mean()) / max(f.std(), 1e-6)
    elif mode == "grad":
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
        f = np.sqrt(gx * gx + gy * gy)
        f = (f - f.mean()) / max(f.std(), 1e-6)
    elif mode == "highpass":
        f = f - cv2.GaussianBlur(f, (0, 0), 6.0)
        f = (f - f.mean()) / max(f.std(), 1e-6)
    return f.ravel()


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))


def main() -> None:
    cache_tpl = {b: probe.read_gray(LOCAL_TEMPLATES / f"{b}.JPG")
                 for b in ("01", "04", "05", "06", "07", "08", "09", "10",
                           "11", "12")}
    items = probe.collect()  # 36 anh loi 01/04
    others = []
    import xml.etree.ElementTree as ET
    from configs.resource_paths import LEGACY_PKU_ROOT
    for ann_dir in sorted(p for p in (LEGACY_PKU_ROOT / "Annotations").iterdir()
                          if p.is_dir()):
        for xp in sorted(ann_dir.glob("*_annotation.xml")):
            r = ET.parse(xp).getroot()
            fn = r.find("filename")
            if fn is None or not fn.text:
                continue
            bid = Path(fn.text.strip()).stem.split("_")[0].zfill(2)
            if bid not in ("05", "06", "07", "08", "09", "10", "11", "12"):
                continue
            stem = xp.stem[:-len("_annotation")] if xp.stem.endswith("_annotation") else xp.stem
            for cand in (stem + ".jpg", stem + ".JPG"):
                q = LEGACY_PKU_ROOT / "images" / ann_dir.name / cand
                if q.exists():
                    others.append({"path": str(q), "board": bid})
                    break
    by_board: dict[str, list] = {}
    for it in others:
        by_board.setdefault(it["board"], []).append(it)
    sample = []
    for bid in sorted(by_board):
        sample.extend(by_board[bid][:2])

    cases = []  # (gray, true_board)
    for idx, it in enumerate(items):
        img = probe.read_bgr(Path(it["path"]))
        gt = [list(b) for _, b in it["gt"]]
        img_c, _ = probe.perturb(img, gt, 20261023 + 101 + idx)
        cases.append((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), it["board"]))
        cases.append((cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY), it["board"]))
    for it in sample:
        cases.append((probe.read_gray(Path(it["path"])), it["board"]))
    for bid in ("01", "04"):
        g = cache_tpl[bid]
        cases.append((g, bid))
        g_c, _ = probe.perturb(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), [], 20261023 + 999)
        cases.append((cv2.cvtColor(g_c, cv2.COLOR_BGR2GRAY), bid))

    print(f"[shortlist] cases={len(cases)}")
    for mode in ("znorm", "grad", "highpass"):
        t0 = time.perf_counter()
        tpl_fp = {b: fingerprint(g, mode) for b, g in cache_tpl.items()}
        build_ms = (time.perf_counter() - t0) * 1000
        top1 = top2 = top3 = 0
        times = []
        for gray, bid in cases:
            t0 = time.perf_counter()
            fp = fingerprint(gray, mode)
            scores = sorted(((ncc(fp, tpl_fp[b]), b) for b in tpl_fp), reverse=True)
            times.append((time.perf_counter() - t0) * 1000)
            names = [b for _, b in scores]
            top1 += names[0] == bid
            top2 += bid in names[:2]
            top3 += bid in names[:3]
        n = len(cases)
        print(f"  {mode:<9} build={build_ms:6.1f}ms  "
              f"top1={top1}/{n}={top1/n:.3f}  top2={top2}/{n}={top2/n:.3f}  "
              f"top3={top3}/{n}={top3/n:.3f}  per_img={np.median(times):.2f}ms")


if __name__ == "__main__":
    main()
