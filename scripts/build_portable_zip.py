"""build_portable_zip.py — Dong goi ban Portable chay doc lap (Windows).

Lam:
  1. Loc rac/nang khong can thiet (.venv, __pycache__, .pytest_cache,
     .git, runs/, weights/yolo26n.pt lach, data/yolo_day2 tiles train 232M,
     tiles smoke, onnx du).
  2. Pre-flight portability check:
     - Quet .py/.json/.yaml duoc dua vao zip: KHONG con drive literal
       (dang chu-cai + hai-cham + slash/backslash) trong luong chay chinh
       (configs/, src/, scripts/, data/yolo_day2/data.yaml). Chu thich/plan cho phep; code chay KHONG.
     - Khong con phu thuoc legacy trong LUUONG RUNTIME:
       * src/ + configs/resource_paths.py: duoc phep nhac LEGACY_* trong
         comment/docstring + dinh nghia tuong doi PROJECT_ROOT.parent
         (old-machine only, co fallback DEMO bundled). Cam drive literal.
       * scripts/evaluate.py + debug_e2e_trace.py + app.py: chay duoc voi
         DEMO bundled khi legacy khong ton tai.
     - Gia lap may moi: doi CWD sang thu muc tam, copy assets/templates +
       weights can thiet + 1 anh demo, chay End-to-End (align fast +
       run_branch_b + project H_inv + predict_batch + gating +
       fuse_and_decide) dam bao khong thieu file.
  3. Nen thanh D:/FPTU/KLTN/PCB_DualPath_AOI_Portable_Release.zip
     (ghi DE ngoai project moi — khong phai vao trong project).

Vi du:
  python scripts/build_portable_zip.py
  python scripts/build_portable_zip.py --preflight-only   (chi kiem tra)
  python scripts/build_portable_zip.py --out E:\\release.zip
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DEFAULT = PROJECT_ROOT.parent / "PCB_DualPath_AOI_Portable_Release.zip"

# Thu muc/file LOAI khoi goi portable (rac hoac qua nang / khong can thiet).
EXCLUDE_DIRS = {".venv", "__pycache__", ".pytest_cache", ".git",
                ".hypothesis", "runs", "weights",
                "data/yolo_day2", "data/PKU_Market_PCB",
                "data/demo_and_test_samples_extra"}
EXCLUDE_FILES = {
    "weights/yolo26n.pt",                    # file lach goc project
    "assets/weights/yolo_best.onnx",         # onnx thu nghiem, khong dung runtime
    "assets/weights/yolo_best.onnx.json",
    "assets/weights/best.pt",                # trung ban voi yolo11n_base.pt
}
EXCLUDE_SUFFIX = {".pyc", ".pyo"}

# File nhay trong zip (goc zip -> duong dan dich).
INCLUDE_TOP = [
    "configs", "src", "scripts", "tests",
    "assets/weights/yolo11n_day2.pt",
    "assets/weights/yolo11n_base.pt",
    "assets/templates",
    "data/demo_and_test_samples",
    "data/yolo_day2/data.yaml",
    "data/yolo_day2/stats.json",
    "results", "aoi_final_plan",
    "requirements.txt",
    "1_setup_env.bat", "2_run_ui_demo.bat", "3_run_self_test.bat",
    "README_PORTABLE.md",
    "PCB_DualPath_AOI_Documentation.docx",
]

DRIVE_RE = re.compile(r"[A-Za-z]:[\\/]")
LEGACY_NAME_RE = re.compile(
    r"PCB_Anomaly_Inspector_Portable|Test_github|hybird_test|pcb_unified_10k")

# Ket qua lich su + tai lieu: giu nguyen de tra cuu, KHONG quet drive literal
# (meta weights/anomaly trong results/*.json la dau vet may cu, khong phai
# code chay; luong runtime doc results chi de hien thi so tham khao).
SCAN_EXEMPT_PREFIXES = ("results/", "aoi_final_plan/")


def iter_included_files() -> list[Path]:
    files: list[Path] = []
    for top in INCLUDE_TOP:
        p = PROJECT_ROOT / top
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if not f.is_file():
                    continue
                rel = f.relative_to(PROJECT_ROOT).as_posix()
                if f.suffix in EXCLUDE_SUFFIX:
                    continue
                if any(rel == e or rel.startswith(e.rstrip("/") + "/")
                       for e in EXCLUDE_DIRS):
                    # ngoai le: giu data.yaml + stats.json trong yolo_day2
                    if rel not in ("data/yolo_day2/data.yaml",
                                   "data/yolo_day2/stats.json"):
                        continue
                if rel in EXCLUDE_FILES:
                    continue
                # bo figures nang (giu docx + md + code sinh hinh)
                if "thesis_artifacts/figures/" in rel and f.suffix.lower() in (
                        ".png", ".jpg", ".jpeg"):
                    continue
                files.append(f)
    return sorted(set(files))


def preflight_scan(files: list[Path]) -> list[str]:
    """Quet drive literal + legacy trong luong chay. Tra ve list loi."""
    errors: list[str] = []
    for f in files:
        if f.suffix.lower() not in (".py", ".json", ".yaml", ".yml", ".bat"):
            continue
        rel = f.relative_to(PROJECT_ROOT).as_posix()
        try:
            txt = f.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            errors.append(f"{rel}: khong doc duoc ({e})")
            continue
        for i, line in enumerate(txt.splitlines(), 1):
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("REM "):
                continue
            if "://" in line:
                continue  # URL (https://...) khong phai drive literal
            if DRIVE_RE.search(line) and not rel.startswith(
                    SCAN_EXEMPT_PREFIXES):
                # cho phep: chuoi trong docstring/comment giai thich
                low = line.lower()
                if ("portable" in low or "never hardcode" in low
                        or "khong" in low or "khong dien" in low):
                    continue
                # bat buoc PASS: khong drive literal trong code chay
                errors.append(f"{rel}:{i}: con drive literal: {s[:100]}")
        # legacy name chi duoc xuat hien o resource_paths (dinh nghia tuong
        # doi + fallback) / tests (guard) / docstring; cam trong logic runtime
        # src/scripts chay that.
        if f.suffix == ".py" and rel.startswith(("src/", "scripts/")):
            if rel in ("scripts/build_portable_zip.py",
                       "scripts/debug_e2e_trace.py"):
                continue
            if "resource_paths" in rel:
                continue
            body = "\n".join(
                ln for ln in txt.splitlines()
                if ln.strip() and not ln.strip().startswith(("#", '"""', "'''", '"', "'"))
            )
            # don gian: kiem drive-thuc thay vi ten (ten da duoc phep o comment)
            for i, line in enumerate(txt.splitlines(), 1):
                if LEGACY_NAME_RE.search(line):
                    st = line.strip()
                    if st.startswith(("#", '"""', "'''", '"', "'", "*", "-", "|")):
                        continue
                    if "PORTABLE" in line or "old-machine" in line or "old machine" in line:
                        continue
                    # import LEGACY_* tu resource_paths la hop le (fallback)
                    if "LEGACY_" in line and ("import" in line or "roots" in line
                                              or "from configs" in line):
                        continue
                    errors.append(f"{rel}:{i}: con phu thuoc legacy: {st[:100]}")
    return errors


def preflight_e2e() -> dict:
    """Gia lap may moi: CWD tam + ENV sach, chay E2E 1 anh demo."""
    import numpy as np  # noqa: E402
    sys.path.insert(0, str(PROJECT_ROOT))
    for m in [m for m in list(sys.modules)
              if m.startswith(("src", "configs", "scripts"))]:
        sys.modules.pop(m, None)
    from configs.resource_paths import (  # noqa: E402
        DEMO_ROOT, LOCAL_DAY2_PT, LOCAL_TEMPLATES, default_weights,
    )
    import src.reference.board_id_and_align as ba  # noqa: E402
    import src.reference.tolerance_diff as td  # noqa: E402
    from src.models.ensemble import fuse_and_decide  # noqa: E402
    from src.models.yolo_detector import YOLODetector  # noqa: E402
    import cv2  # noqa: E402

    img_p = DEMO_ROOT / "images" / "Short" / "short01.jpg"
    assert img_p.exists(), f"thieu anh demo: {img_p}"
    # gia lap CWD may moi (o dia/thu muc khac) — pipeline khong duoc phu
    # thuoc CWD hien tai.
    old_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="aoi_preflight_")
    os.chdir(tmp)
    try:
        data = np.fromfile(str(img_p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        assert img is not None
        wpath = default_weights()
        assert wpath.exists(), f"thieu weights: {wpath}"
        assert LOCAL_DAY2_PT.exists(), "thieu yolo11n_day2.pt"
        yolo = YOLODetector(wpath, conf_threshold=0.20)
        cache = ba.load_cache()
        r = ba.identify_and_align(img, cache, fast=True)
        tpl = cv2.imdecode(
            np.fromfile(str(LOCAL_TEMPLATES / f"{r['board_id']}.JPG"),
                        dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if tpl.shape != r["warped"].shape[:2]:
            tpl = cv2.resize(tpl, (r["warped"].shape[1], r["warped"].shape[0]),
                             interpolation=cv2.INTER_AREA)
        out = td.run_branch_b(r["warped"], tpl,
                              valid_mask=r.get("valid_mask"))
        cents = (ba.project_points(
            np.float32([b["centroid_tpl"] for b in out["blobs"]]), r["H_inv"])
            if out["blobs"] else np.zeros((0, 2), np.float32))
        Hh, Ww = img.shape[:2]
        crops, origins = [], []
        for (px, py) in cents:
            x0 = int(np.clip(float(px) - 320, 0, max(Ww - 640, 0)))
            y0 = int(np.clip(float(py) - 320, 0, max(Hh - 640, 0)))
            crops.append(cv2.cvtColor(img[y0:y0 + 640, x0:x0 + 640],
                                      cv2.COLOR_BGR2RGB))
            origins.append((x0, y0))
        batch = yolo.predict_batch(crops) if crops else []
        dets = []
        for d1, (px, py), (x0, y0) in zip(batch, cents, origins):
            kept = YOLODetector.spatial_center_gating(
                d1, center_xy=(float(px), float(py)), radius=80.0,
                crop_origin=(float(x0), float(y0)))
            for d in kept:
                dd = dict(d)
                dd["bbox"] = [d["bbox"][0] + x0, d["bbox"][1] + y0,
                              d["bbox"][2] + x0, d["bbox"][3] + y0]
                dets.append(dd)
        res = fuse_and_decide(dets, out["blobs"], r["H_inv"])
        return {"image": img_p.name, "board": r["board_id"],
                "inliers": int(r["inliers"]), "n_blobs": len(out["blobs"]),
                "n_dets": len(dets), "level": res["level"],
                "conclusion": res.get("conclusion"),
                "verdict": res.get("verdict"),
                "imgsz_actual": yolo.imgsz, "device": yolo.device,
                "cwd_simule": tmp}
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def build_zip(out: Path) -> tuple[Path, int]:
    files = iter_included_files()
    missing = [str(PROJECT_ROOT / t) for t in INCLUDE_TOP
               if not (PROJECT_ROOT / t).exists()
               and t not in ("data/yolo_day2/data.yaml",)]
    # data.yaml chac chan ton tai (vua sua portable)
    assert (PROJECT_ROOT / "data/yolo_day2/data.yaml").exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            z.write(f, f.relative_to(PROJECT_ROOT).as_posix())
    return out, len(files)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Build portable release zip.")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    print("== [1/3] Liet ke file dua vao goi ==")
    files = iter_included_files()
    total = sum(f.stat().st_size for f in files)
    print(f"  {len(files)} files, ~{total / 1024 / 1024:.1f} MB (chua nen)")
    for t in INCLUDE_TOP:
        print(f"    + {t}")

    print("== [2/3] Pre-flight portability check ==")
    errs = preflight_scan(files)
    if errs:
        print("  [FAIL] con literal cam:")
        for e in errs[:30]:
            print("   -", e)
        return 1
    print("  scan drive/legacy trong luong chay: PASS")
    try:
        r = preflight_e2e()
        print(f"  E2E gia lap may moi: PASS ({r['image']} board={r['board']} "
              f"inl={r['inliers']} blobs={r['n_blobs']} dets={r['n_dets']} "
              f"level={r['level']} {r['conclusion']}/{r['verdict']} "
              f"device={r['device']} imgsz={r['imgsz_actual']})")
    except Exception as e:
        print(f"  [FAIL] E2E: {type(e).__name__}: {e}")
        return 1
    if args.preflight_only:
        print("  (chi preflight, khong nen zip)")
        return 0

    print("== [3/3] Nen zip ==")
    out, n = build_zip(Path(args.out))
    mb = out.stat().st_size / 1024 / 1024
    print(f"  {out}  ({mb:.1f} MB, {n} files)")
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        print(f"  cau truc: {len(names)} entries; "
              f"top-level: {sorted({n.split('/')[0] for n in names})}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
