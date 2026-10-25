"""debug_e2e_trace.py — Trace End-to-End 1 anh demo hoặc 4 ca kiểm thử Clean-Room (PORTABLE).

Chạy pipeline đầy đủ Nhánh A + Nhánh B + Fusion 4 mức trên 1 ảnh mẫu
(board 01) và in trace chi tiết từng bước: align (board/inliers/ms) ->
diff (n blob, top Z) -> crop/batch ROI (kept/box) -> fusion -> quyết định.

Dùng cho:
  * 3_run_self_test.bat trên máy mới (kiểm tra nhanh hệ thống).
  * Pre-flight check của scripts/build_portable_zip.py.
  * Clean-Room QA với cờ --all-4-cases (kiểm định đủ 4 ca độc lập).

Ví dụ:
  python scripts/debug_e2e_trace.py
  python scripts/debug_e2e_trace.py --sample Short/short01.jpg --full
  python scripts/debug_e2e_trace.py --good01            (ảnh GOOD template 01)
  python scripts/debug_e2e_trace.py --all-4-cases       (chạy đủ 4 ca kiểm định)
  python scripts/debug_e2e_trace.py --json results/e2e_trace.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.resource_paths import (  # noqa: E402
    DEMO_ROOT,
    LOCAL_TEMPLATES,
    PROJECT_ROOT,
    assert_writable_ok,
    default_weights,
    find_pku_root,
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import src.reference.board_id_and_align as ba  # noqa: E402
import src.reference.tolerance_diff as td  # noqa: E402
from src.models.ensemble import (  # noqa: E402
    CONF_BOOST, FUSION_IOU, GATE_R, fuse_and_decide,
)
from src.models.yolo_detector import (  # noqa: E402
    CENTER_GATE_RADIUS, DEFAULT_CONF, DEFAULT_IMGSZ, YOLODetector,
)


def read_bgr(p: Path) -> np.ndarray:
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot decode image: {p}")
    return img


def resolve_default_sample() -> Path:
    """Ảnh mặc định: Short/short01.jpg board 01 trong DEMO bundled."""
    cand = DEMO_ROOT / "images" / "Short" / "short01.jpg"
    if cand.exists():
        return cand
    root = find_pku_root()
    if root is not None:
        q = root / "images" / "Short" / "short01.jpg"
        if q.exists():
            return q
    raise FileNotFoundError(
        "Không tìm thấy ảnh demo Short/short01.jpg "
        "(cần data/demo_and_test_samples/ trong gói portable).")


def perturb_camera_sim(img_bgr: np.ndarray, seed: int = 20261023) -> np.ndarray:
    """Mô phỏng nhiễu Camera (dịch chuyển, xoay, phối cảnh nhẹ, lệch sáng)."""
    rng = np.random.default_rng(seed)
    h, w = img_bgr.shape[:2]
    dx, dy = float(rng.uniform(-25, 25)), float(rng.uniform(-25, 25))
    ang = float(rng.uniform(-2.0, 2.0))
    cx, cy = w / 2.0, h / 2.0
    m_aff = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    m_aff[0, 2] += dx
    m_aff[1, 2] += dy
    h_aff = np.vstack([m_aff, [0.0, 0.0, 1.0]])

    j = 0.008
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-j, j, size=(4, 2)).astype(np.float32) * np.float32([w, h])
    h_persp, _ = cv2.findHomography(src, dst)
    if h_persp is None:
        h_tot = h_aff
    else:
        h_tot = h_aff @ h_persp

    warped = cv2.warpPerspective(img_bgr, h_tot, (w, h), borderMode=cv2.BORDER_REPLICATE)
    gain = float(rng.uniform(0.85, 1.15))
    bias = float(rng.uniform(-10.0, 10.0))
    out = np.clip(warped.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    return out


def run_pipeline_on_image(img_bgr: np.ndarray, yolo: YOLODetector, cache: dict,
                          fast: bool = True, verbose: bool = True) -> dict:
    """Chạy toàn bộ pipeline Dual-Path trên 1 ảnh ndarray."""
    h_img, w_img = img_bgr.shape[:2]

    # Nhánh B: align
    t0 = time.perf_counter()
    r = ba.identify_and_align(img_bgr, cache, fast=fast)
    t_align = (time.perf_counter() - t0) * 1000.0

    tpl_p = LOCAL_TEMPLATES / f"{r['board_id']}.JPG"
    tpl = read_bgr(tpl_p)
    tpl_g = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    if tpl_g.shape != r["warped"].shape[:2]:
        tpl_g = cv2.resize(tpl_g, (r["warped"].shape[1], r["warped"].shape[0]),
                           interpolation=cv2.INTER_AREA)

    # Nhánh B: tolerance diff
    t0 = time.perf_counter()
    out = td.run_branch_b(r["warped"], tpl_g, valid_mask=r.get("valid_mask"))
    t_diff = (time.perf_counter() - t0) * 1000.0
    blobs = out["blobs"]

    # Chốt 1: centroid qua H_inv
    cents = (ba.project_points(
        np.float32([b["centroid_tpl"] for b in blobs]), r["H_inv"])
        if blobs else np.zeros((0, 2), np.float32))

    # Nhánh A: batch ROI (Chốt 3: 1 call)
    crops, origins = [], []
    for (px, py) in cents:
        x0 = int(np.clip(float(px) - 320, 0, max(w_img - 640, 0)))
        y0 = int(np.clip(float(py) - 320, 0, max(h_img - 640, 0)))
        x1, y1 = min(x0 + 640, w_img), min(y0 + 640, h_img)
        crop = img_bgr[y0:y1, x0:x1]
        if crop.shape[0] != 640 or crop.shape[1] != 640:
            pad = np.zeros((640, 640, 3), np.uint8)
            pad[:crop.shape[0], :crop.shape[1]] = crop
            crop = pad
        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        origins.append((x0, y0))

    t0 = time.perf_counter()
    batch = yolo.predict_batch(crops) if crops else []
    t_roi = (time.perf_counter() - t0) * 1000.0
    yolo_dets_all, kept_per_crop = [], []
    for dets_one, (px, py), (x0, y0) in zip(batch, cents, origins):
        kept = YOLODetector.spatial_center_gating(
            dets_one, center_xy=(float(px), float(py)),
            radius=float(CENTER_GATE_RADIUS),
            crop_origin=(float(x0), float(y0)))
        kept_per_crop.append(len(kept))
        for d in kept:
            dd = dict(d)
            dd["bbox"] = [d["bbox"][0] + x0, d["bbox"][1] + y0,
                          d["bbox"][2] + x0, d["bbox"][3] + y0]
            yolo_dets_all.append(dd)

    # Fusion + quyết định 4 mức
    res = fuse_and_decide(yolo_dets_all, blobs, r["H_inv"])

    if verbose:
        print(f"      [ALIGN] board={r['board_id']} inliers={r['inliers']} ({t_align:.0f}ms)")
        print(f"      [DIFF] blobs={len(blobs)} ({t_diff:.0f}ms)")
        print(f"      [ROI] crops={len(crops)} yolo_dets={len(yolo_dets_all)} ({t_roi:.0f}ms)")
        print(f"      [VERDICT] Level {res['level']}: {res.get('conclusion')} ({res.get('verdict')})")

    return {
        "align": r, "blobs": blobs, "dets": yolo_dets_all, "res": res,
        "t_align": t_align, "t_diff": t_diff, "t_roi": t_roi,
        "shape": [w_img, h_img],
    }


def run_all_4_cases(yolo: YOLODetector, cache: dict, fast: bool = True) -> bool:
    """Kiểm tra đủ 4 kịch bản chuẩn cho Clean-Room QA."""
    print("=" * 64)
    print("DEBUG E2E TRACE: CHẠY 4 CA KIỂM ĐỊNH PHÒNG SẠCH (CLEAN-ROOM QA)")
    print("=" * 64)

    all_pass = True

    # Ca 1: Ảnh GOOD 01.JPG
    print("\n--- [Ca 1] Ảnh GOOD (assets/templates/01.JPG) ---")
    good_p = LOCAL_TEMPLATES / "01.JPG"
    assert good_p.exists(), f"Thiếu {good_p}"
    img1 = read_bgr(good_p)
    out1 = run_pipeline_on_image(img1, yolo, cache, fast=fast, verbose=True)
    level1 = out1["res"]["level"]
    if level1 == 4:
        print("  -> Ca 1: PASS [OK] (Đạt Level 4: PASS, FAR=0)")
    else:
        print(f"  -> Ca 1: FAILED (Kỳ vọng Level 4, thực tế Level {level1})")
        all_pass = False

    # Ca 2: Ảnh DEFECT chuẩn (Short/short01.jpg hoặc Missing_hole)
    print("\n--- [Ca 2] Ảnh DEFECT chuẩn (data/demo_and_test_samples/.../short01.jpg) ---")
    defect_p = resolve_default_sample()
    img2 = read_bgr(defect_p)
    out2 = run_pipeline_on_image(img2, yolo, cache, fast=fast, verbose=True)
    level2 = out2["res"]["level"]
    if level2 in (1, 2, 3):
        print(f"  -> Ca 2: PASS [OK] (Bắt được lỗi ở Level {level2}: {out2['res'].get('conclusion')})")
    else:
        print(f"  -> Ca 2: FAILED (Bỏ lọt lỗi sang Level {level2})")
        all_pass = False

    # Ca 3: Ảnh DEFECT nhiễu Camera (Perturbation shift/rotate/homography)
    print("\n--- [Ca 3] Ảnh DEFECT chịu nhiễu Camera (Perturbation) ---")
    img3 = perturb_camera_sim(img2, seed=20261023)
    out3 = run_pipeline_on_image(img3, yolo, cache, fast=fast, verbose=True)
    level3 = out3["res"]["level"]
    inliers3 = out3["align"]["inliers"]
    if inliers3 >= 30 and level3 in (1, 2, 3):
        print(f"  -> Ca 3: PASS [OK] (Căn chỉnh thành công inliers={inliers3}, bắt lỗi ở Level {level3})")
    else:
        print(f"  -> Ca 3: FAILED (inliers={inliers3}, Level={level3})")
        all_pass = False

    # Ca 4: Ảnh lỗi lạ Open-Set (Synthetic anomaly trên 01.JPG)
    print("\n--- [Ca 4] Ảnh lỗi lạ Open-Set (Synthetic Anomaly trên 01.JPG) ---")
    img4 = read_bgr(good_p)
    # Vẽ mảng dị vật đậm màu 150x150 px đè lên vùng đường đồng (x=1400, y=800)
    cv2.rectangle(img4, (1400, 800), (1550, 950), (20, 20, 20), -1)
    out4 = run_pipeline_on_image(img4, yolo, cache, fast=fast, verbose=True)
    level4 = out4["res"]["level"]
    has_branchb = any(it.get("source") == "branchb" for it in out4["res"]["items"])
    max_z = max([b["z_diff"] for b in out4["blobs"]] or [0.0])
    if level4 == 3 and max_z >= 3.5:
        print(f"  -> Ca 4: PASS [OK] (Kích hoạt Rule 3b: Level 3 REVIEW / UNKNOWN_DEFECT, max Z={max_z:.2f} >= 3.5)")
    elif level4 in (1, 2, 3):
        print(f"  -> Ca 4: PASS (Chấp nhận được: Bắt được dị vật ở Level {level4}, max Z={max_z:.2f})")
    else:
        print(f"  -> Ca 4: FAILED (Bỏ lọt lỗi lạ: Level {level4}, max Z={max_z:.2f})")
        all_pass = False

    print("\n" + "=" * 64)
    if all_pass:
        print("TỔNG KẾT: CẢ 4 CA KIỂM THỬ ĐỀU ĐẠT 100% PASS!")
    else:
        print("TỔNG KẾT: CÓ CA KIỂM THỬ CHƯA ĐẠT!")
    print("=" * 64)
    return all_pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Trace E2E 1 ảnh demo Dual-Path hoặc chạy 4 ca Clean-Room.")
    p.add_argument("--sample", type=str, default=None,
                   help="Tên file rel trong images/ (VD Short/short01.jpg).")
    p.add_argument("--good01", action="store_true",
                   help="Dùng ảnh GOOD template 01.JPG thay vì ảnh lỗi.")
    p.add_argument("--all-4-cases", action="store_true",
                   help="Chạy toàn bộ 4 ca kiểm thử Clean-Room (GOOD, DEFECT, CAMERA, OPEN-SET).")
    p.add_argument("--weights", type=Path, default=None,
                   help="YOLO .pt (mặc định portable: day2 nếu có).")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--imgsz", type=int, default=None,
                   help="Ghi đè imgsz Nhánh A (mặc định: auto-adapt theo VRAM/CPU).")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--full", action="store_true",
                   help="Align full (~5.7s) thay vì fast (~45ms).")
    p.add_argument("--json", type=Path, default=None,
                   help="Ghi trace JSON (VD results/e2e_trace.json).")
    return p


def main() -> int:
    args = build_parser().parse_args()

    wpath = Path(args.weights) if args.weights else default_weights()
    if not wpath.exists():
        raise FileNotFoundError(f"Thiếu weights YOLO: {wpath}")

    import torch
    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    req_imgsz = int(args.imgsz) if args.imgsz else None
    yolo = YOLODetector(wpath, conf_threshold=float(args.conf),
                        imgsz=req_imgsz or DEFAULT_IMGSZ, device=dev)
    cache = ba.load_cache()

    if args.all_4_cases:
        ok = run_all_4_cases(yolo, cache, fast=not args.full)
        return 0 if ok else 1

    # Chế độ chạy đơn ảnh thông thường
    if args.good01:
        img_path = LOCAL_TEMPLATES / "01.JPG"
    elif args.sample:
        cands = [DEMO_ROOT / "images" / args.sample]
        root = find_pku_root()
        if root is not None:
            cands.append(root / "images" / args.sample)
        img_path = next((c for c in cands if c.exists()), cands[0])
    else:
        img_path = resolve_default_sample()

    print(f"[e2e] Ảnh: {img_path}")
    img_bgr = read_bgr(img_path)
    h_img, w_img = img_bgr.shape[:2]
    print(f"[e2e] Kích thước: {w_img}x{h_img} | device={yolo.device} half={yolo.half} imgsz={yolo.imgsz}")

    out = run_pipeline_on_image(img_bgr, yolo, cache, fast=not args.full, verbose=True)

    if args.json is not None:
        out_p = assert_writable_ok(Path(args.json))
        out_p.parent.mkdir(parents=True, exist_ok=True)
        res = out["res"]
        r = out["align"]
        blobs = out["blobs"]
        payload = {
            "image": str(img_path), "shape": [w_img, h_img],
            "weights": str(wpath), "device": yolo.device,
            "imgsz_requested": yolo.requested_imgsz,
            "imgsz_actual": yolo.imgsz,
            "board_id": r["board_id"], "inliers": int(r["inliers"]),
            "n_blobs": len(blobs), "n_dets": len(out["dets"]),
            "level": res["level"],
            "conclusion": res.get("conclusion"),
            "verdict": res.get("verdict"),
            "reasons": res.get("reasons", []),
        }
        out_p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[e2e] Đã ghi {out_p}")

    print("[debug_e2e_trace] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
