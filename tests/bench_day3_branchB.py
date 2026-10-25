"""Verifier Day-3 independent GPU benchmark: Branch B board-ID + recall + GOOD + ROI.

Viet boi VERIFIER (duoc phep: tests/ + results/). Khong sua src/.
Chay: <venv>/python.exe tests/bench_day3_branchB.py
Output: results/branchB_day3.json + results/branchB_day3.csv
"""
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJ = Path(__file__).resolve().parent.parent  # PORTABLE: tuong doi theo project root
sys.path.insert(0, str(PROJ))

from configs.resource_paths import (  # noqa: E402
    LEGACY_PKU_ROOT, LOCAL_TEMPLATES, NEW_PROJECT_DIR, assert_writable_ok,
)

import importlib.util as _ilu

_SPEC = _ilu.spec_from_file_location(
    "ev_day1", str(PROJ / "scripts" / "evaluate.py"))
ev = _ilu.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ev)

import src.reference.board_id_and_align as ba  # noqa: E402
import src.reference.tolerance_diff as td  # noqa: E402
from src.models.yolo_detector import YOLODetector  # noqa: E402


def read_bgr(p) -> np.ndarray:
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    assert img is not None, p
    return img


def read_gray(p) -> np.ndarray:
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    assert img is not None, p
    return img


def match_blobs_to_gt(blobs, h_inv, gt_boxes, tau_tol=3.0):
    """centroid_tpl -> input qua H_inv; match neu nam trong GT box (+tau).

    Tra ve (n_matched, matches[(blob_idx, gt_idx)], proj_pts, iou_hits).
    """
    if not blobs or not gt_boxes:
        return 0, [], np.zeros((0, 2)), 0
    cents = np.float32([b["centroid_tpl"] for b in blobs])
    proj = ba.project_points(cents, h_inv)
    used = set()
    pairs = []
    for bi, (px, py) in enumerate(proj):
        for gi, gb in enumerate(gt_boxes):
            if gi in used:
                continue
            if gb[0] - tau_tol <= px <= gb[2] + tau_tol and \
               gb[1] - tau_tol <= py <= gb[3] + tau_tol:
                used.add(gi)
                pairs.append((bi, gi))
                break
    # IoU bien the: bbox_tpl 4 goc qua H_inv -> AABB, IoU>=0.3 voi GT
    iou_hits = 0
    used2 = set()
    for bi, b in enumerate(blobs):
        x1, y1, x2, y2 = b["bbox_tpl"]
        corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        q = ba.project_points(corners, h_inv)
        ax1, ay1 = float(q[:, 0].min()), float(q[:, 1].min())
        ax2, ay2 = float(q[:, 0].max()), float(q[:, 1].max())
        aa = [ax1, ay1, ax2, ay2]
        for gi, gb in enumerate(gt_boxes):
            if gi in used2:
                continue
            ix1, iy1 = max(aa[0], gb[0]), max(aa[1], gb[1])
            ix2, iy2 = min(aa[2], gb[2]), min(aa[3], gb[3])
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            ua = (aa[2] - aa[0]) * (aa[3] - aa[1]) + \
                (gb[2] - gb[0]) * (gb[3] - gb[1]) - inter
            iou = inter / ua if ua > 0 else 0.0
            if iou >= 0.3:
                used2.add(gi)
                iou_hits += 1
                break
    return len(used), pairs, proj, iou_hits


def main():
    t_all = time.perf_counter()
    cache = ba.load_cache()
    print(f"[bench] cache boards={cache['board_ids']}")

    defect = ev.collect_defect_items(LEGACY_PKU_ROOT, ev.TEST_BOARDS)
    goods = ev.collect_good_items(ev.TEST_BOARDS)
    print(f"[bench] defect={len(defect)} boxes={sum(len(i['gt']) for i in defect)} "
          f"goods={len(goods)}")
    # mau train/val de board-ID khong nham sang 01/04 (2 anh/board x 8 boards)
    others = ev.collect_defect_items(
        LEGACY_PKU_ROOT, {"05", "06", "07", "08", "09", "10", "11", "12"})
    by_board: dict[str, list] = {}
    for it in others:
        by_board.setdefault(it["board"], []).append(it)
    sample_other = []
    for bid in sorted(by_board):
        sample_other.extend(by_board[bid][:2])
    print(f"[bench] sample train/val boards={sorted(by_board)} n={len(sample_other)}")

    templates = {bid: read_gray(LOCAL_TEMPLATES / f"{bid}.JPG")
                 for bid in cache["board_ids"]}

    # warmup (khong tinh latency): 2 align fast + 1 diff
    wimg = read_bgr(defect[0]["path"])
    for _ in range(2):
        ba.identify_and_align(wimg, cache, fast=True)

    rows = []
    n_bid_ok = n_bid_tot = 0
    std_tp = std_npos = std_iou = 0
    cam_tp = cam_npos = cam_iou = 0
    lat_align, lat_diff = [], []
    lat_align_full = []
    std_blobs_z, cam_blobs_z = [], []
    std_nblob, cam_nblob = [], []

    # full-mode parity tren 6 anh dau (doi chieu chinh xac)
    for k in range(min(6, len(defect))):
        it = defect[k]
        img = read_bgr(it["path"])
        t0 = time.perf_counter()
        r = ba.identify_and_align(img, cache, fast=False)
        lat_align_full.append((time.perf_counter() - t0) * 1000.0)
        ok = (r["board_id"] == it["board"])
        n_bid_ok += ok
        n_bid_tot += 1
        rows.append({"img": Path(it["path"]).name, "board": it["board"],
                     "scenario": "standard", "mode": "full",
                     "pred": r["board_id"], "inliers": r["inliers"],
                     "ok": ok, "lat_ms": round(lat_align_full[-1], 1)})

    for i, it in enumerate(defect):
        img = read_bgr(it["path"])
        gt = [list(b) for _, b in it["gt"]]
        # camera: seed GIONG evaluate.py yolo3072 (+101) de so sanh truc tiep
        img_c, gt_c = ev.perturb_image_and_boxes(img, gt, 20261023 + 101 + i)
        for tag, im, g in (("standard", img, gt), ("camera", img_c, gt_c)):
            t0 = time.perf_counter()
            try:
                r = ba.identify_and_align(im, cache, fast=True)
                dt_a = (time.perf_counter() - t0) * 1000.0
                ok = (r["board_id"] == it["board"])
                t1 = time.perf_counter()
                tpl = templates[r["board_id"]]
                assert r["warped"].shape[:2] == tpl.shape, \
                    (r["warped"].shape, tpl.shape)
                out = td.run_branch_b(r["warped"], tpl)
                dt_d = (time.perf_counter() - t1) * 1000.0
                m, _, _, miou = match_blobs_to_gt(
                    out["blobs"], r["H_inv"], g)
                err = None
            except Exception as e:  # ghi ro, khong che dau
                dt_a = (time.perf_counter() - t0) * 1000.0
                dt_d, m, miou, ok = 0.0, 0, 0, False
                err, out, r = f"{type(e).__name__}: {e}", {"blobs": []}, None
            n_bid_ok += ok
            n_bid_tot += 1
            lat_align.append(dt_a)
            lat_diff.append(dt_d)
            if tag == "standard":
                std_tp += m
                std_npos += len(g)
                std_iou += miou
            else:
                cam_tp += m
                cam_npos += len(g)
                cam_iou += miou
            (std_blobs_z if tag == "standard" else cam_blobs_z).extend(
                b["z_diff"] for b in out["blobs"])
            (std_nblob if tag == "standard" else cam_nblob).append(len(out["blobs"]))
            rows.append({"img": Path(it["path"]).name, "board": it["board"],
                         "scenario": tag, "mode": "fast",
                         "pred": r["board_id"] if r else None,
                         "inliers": r["inliers"] if r else 0,
                         "ok": ok, "lat_ms": round(dt_a + dt_d, 1),
                         "n_blob": len(out["blobs"]),
                         "max_z": round(max([b["z_diff"] for b in out["blobs"]] + [0.0]), 2),
                         "recall_hit": m, "npos": len(g), "iou_hit": miou,
                         "error": err})

    # GOOD sach: khong duoc co blob Z>=3.5 & area>=100 (nguong open-set 3b)
    good_rows = []
    for it in goods:
        img = read_bgr(it["path"])
        img_c, _ = ev.perturb_image_and_boxes(img, [], 20261023 + 999)
        for tag, im in (("standard", img), ("camera", img_c)):
            r = ba.identify_and_align(im, cache, fast=True)
            tpl = templates[r["board_id"]]
            out = td.run_branch_b(r["warped"], tpl)
            trig = [b for b in out["blobs"]
                    if b["z_diff"] >= 3.5 and b["area"] >= 100]
            good_rows.append({"img": Path(it["path"]).name, "scenario": tag,
                              "pred": r["board_id"], "inliers": r["inliers"],
                              "n_blob": len(out["blobs"]),
                              "max_z": round(max([b["z_diff"] for b in out["blobs"]] + [0.0]), 2),
                              "openset_trig": len(trig)})
            n_bid_ok += (r["board_id"] == it["board"])
            n_bid_tot += 1

    # mau train/val: chi board-ID (chong nham sang 01/04)
    for it in sample_other:
        img = read_bgr(it["path"])
        try:
            r = ba.identify_and_align(img, cache, fast=True)
            ok = (r["board_id"] == it["board"])
            pred, inl, err = r["board_id"], r["inliers"], None
        except Exception as e:
            ok, pred, inl, err = False, None, 0, f"{type(e).__name__}: {e}"
        n_bid_ok += ok
        n_bid_tot += 1
        rows.append({"img": Path(it["path"]).name, "board": it["board"],
                     "scenario": "trainval-sample", "mode": "fast",
                     "pred": pred, "inliers": inl, "ok": ok, "error": err})

    # ROI batch 1-call + gating (3 anh 01 dau, weights day2)
    yolo = YOLODetector(str(NEW_PROJECT_DIR / "assets" / "weights" / "yolo11n_day2.pt"))
    roi_rows = []
    for it in [d for d in defect if d["board"] == "01"][:3]:
        img = read_bgr(it["path"])
        r = ba.identify_and_align(img, cache, fast=True)
        tpl = templates[r["board_id"]]
        out = td.run_branch_b(r["warped"], tpl)
        blobs = out["blobs"][:8]
        if not blobs:
            continue
        cents = ba.project_points(np.float32([b["centroid_tpl"] for b in blobs]),
                                  r["H_inv"])
        crops, origins = [], []
        H, W = img.shape[:2]
        for (px, py) in cents:
            x0 = int(np.clip(px - 320, 0, max(W - 640, 0)))
            y0 = int(np.clip(py - 320, 0, max(H - 640, 0)))
            crops.append(cv2.cvtColor(img[y0:y0 + 640, x0:x0 + 640],
                                      cv2.COLOR_BGR2RGB))
            origins.append((x0, y0))
        t0 = time.perf_counter()
        batch = yolo.predict_batch(crops)
        dt_b = (time.perf_counter() - t0) * 1000.0
        kept = 0
        for dets, (px, py), (x0, y0) in zip(batch, cents, origins):
            kept += len(YOLODetector.spatial_center_gating(
                dets, center_xy=(float(px), float(py)),
                radius=80.0, crop_origin=(float(x0), float(y0))))
        roi_rows.append({"img": Path(it["path"]).name, "n_crop": len(crops),
                         "batch_ms": round(dt_b, 1),
                         "n_det": sum(len(d) for d in batch), "kept": kept})

    summary = {
        "meta": {"conf": "board-ID fast ORB2500-halfres/top3-H; Branch B "
                 "TAU=3 MIN_AREA=60 SIGMA_FLOOR=5.0 TOP_K=8; camera seeds "
                 "20261023+101+i (giong evaluate.py yolo3072); "
                 "boards=Zero-defect-training 01/04",
                 "n_defect_imgs": len(defect),
                 "n_defect_boxes": sum(len(i["gt"]) for i in defect),
                 "total_time_s": round(time.perf_counter() - t_all, 1)},
        "board_id": {"n_total": n_bid_tot, "n_ok": n_bid_ok,
                     "acc": round(n_bid_ok / max(n_bid_tot, 1), 4),
                     "lat_full_ms_median": round(float(np.median(lat_align_full)), 1) if lat_align_full else None,
                     "lat_fast_align_ms_median": round(float(np.median(lat_align)), 1) if lat_align else None},
        "branchB_standard": {"recall_centroid": round(std_tp / max(std_npos, 1), 4),
                             "tp": std_tp, "npos": std_npos,
                             "recall_iou03": round(std_iou / max(std_npos, 1), 4),
                             "iou_tp": std_iou,
                             "lat_diff_ms_median": round(float(np.median(lat_diff)), 1) if lat_diff else None,
                             "avg_blobs": round(float(np.mean(std_nblob)), 2) if std_nblob else 0,
                             "max_z": round(max(std_blobs_z + [0.0]), 2)},
        "branchB_camera": {"recall_centroid": round(cam_tp / max(cam_npos, 1), 4),
                            "tp": cam_tp, "npos": cam_npos,
                            "recall_iou03": round(cam_iou / max(cam_npos, 1), 4),
                            "iou_tp": cam_iou,
                            "avg_blobs": round(float(np.mean(cam_nblob)), 2) if cam_nblob else 0,
                            "max_z": round(max(cam_blobs_z + [0.0]), 2)},
        "good_clean": good_rows,
        "roi_batch": roi_rows,
    }
    out_json = assert_writable_ok(NEW_PROJECT_DIR / "results" / "branchB_day3.json")
    out_json.write_text(json.dumps({"summary": summary, "rows": rows},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    out_csv = assert_writable_ok(NEW_PROJECT_DIR / "results" / "branchB_day3.csv")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img", "board", "scenario", "mode", "pred", "inliers",
                    "ok", "lat_ms", "n_blob", "max_z", "recall_hit",
                    "npos", "iou_hit", "error"])
        for r_ in rows:
            w.writerow([r_.get("img"), r_.get("board"), r_.get("scenario"),
                        r_.get("mode"), r_.get("pred"), r_.get("inliers"),
                        r_.get("ok"), r_.get("lat_ms"), r_.get("n_blob"),
                        r_.get("max_z"), r_.get("recall_hit"),
                        r_.get("npos"), r_.get("iou_hit"), r_.get("error")])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[bench] wrote {out_json} + {out_csv}")


if __name__ == "__main__":
    main()
