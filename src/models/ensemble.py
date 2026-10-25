"""Box Fusion + hop nhat Nhanh A (YOLO) / Nhanh B (template) — Ngay 5 (plan.md Phan 2.4).

Input:
  * yolo_dets: list det tu `YOLODetector.predict` (toa do anh goc),
    {"bbox":[x1,y1,x2,y2], "class_id", "class_name", "conf"}.
  * blobs: list tu `td.run_branch_b` (toa do Template),
    {"centroid_tpl":[cx,cy], "area", "z_diff", ...}.
  * h_inv: ma tran Template -> Input (Chot 1: chieu tam blob qua H_inv
    TRUOC khi crop/tinh IoU).

Box Fusion (plan 2.4):
  IoU(B_A, B_B) >= 0.3  ->  S_final = min(1.0, S_A + 0.15)

Bao ve (Chot Nhanh B buoc 5): chi chap nhan box YOLO neu tam nam trong
ban kinh R=80px quanh tam diff (quy he anh goc) HOAC giao cat (IoU>0) voi
diff_box. Loai bo hoan toan box bao gia o ria crop.

BUG-DAY3-03: fusion tren **centroid (da qua H_inv) + R=80/diff_box**,
KHONG dung bbox blob chat qua H_inv (IoU blob chatch thap: 13/105 std,
19/105 camera).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.decision.logic import decide
from src.models.yolo_detector import YOLODetector
import src.reference.board_id_and_align as ba

FUSION_IOU = 0.30      # plan 2.4: IoU(B_A, B_B) >= 0.3 -> hop nhat
CONF_BOOST = 0.15      # plan 2.4: S_final = min(1.0, S_A + 0.15)
GATE_R = 80.0          # px (plan 2.3 buoc 5 / Chot Nhanh B)


def _box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    ua = (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])) \
        + (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])) - inter
    return inter / ua if ua > 0.0 else 0.0


def _project_centroid(centroid_tpl: list[float], h_inv: np.ndarray) -> tuple[float, float]:
    """Chot 1: tam blob (khong gian Template) -> khong gian anh goc qua H_inv."""
    cx, cy = ba.project_points(
        np.float32([[float(centroid_tpl[0]), float(centroid_tpl[1])]]),
        h_inv)[0]
    return float(cx), float(cy)


def fuse(yolo_dets: list[dict[str, Any]],
         blobs: list[dict[str, Any]],
         h_inv: np.ndarray,
         img_shape: tuple[int, int] | None = None,
         iou_thr: float = FUSION_IOU,
         conf_boost: float = CONF_BOOST,
         gate_r: float = GATE_R) -> list[dict[str, Any]]:
    """Hop nhat det YOLO voi blob Nhanh B. Tra ve danh sach evidence item
    cho `src.decision.logic.decide`.

    Muc doi tuong:
      * fused: YOLO det khop IoU>=iou_thr voi diff_box cua mot blob ->
        conf = min(1.0, S_A + conf_boost) (Box Fusion, plan 2.4).
      * yolo: YOLO det khop voi blob (tam det trong diff_box / IoU>0, xem
        Buoc 5 plan 2.3) nhung IoU < iou_thr -> khong duoc boost,
        conf = S_A, class_id duoc giu de logic muc 1/2/3a xet duyet.
      * branchb: blob khong khop YOLO nao (conf=None, z_diff/area duoc giu
        de logic muc 3b open-set xet duyet).

    Dieu kien "khop" theo Buoc 5 (plan 2.3): tam box YOLO nam trong
    ban kinh R=80 quanh tam diff (duong vien diff_box) HOAC giao cat
    (IoU>0) voi diff_box. Vay moi det duoc spatial_center_gating giu
    se khop duoc blob tuong ung -> khong con truong hop toan bo bi loai
    (n_det=0 / tp=0) khi IoU det nho vs diff_box 160x160 < 0.3.
    """
    items: list[dict[str, Any]] = []
    matched_yolo: set[int] = set()
    gate_boxes: list[list[float]] = []  # diff_box cua moi blob (toa do anh goc)

    for b in blobs:
        cx_tpl, cy_tpl = float(b["centroid_tpl"][0]), float(b["centroid_tpl"][1])
        cx, cy = _project_centroid([cx_tpl, cy_tpl], h_inv)
        diff_box = [cx - gate_r, cy - gate_r, cx + gate_r, cy + gate_r]
        gate_boxes.append(diff_box)

        best_iou, best_j = 0.0, None
        for j, d in enumerate(yolo_dets):
            if j in matched_yolo:
                continue
            iou = _box_iou([float(v) for v in d["bbox"]], diff_box)
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_j is not None and best_iou > 0.0:
            # Tam det nam trong diff_box (R=80) <=> IoU(det, diff_box) > 0
            # (Buoc 5). Chon det co IoU cao nhat voi diff_box.
            matched_yolo.add(best_j)
            d = yolo_dets[best_j]
            s_a = float(d["conf"])
            if best_iou >= iou_thr:
                s_final = min(1.0, s_a + conf_boost)
                source = "fused"
            else:
                s_final = s_a
                source = "yolo"
            items.append({
                "class_id": d["class_id"], "class_name": d["class_name"],
                "conf": s_final, "source": source,
                "bbox": [float(v) for v in d["bbox"]],
                "z_diff": float(b.get("z_diff", 0.0)),
                "area": float(b.get("area", 0.0)),
                "diff_box": diff_box, "centroid_input": [cx, cy]})
        else:
            items.append({
                "class_id": None, "class_name": None, "conf": None,
                "source": "branchb", "bbox": diff_box,
                "z_diff": float(b.get("z_diff", 0.0)),
                "area": float(b.get("area", 0.0)),
                "diff_box": diff_box, "centroid_input": [cx, cy]})

    # YOLO con lai (chua fusion): kep neu o trong gate cua it nhat mot blob.
    for j, d in enumerate(yolo_dets):
        if j in matched_yolo:
            continue
        gx1, gy1, gx2, gy2 = [float(v) for v in d["bbox"]]
        bx = (gx1 + gx2) / 2.0
        by = (gy1 + gy2) / 2.0
        kept = False
        for gb in gate_boxes:
            cx, cy = gb[0] + gate_r, gb[1] + gate_r
            if math.hypot(bx - cx, by - cy) <= gate_r:
                kept = True
                break
            if _box_iou([gx1, gy1, gx2, gy2], gb) > 0.0:
                kept = True
                break
        if kept:
            items.append({
                "class_id": d["class_id"], "class_name": d["class_name"],
                "conf": float(d["conf"]), "source": "yolo",
                "bbox": [gx1, gy1, gx2, gy2],
                "z_diff": None, "area": None,
                "diff_box": None, "centroid_input": None})

    return items


def fuse_and_decide(yolo_dets: list[dict[str, Any]],
                    blobs: list[dict[str, Any]],
                    h_inv: np.ndarray,
                    img_shape: tuple[int, int] | None = None,
                    **kwargs: Any) -> dict[str, Any]:
    """Fusion roi ngay chung trinh quyet dinh 4 muc. Tra ve dict cua
    `decide` them truong 'items' (evidence chi tiet)."""
    items = fuse(yolo_dets, blobs, h_inv, img_shape, **kwargs)
    res = decide(items)
    res["items"] = items
    return res
