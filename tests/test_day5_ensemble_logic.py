"""Day 5 verifier test: quy tac quyet dinh 4 muc + Box Fusion (plan.md Phan 2.4).

Chi test ham decide/fuse/fuse_and_decide voi du lieu gia — khong can GPU/YOLO thuc.
Khong sua src/; chi doc va assert. Chay CPU, nhanh.

Chay: <venv>/python.exe -m pytest tests/test_day5_ensemble_logic.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from src.decision.logic import (
    decide, FAIL_CLASSES, REWORK_CLASSES, ALL_CLASSES,
)
from src.models.ensemble import (
    fuse, fuse_and_decide, FUSION_IOU, CONF_BOOST, GATE_R,
)


# ---------------------------------------------------------------------------
# helpers tao evidence item (giong output ensemble.fuse)
# ---------------------------------------------------------------------------
def item(class_id=None, class_name=None, conf=None, source="yolo",
         bbox=None, z_diff=None, area=None):
    return {
        "class_id": class_id, "class_name": class_name, "conf": conf,
        "source": source, "bbox": bbox,
        "z_diff": z_diff, "area": area,
    }


def yolo_det(bbox, class_id, class_name, conf):
    return {"bbox": [float(v) for v in bbox], "class_id": class_id,
            "class_name": class_name, "conf": float(conf)}


def blob_tpl(centroid_tpl, area, z_diff, bbox_tpl=None):
    b = {"centroid_tpl": [float(centroid_tpl[0]), float(centroid_tpl[1])],
         "area": float(area), "z_diff": float(z_diff)}
    if bbox_tpl is not None:
        b["bbox_tpl"] = [float(v) for v in bbox_tpl]
    return b


# ---------------------------------------------------------------------------
# H_inv: identity (mac dinh) hoac translation de test Chot 1
# ---------------------------------------------------------------------------
H_ID = np.eye(3, dtype=np.float64)
H_TRANS = np.array([[1.0, 0.0, 100.0],
                    [0.0, 1.0, 50.0],
                    [0.0, 0.0, 1.0]], dtype=np.float64)


# ===========================================================================
# decide() — logic.py
# ===========================================================================

def test_decide_fail_high():
    res = decide([item(0, "open_circuit", 0.60, "fused", [10, 10, 60, 60])])
    assert res["level"] == 1
    assert res["conclusion"] == "FAIL"
    assert res["verdict"] == "HIGH"


def test_decide_rework_medium():
    res = decide([item(2, "mouse_bite", 0.40, "fused", [10, 10, 60, 60])])
    assert res["level"] == 2
    assert res["conclusion"] == "REWORK"
    assert res["verdict"] == "MEDIUM"


def test_decide_review_3a_recall_rescue():
    # 0.20 <= conf < 0.35 va Z_diff >= 2.0 -> RECALL_RESCUE
    res = decide([item(3, "spur", 0.25, "fused", [10, 10, 60, 60],
                      z_diff=2.5, area=120.0)])
    assert res["level"] == 3
    assert res["conclusion"] == "REVIEW"
    assert res["verdict"] == "RECALL_RESCUE"


def test_decide_review_3b_open_set_independent_of_yolo_conf():
    # REGRA 3B: YOLO bo sot (conf=None), Nhanh B phat hien vung lon
    # Z>=3.5 VA Area>=100 -> REVIEW/UNKNOWN_DEFECT, KHONG bi khoa bo yolo_conf>=0.30
    res = decide([item(None, None, None, "branchb", [100, 100, 180, 180],
                       z_diff=4.0, area=150.0)])
    assert res["level"] == 3
    assert res["conclusion"] == "REVIEW"
    assert res["verdict"] == "UNKNOWN_DEFECT"


def test_decide_review_3b_conf_none_still_triggers():
    # conf=None (khong co YOLO) khong chan rule 3b
    res = decide([item(None, None, None, "branchb", [0, 0, 50, 50],
                       z_diff=5.0, area=200.0)])
    assert res["level"] == 3
    assert res["verdict"] == "UNKNOWN_DEFECT"


def test_decide_review_3b_below_area_is_pass():
    # Z cao nhung Area < 100 -> khong du 3b -> PASS (neu khong co muc khac)
    res = decide([item(None, None, None, "branchb", [0, 0, 20, 20],
                       z_diff=5.0, area=50.0)])
    assert res["level"] == 4
    assert res["conclusion"] == "PASS"


def test_decide_review_3b_below_z_is_pass():
    # Area >= 100 nhung Z < 3.5 -> PASS
    res = decide([item(None, None, None, "branchb", [0, 0, 50, 50],
                       z_diff=2.0, area=150.0)])
    assert res["level"] == 4
    assert res["conclusion"] == "PASS"


def test_decide_conf_below_020_is_pass():
    # conf 0.19 < 0.20 -> khong du 3a -> PASS
    res = decide([item(0, "open_circuit", 0.19, "fused", [10, 10, 60, 60])])
    assert res["level"] == 4
    assert res["conclusion"] == "PASS"


def test_decide_conf_exactly_020_with_z_is_3a():
    # conf == 0.20 (gioi han duoi) + Z >= 2.0 -> 3a
    res = decide([item(1, "short", 0.20, "fused", [10, 10, 60, 60],
                      z_diff=2.0, area=80.0)])
    assert res["level"] == 3
    assert res["verdict"] == "RECALL_RESCUE"


def test_decide_conf_exactly_035_is_rework():
    # conf == 0.35 (gioi lan REWORK) -> REWORK
    res = decide([item(2, "mouse_bite", 0.35, "fused", [10, 10, 60, 60])])
    assert res["level"] == 2
    assert res["verdict"] == "MEDIUM"


def test_decide_conf_exactly_055_is_fail():
    # conf == 0.55 (gioi lan FAIL) -> FAIL
    res = decide([item(5, "missing_hole", 0.55, "fused", [10, 10, 60, 60])])
    assert res["level"] == 1
    assert res["verdict"] == "HIGH"


def test_decide_priority_fail_over_review3b():
    # FAIL (cap 1) uu tien hon REVIEW 3b (cap 3)
    res = decide([
        item(None, None, None, "branchb", [0, 0, 50, 50], z_diff=5.0, area=200.0),
        item(0, "open_circuit", 0.80, "fused", [10, 10, 60, 60]),
    ])
    assert res["level"] == 1
    assert res["conclusion"] == "FAIL"


def test_decide_priority_rework_over_review3a():
    # REWORK (cap 2) uu tien hon REVIEW 3a (cap 3)
    res = decide([
        item(3, "spur", 0.25, "fused", [10, 10, 60, 60], z_diff=2.5, area=120.0),
        item(4, "spurious_copper", 0.50, "fused", [20, 20, 70, 70]),
    ])
    assert res["level"] == 2
    assert res["conclusion"] == "REWORK"


def test_decide_empty_items_is_pass():
    res = decide([])
    assert res["level"] == 4
    assert res["conclusion"] == "PASS"


# ===========================================================================
# fuse() — ensemble.py (Box Fusion + Chot 1 + gating R=80)
# ===========================================================================

def test_fuse_box_fusion_iou_boost():
    # YOLO det khop IoU>=0.3 voi diff_box -> conf = min(1.0, S_A + 0.15)
    # h_inv = identity: centroid_tpl == centroid_input
    cx, cy = 500.0, 500.0
    blobs = [blob_tpl([cx, cy], area=200.0, z_diff=3.0,
                      bbox_tpl=[cx-10, cy-10, cx+10, cy+10])]
    # diff_box = [cx-80, cy-80, cx+80, cy+80] = [420,420,580,580]
    # YOLO det trong diff_box (IoU cao) -> fused, boost
    yolo = [yolo_det([450, 450, 550, 550], 0, "open_circuit", 0.50)]
    items = fuse(yolo, blobs, H_ID)
    fused = [it for it in items if it["source"] == "fused"]
    assert len(fused) == 1
    assert fused[0]["conf"] == min(1.0, 0.50 + CONF_BOOST)  # 0.65
    assert fused[0]["class_id"] == 0
    assert fused[0]["class_name"] == "open_circuit"


def test_fuse_iou_below_threshold_no_boost():
    # IoU < 0.30 -> khong boost; neu trong gate R thi source="yolo" conf=S_A
    cx, cy = 500.0, 500.0
    blobs = [blob_tpl([cx, cy], area=200.0, z_diff=3.0,
                      bbox_tpl=[cx-10, cy-10, cx+10, cy+10])]
    yolo = [yolo_det([500, 500, 540, 540], 1, "short", 0.40)]
    items = fuse(yolo, blobs, H_ID)
    # IoU(det [500,500,540,540], diff_box [420,420,580,580]) = 1600/25600 = 0.0625 < 0.3
    # center (520,520) dist to (500,500) = 28 <= 80 -> kept as "yolo", conf=0.40 (no boost)
    yolos = [it for it in items if it["source"] == "yolo"]
    assert len(yolos) == 1
    assert yolos[0]["conf"] == 0.40
    assert yolos[0]["source"] == "yolo"


def test_fuse_branchb_blob_no_yolo_match():
    # Blob khong co YOLO nao khop -> source branchb, conf None, z_diff/area duoc giu
    cx, cy = 300.0, 300.0
    blobs = [blob_tpl([cx, cy], area=150.0, z_diff=4.0,
                      bbox_tpl=[cx-5, cy-5, cx+5, cy+5])]
    items = fuse([], blobs, H_ID)
    branchb = [it for it in items if it["source"] == "branchb"]
    assert len(branchb) == 1
    assert branchb[0]["conf"] is None
    assert branchb[0]["z_diff"] == 4.0
    assert branchb[0]["area"] == 150.0


def test_fuse_spatial_gating_r80_keeps_inside_drops_outside():
    cx, cy = 500.0, 500.0
    blobs = [blob_tpl([cx, cy], area=200.0, z_diff=3.0,
                      bbox_tpl=[cx-10, cy-10, cx+10, cy+10])]
    # Det 1: center trong R=80 -> kept (as yolo, low IoU so not fused)
    # Det 2: center ngoai R=80 va IoU=0 voi diff_box -> dropped
    yolo = [
        yolo_det([500, 500, 540, 540], 0, "open_circuit", 0.50),   # center (520,520) dist 28 -> kept
        yolo_det([900, 900, 940, 940], 1, "short", 0.50),          # center (920,920) dist ~594 -> dropped
    ]
    items = fuse(yolo, blobs, H_ID)
    yolos = [it for it in items if it["source"] == "yolo"]
    fused = [it for it in items if it["source"] == "fused"]
    assert len(yolos) == 1
    assert yolos[0]["conf"] == 0.50
    assert len(fused) == 0
    # det2 (ngoai R=80, IoU=0) bi drop hoan toan; blob DA khop det1
    # (IoU>0, source=yolo) nen khong sinh branchb -> chi con 1 item.
    # (Ky vong cu "1 yolo + 1 branchb" loi thoi: tu fix root-cause-2,
    #  blob khop YOLO khong con phat branchb nua.)
    assert len(items) == 1  # 1 yolo (det1 khop blob nhung IoU<0.3 nen chua boost)


def test_fuse_centroid_projected_via_h_inv():
    # Chot 1: centroid_tpl duoc chieu qua h_inv ve he anh goc truoc khi tinh diff_box/gating
    # h_inv = translation +100,+50: centroid_tpl (500,500) -> centroid_input (600,550)
    cx_tpl, cy_tpl = 500.0, 500.0
    blobs = [blob_tpl([cx_tpl, cy_tpl], area=200.0, z_diff=3.0,
                      bbox_tpl=[cx_tpl-10, cy_tpl-10, cx_tpl+10, cy_tpl+10])]
    # diff_box tren he anh goc = [600-80, 550-80, 600+80, 550+80] = [520,470,680,630]
    # YOLO det khop IoU>=0.3 voi diff_box (100x100 => IoU ~0.39) -> fused
    yolo = [yolo_det([550, 500, 650, 600], 0, "open_circuit", 0.40)]
    items = fuse(yolo, blobs, H_TRANS)
    fused = [it for it in items if it["source"] == "fused"]
    assert len(fused) == 1
    assert fused[0]["centroid_input"] == [600.0, 550.0]


def test_fuse_not_raw_blob_bbox_bug_day3_03():
    # BUG-DAY3-03: fusion phai dung diff_box (centroid +- R), KHONG dung bbox blob chat.
    # Blob co bbox_tpl be (IoU thap vs YOLO) nhung diff_box rong (R=80) -> con the fuse qua diff_box.
    cx, cy = 500.0, 500.0
    blobs = [blob_tpl([cx, cy], area=200.0, z_diff=3.0,
                      bbox_tpl=[cx-2, cy-2, cx+2, cy+2])]  # bbox be
    # diff_box = [420,420,580,580] (rong). YOLO det [430,430,570,570]:
    #   IoU vs raw bbox [498,498,502,502] ~ 0 (thap) -> neu dung bbox blob thi khong fuse
    #   IoU vs diff_box [420,420,580,580] cao -> fused duoc
    yolo = [yolo_det([430, 430, 570, 570], 0, "open_circuit", 0.40)]
    items = fuse(yolo, blobs, H_ID)
    fused = [it for it in items if it["source"] == "fused"]
    assert len(fused) == 1, f"expected fused via diff_box, got sources={[it['source'] for it in items]}"


def test_fuse_multiple_blobs_one_yolo_each():
    # 2 blob, 2 yolo: moi yolo khop 1 blob -> 2 fused
    blobs = [
        blob_tpl([100.0, 100.0], 100.0, 3.0, [90, 90, 110, 110]),
        blob_tpl([300.0, 300.0], 100.0, 3.0, [290, 290, 310, 310]),
    ]
    yolo = [
        yolo_det([20, 20, 180, 180], 0, "open_circuit", 0.50),   # == diff_box blob1 -> IoU=1.0
        yolo_det([220, 220, 380, 380], 1, "short", 0.60),        # == diff_box blob2 -> IoU=1.0
    ]
    items = fuse(yolo, blobs, H_ID)
    fused = [it for it in items if it["source"] == "fused"]
    assert len(fused) == 2


# ===========================================================================
# fuse_and_decide() — end-to-end
# ===========================================================================

def test_fuse_and_decide_fail_end_to_end():
    cx, cy = 500.0, 500.0
    blobs = [blob_tpl([cx, cy], 200.0, 3.0, [cx-10, cy-10, cx+10, cy+10])]
    yolo = [yolo_det([450, 450, 550, 550], 0, "open_circuit", 0.60)]
    res = fuse_and_decide(yolo, blobs, H_ID)
    assert res["level"] == 1
    assert res["conclusion"] == "FAIL"
    assert res["verdict"] == "HIGH"
    assert "items" in res
    assert len(res["items"]) >= 1


def test_fuse_and_decide_open_set_review_end_to_end():
    # Blob lon, khong co YOLO khop -> branchb -> REVIEW 3b
    cx, cy = 500.0, 500.0
    blobs = [blob_tpl([cx, cy], 300.0, 4.0, [cx-5, cy-5, cx+5, cy+5])]
    res = fuse_and_decide([], blobs, H_ID)
    assert res["level"] == 3
    assert res["verdict"] == "UNKNOWN_DEFECT"


def test_constants_match_plan():
    assert FUSION_IOU == 0.30
    assert CONF_BOOST == 0.15
    assert GATE_R == 80.0
    assert FAIL_CLASSES == {0, 1, 5}
    assert REWORK_CLASSES == {2, 3, 4}
    assert ALL_CLASSES == {0, 1, 2, 3, 4, 5}
