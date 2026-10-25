"""Verifier Day-1 strict tests for scripts/evaluate.py (board-disjoint 01/04).

Scope: chi doc src/scripts (READ-ONLY), khong sua src/.
Chay: <venv>/python.exe -m pytest tests/test_day1_evaluate.py -v
"""
import importlib.util
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent  # PORTABLE: tuong doi theo project root
sys.path.insert(0, str(PROJ))

SPEC = importlib.util.spec_from_file_location(
    "ev_day1", str(PROJ / "scripts" / "evaluate.py"))
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)

from configs.resource_paths import (  # noqa: E402
    LEGACY_PKU_ROOT, LOCAL_TEMPLATES, LOCAL_BASE_PT,
    LEGACY_PATCHCORE_CKPT, LEGACY_PROJECT_DIR, assert_writable_ok,
)


def test_split_board_disjoint_no_leakage():
    import pytest
    if not LEGACY_PKU_ROOT.exists():
        pytest.skip('LEGACY_PKU_ROOT not present in Portable Clean-Room')
    assert ev.TEST_BOARDS == {"01", "04"}
    assert ev.TRAIN_BOARDS.isdisjoint(ev.TEST_BOARDS)
    assert ev.VAL_BOARDS.isdisjoint(ev.TEST_BOARDS)
    assert ev.TRAIN_BOARDS.isdisjoint(ev.VAL_BOARDS)
    defect = ev.collect_defect_items(LEGACY_PKU_ROOT, ev.TEST_BOARDS)
    # Coder bao 36 anh / 105 boxes — Verifier dem doc lap
    assert len(defect) == 36, f"test imgs={len(defect)} != 36"
    nbox = sum(len(i["gt"]) for i in defect)
    assert nbox == 105, f"test boxes={nbox} != 105"
    boards = {i["board"] for i in defect}
    assert boards == {"01", "04"}, boards
    # train/val ton tai, khong de trong (chong rui ro split rong)
    tr = ev.collect_defect_items(LEGACY_PKU_ROOT, ev.TRAIN_BOARDS)
    va = ev.collect_defect_items(LEGACY_PKU_ROOT, ev.VAL_BOARDS)
    assert len(tr) > 0 and len(va) > 0
    # khong trung file giua test vs train/val
    p_test = {i["path"] for i in defect}
    assert p_test.isdisjoint({i["path"] for i in tr})
    assert p_test.isdisjoint({i["path"] for i in va})


def test_good_templates_are_golden_sample():
    goods = ev.collect_good_items(ev.TEST_BOARDS)
    assert len(goods) == 2
    names = sorted(Path(g["path"]).name for g in goods)
    assert names == ["01.JPG", "04.JPG"], names
    for g in goods:
        assert g["gt"] == []
        assert Path(g["path"]).exists()


def test_yolo_runner_forces_rect_half():
    import inspect
    src = inspect.getsource(ev.YoloRunner.infer_bgr)
    assert "rect=True" in src, "thieu rect=True (bay Square Padding)"
    assert "half" in src, "thieu half (Chot latency)"
    # mac dinh conf=0.20 theo plan 2.2
    assert ev.DEFAULT_CONF == 0.20


def test_class_map_matches_weights():
    from ultralytics import YOLO
    m = YOLO(str(LOCAL_BASE_PT))
    # weights: {0: open_circuit, ... 5: missing_hole}
    assert m.names[0] == "open_circuit"
    assert m.names[5] == "missing_hole"
    for name, cid in ev.CLASS_ID.items():
        assert m.names[cid] == name, (cid, m.names[cid], name)


def test_no_patchcore_sahi_in_decision_pipeline():
    # src/ hien chi co __init__ (Ngay 3-5 chua toi) -> khong duoc co PatchCore/SAHI
    for p in PROJ.glob("src/**/*.py"):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        assert "patchcore" not in txt.lower(), f"PatchCore lot vao {p}"
        assert "sahi" not in txt.lower(), f"SAHI lot vao {p}"
    # trong evaluate.py, PatchCore/SAHI phai co nhan ABLATION / baseline
    txt = (PROJ / "scripts" / "evaluate.py").read_text(encoding="utf-8")
    assert "[ABLATION]" in txt


def test_no_ecc_float32_risk():
    txt = (PROJ / "scripts" / "evaluate.py").read_text(encoding="utf-8")
    # Ngay 1 khong dung ECC; neu co thi phai co inputMask=None hoac CV_8UC1
    if "findTransformECC" in txt:
        assert "inputMask=None" in txt or "CV_8U" in txt, \
            "ECC mask phai la None hoac CV_8UC1 (crash OpenCV 4.11)"
    # findHomography dung cho perturbation tong hop la hop le (khong phai ECC)
    assert "findHomography" in txt


def test_zero_defect_terminology_no_unseen_boards():
    txt = (PROJ / "scripts" / "evaluate.py").read_text(encoding="utf-8").lower()
    assert "zero-defect-training" in txt
    # cam cum tu "unseen boards" (chi duoc phep nhac "khong goi la unseen")
    assert "unseen boards" not in txt, "cam goi 01/04 la 'unseen boards'"


def test_writes_guarded_inside_project():
    # output phai qua assert_writable_ok, khong ghi de project cu
    txt = (PROJ / "scripts" / "evaluate.py").read_text(encoding="utf-8")
    assert "assert_writable_ok" in txt
    # mo file legacy mode 'w' la vi pham zero-touch
    assert "LEGACY_PKU_ROOT" in txt
    # khong co shutil.copytree / rmtree nguy hiem trong evaluate
    assert "rmtree" not in txt and "unlink()" not in txt
    # assert_writable_ok chan ghi ngoai project
    import pytest
    with pytest.raises(PermissionError):
        assert_writable_ok(LEGACY_PROJECT_DIR / "x.json")  # PORTABLE: van ngoai project -> phai raise


def test_perturb_warps_boxes_deterministically():
    import numpy as np
    img = (np.random.default_rng(0).uniform(0, 255, (800, 600, 3))).astype("uint8")
    boxes = [[100.0, 100.0, 200.0, 150.0]]
    o1, b1 = ev.perturb_image_and_boxes(img, boxes, 123)
    o2, b2 = ev.perturb_image_and_boxes(img, boxes, 123)
    assert o1.shape == img.shape
    assert b1 == b2, "perturb phai deterministic theo seed"
    assert len(b1) == 1 and b1[0] != boxes[0], "box phai duoc warp theo H"
    # seed khac -> anh khac (chongBug seed cung)
    o3, _ = ev.perturb_image_and_boxes(img, boxes, 999)
    assert not (o1 == o3).all()


def test_metrics_sanity_perfect_and_worst():
    # VOC AP: perfect dets -> AP=1
    dets = [(0, 0.9, [0, 0, 10, 10], 0)]
    gts = {0: [(0, [0, 0, 10, 10])]}
    ap, _ = ev.voc_ap(dets, gts, thr=0.5)
    assert abs(ap - 1.0) < 1e-6
    tp, fp, npos = ev.pr_counts(dets, gts, thr=0.5)
    assert (tp, fp, npos) == (1, 0, 1)
    # AUROC: phan tach hoan hao -> 1.0; dao nguoc -> 0.0
    assert abs(ev.compute_auroc([0.9, 0.1], [1, 0]) - 1.0) < 1e-9
    assert abs(ev.compute_auroc([0.1, 0.9], [1, 0]) - 0.0) < 1e-9


def test_h_inv_sigma_floor_not_applicable_day1():
    # Cap nhat Day3: Nhanh B da ton tai -> pham vi Day1 khong audit H_inv;
    # audit that nam trong tests/test_day3_branchB.py (15/15 PASS).
    assert (PROJ / "src" / "reference" / "tolerance_diff.py").exists()
    assert (PROJ / "src" / "reference" / "board_id_and_align.py").exists()
