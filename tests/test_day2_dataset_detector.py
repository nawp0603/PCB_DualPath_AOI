"""Verifier Day-2 strict tests: prepare_yolo_dataset + YOLODetector + finetune script.

Scope: chi doc src/scripts (READ-ONLY), khong sua src/.
Chay: <venv>/python.exe -m pytest tests/test_day2_dataset_detector.py -v
"""
import importlib.util
import inspect
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent  # PORTABLE: tuong doi theo project root
sys.path.insert(0, str(PROJ))

SPEC_PREP = importlib.util.spec_from_file_location(
    "prep_day2", str(PROJ / "scripts" / "prepare_yolo_dataset.py"))
prep = importlib.util.module_from_spec(SPEC_PREP)
SPEC_PREP.loader.exec_module(prep)

SPEC_FT = importlib.util.spec_from_file_location(
    "ft_day2", str(PROJ / "scripts" / "finetune_yolo_day2.py"))
ft = importlib.util.module_from_spec(SPEC_FT)
SPEC_FT.loader.exec_module(ft)

from configs.resource_paths import (  # noqa: E402
    LEGACY_PKU_ROOT, LOCAL_TEMPLATES, LOCAL_BASE_PT,
    LEGACY_PROJECT_DIR, NEW_PROJECT_DIR, assert_writable_ok,
)

YOLO_DAY2_DIR = NEW_PROJECT_DIR / "data" / "yolo_day2"
DAY2_PT = NEW_PROJECT_DIR / "assets" / "weights" / "yolo11n_day2.pt"


def test_clip_keeps_partial_drops_tiny():
    # bbox 100x100 tai (0,0)-(100,100), tile 640 tai (50,0): giao 50x100 = 50% -> giu
    kept = prep.clip_bbox_to_tile([0, 0, 100, 100], 50, 0, 640)
    assert kept is not None, "clip cat ria 50% phai GIU (ban cu drop -> FN)"
    assert all(0.0 <= v <= 1.0 for v in kept)
    # sliver 5px -> drop (canh < 8px)
    assert prep.clip_bbox_to_tile([0, 0, 100, 100], 97, 0, 640) is None
    # giao <25% dien tich -> drop
    assert prep.clip_bbox_to_tile([0, 0, 100, 100], 80, 0, 640) is None
    # bbox nam hoan toan trong tile -> giu, to dung
    full = prep.clip_bbox_to_tile([100, 100, 200, 150], 0, 0, 640)
    assert full is not None
    cx, cy, w, h = full
    assert abs(cx * 640 - 150) < 1e-6 and abs(cy * 640 - 125) < 1e-6


def test_dataset_counts_match_coder_report():
    import pytest
    if not (YOLO_DAY2_DIR / 'train' / 'images').exists():
        pytest.skip('Training tiles excluded in Portable Clean-Room')
    stats = json.loads((YOLO_DAY2_DIR / "stats.json").read_text(encoding="utf-8"))
    assert stats["train_defect_tiles"] == 213, stats["train_defect_tiles"]
    assert stats["train_bg_tiles"] == 53, stats["train_bg_tiles"]
    assert stats["val_defect_tiles"] == 55
    assert stats["val_bg_tiles"] == 10
    # ti le bg ~25%
    assert abs(stats["train_bg_tiles"] / stats["train_defect_tiles"] - 0.25) < 0.01
    # dem file thuc te
    tr_img = list((YOLO_DAY2_DIR / "train" / "images").glob("*.png"))
    tr_lb = list((YOLO_DAY2_DIR / "train" / "labels").glob("*.txt"))
    va_img = list((YOLO_DAY2_DIR / "val" / "images").glob("*.png"))
    assert len(tr_img) == 266 and len(tr_lb) == 266, (len(tr_img), len(tr_lb))
    assert len(va_img) == 65
    # moi anh co label tuong ung
    for im in tr_img[:20]:
        assert (YOLO_DAY2_DIR / "train" / "labels" / (im.stem + ".txt")).exists()


def test_no_board_01_04_leakage_in_train_val():
    import pytest
    if not (YOLO_DAY2_DIR / 'train' / 'images').exists():
        pytest.skip('Training tiles excluded in Portable Clean-Room')
    # khong tile nao tu board 01/04 trong train/val
    for split in ("train", "val"):
        for p in (YOLO_DAY2_DIR / split / "images").glob("*.png"):
            n = p.name
            assert "_01_" not in n and n[12:14] != "01" or "_01_" not in n, n
            # cach chac: ten chua board id sau class: <class>_<bid>_ ; bid chi 05-12/07
            assert "_01_" not in n and "_04_" not in n, f"leak 01/04: {n}"
    stats = json.loads((YOLO_DAY2_DIR / "stats.json").read_text(encoding="utf-8"))
    assert stats["boards"]["heldout_zero_defect_training"] == ["01", "04"]
    assert set(stats["boards"]["train"]) == {"05", "06", "08", "09", "10", "11", "12"}
    assert set(stats["boards"]["val"]) == {"07"}
    # bg tiles chi tu board train/val
    for p in (YOLO_DAY2_DIR / "train" / "images").glob("bg_*.png"):
        bid = p.stem.split("_")[1]
        assert bid in {"05", "06", "08", "09", "10", "11", "12"}, bid


def test_labels_yolo_format_and_bg_empty():
    import pytest
    if not (YOLO_DAY2_DIR / 'train' / 'images').exists():
        pytest.skip('Training tiles excluded in Portable Clean-Room')
    # label YOLO: cls cx cy w h normalized; bg tiles rong
    n_bg_checked = 0
    for lb in (YOLO_DAY2_DIR / "train" / "labels").glob("bg_*.txt"):
        assert lb.read_text(encoding="utf-8").strip() == "", f"bg label phai rong: {lb}"
        n_bg_checked += 1
    assert n_bg_checked == 53
    for lb in list((YOLO_DAY2_DIR / "train" / "labels").glob("*.txt"))[:30]:
        if lb.name.startswith("bg_"):
            continue
        for line in lb.read_text(encoding="utf-8").strip().splitlines():
            parts = line.split()
            assert len(parts) == 5, line
            cid = int(parts[0])
            assert 0 <= cid <= 5, line
            for v in parts[1:]:
                assert 0.0 <= float(v) <= 1.0, line


def test_data_yaml_inside_project_and_names():
    import yaml
    cfg = yaml.safe_load((YOLO_DAY2_DIR / "data.yaml").read_text(encoding="utf-8"))
    assert cfg["names"] == {0: "open_circuit", 1: "short", 2: "mouse_bite",
                            3: "spur", 4: "spurious_copper", 5: "missing_hole"}
    assert cfg.get("path") in (None, "", str(YOLO_DAY2_DIR))
    if not (YOLO_DAY2_DIR / cfg['train']).is_dir():
        import pytest
        pytest.skip('Training tiles excluded in Portable Clean-Room')
    assert (YOLO_DAY2_DIR / cfg["val"]).is_dir()


def test_zero_touch_writes_guarded():
    txt_prep = (PROJ / "scripts" / "prepare_yolo_dataset.py").read_text(encoding="utf-8")
    assert "assert_writable_ok" in txt_prep
    assert "rmtree" not in txt_prep and "unlink()" not in txt_prep
    # khong co cum 'unseen boards'
    assert "unseen boards" not in txt_prep.lower()
    assert "Zero-defect-training" in txt_prep or "zero_defect" in txt_prep.lower()
    import pytest
    with pytest.raises(PermissionError):
        assert_writable_ok(LEGACY_PROJECT_DIR / "data" / "x.json")  # PORTABLE: van ngoai project -> phai raise
    # finetune: out va project deu qua assert_writable_ok, khong ghi de base
    assert ft.DEFAULT_OUT != ft.BASE_PT
    assert ft.DEFAULT_OUT.name == "yolo11n_day2.pt"
    assert LOCAL_BASE_PT.exists()
    txt_ft = (PROJ / "scripts" / "finetune_yolo_day2.py").read_text(encoding="utf-8")
    assert txt_ft.count("assert_writable_ok") >= 2
    assert "resolve()" in txt_ft  # fix absolute --project


def test_yolo_detector_forces_rect_half_and_defaults():
    from src.models.yolo_detector import YOLODetector
    import src.models.yolo_detector as yd
    assert yd.DEFAULT_IMGSZ == 3072
    assert yd.DEFAULT_CONF == 0.20
    assert yd.CENTER_GATE_RADIUS == 80.0
    src_single = inspect.getsource(YOLODetector.predict)
    assert "rect=True" in src_single, "thieu rect=True (bay Square Padding)"
    assert "half" in src_single, "thieu half"
    src_batch = inspect.getsource(YOLODetector.predict_batch)
    src_batch_full = src_batch
    assert "rect=True" in src_batch_full
    assert "half" in src_batch_full
    # predict_batch chi goi model.predict 1 lan (Chot 3 latency guardrail)
    assert src_batch_full.count(".predict(") == 1 or inspect.getsource(YOLODetector.predict_batch).count("_predict_with_oom_fallback") == 1, "predict_batch phai goi GPU 1 lan duy nhat"
    assert "imgsz" in src_batch and "640" in src_batch


def test_yolo_detector_class_map_matches_weights():
    from ultralytics import YOLO
    m = YOLO(str(LOCAL_BASE_PT))
    assert m.names[0] == "open_circuit" and m.names[5] == "missing_hole"
    from src.models.yolo_detector import CLASS_NAMES
    for i in range(6):
        assert CLASS_NAMES[i] == m.names[i]


def test_spatial_center_gating_r80_with_origin():
    from src.models.yolo_detector import YOLODetector
    # crop origin (100,100); box tam crop (320,320) -> global (420,420)
    dets = [{"bbox": [300, 300, 340, 340], "class_id": 0, "class_name": "open_circuit", "conf": 0.5},
            {"bbox": [0, 0, 20, 20], "class_id": 1, "class_name": "short", "conf": 0.9}]
    kept = YOLODetector.spatial_center_gating(dets, center_xy=(420, 420),
                                              radius=80.0, crop_origin=(100, 100))
    assert len(kept) == 1 and kept[0]["class_id"] == 0, "chi giu box tam trong R=80"
    # box xa tam -> drop du conf cao (triet FP ria crop)
    far = YOLODetector.spatial_center_gating(dets, center_xy=(900, 900),
                                             radius=80.0, crop_origin=(100, 100))
    assert far == []


def test_no_patchcore_sahi_no_ecc_in_branchA():
    import ast as _ast
    for p in (PROJ / "src").rglob("*.py"):
        t = p.read_text(encoding="utf-8", errors="ignore")
        assert "patchcore" not in t.lower(), f"PatchCore lot vao {p}"
        assert "sahi" not in t.lower(), f"SAHI lot vao {p}"
        # Cam CALL findTransformECC (docstring canh bao inputMask/CV_8UC1 duoc phep,
        # giong test_day3). Khong fail vi docstring.
        assert "findTransformECC(" not in t, f"ECC call trong {p}"
        tree = _ast.parse(t)
        for n in _ast.walk(tree):
            if isinstance(n, _ast.Call):
                f = n.func
                name = f.attr if isinstance(f, _ast.Attribute) else (f.id if isinstance(f, _ast.Name) else "")
                assert name != "findTransformECC", f"ECC call AST trong {p}"


def test_h_inv_sigma_floor_logic_pending_day34_5():
    # Cap nhat Day5 (Verifier): Nhanh B + logic 4 muc + fusion DA ton tai.
    # Audit that: test_day3_branchB.py (H_inv/sigma_floor),
    # test_day5_ensemble_logic.py (logic 4 muc + fusion). Placeholder PENDING
    # nay duoc giu lai de dam bao file ton tai (khong con assert "not exists").
    assert (PROJ / "src" / "reference" / "board_id_and_align.py").exists()
    assert (PROJ / "src" / "reference" / "tolerance_diff.py").exists()
    assert (PROJ / "src" / "decision" / "logic.py").exists()
    assert (PROJ / "src" / "models" / "ensemble.py").exists()
