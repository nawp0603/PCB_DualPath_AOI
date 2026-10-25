"""Verifier Day-3 strict tests: Branch B (board_id_and_align + tolerance_diff).

Scope: chi doc src/ (READ-ONLY), khong sua src/.
Chay: <venv>/python.exe -m pytest tests/test_day3_branchB.py -v
"""
import ast
import importlib.util
import inspect
import sys
from pathlib import Path

import cv2
import numpy as np

PROJ = Path(__file__).resolve().parent.parent  # PORTABLE: tuong doi theo project root
sys.path.insert(0, str(PROJ))

from configs.resource_paths import (  # noqa: E402
    LEGACY_PKU_ROOT, LOCAL_TEMPLATES, LEGACY_PROJECT_DIR, NEW_PROJECT_DIR, assert_writable_ok,
)

import src.reference.board_id_and_align as ba  # noqa: E402
import src.reference.tolerance_diff as td  # noqa: E402


def _first_defect_of_board(bid: str):
    import xml.etree.ElementTree as ET
    from configs.resource_paths import DEMO_ROOT
    roots = [LEGACY_PKU_ROOT, DEMO_ROOT]
    for rpath in roots:
        ann_root = rpath / "Annotations"
        if not ann_root.exists():
            continue
        for ann_dir in sorted(p for p in ann_root.iterdir() if p.is_dir()):
            for xp in sorted(ann_dir.glob("*_annotation.xml")):
                try:
                    r = ET.parse(xp).getroot()
                except Exception:
                    continue
                node = r.find("filename")
                if node is None or not node.text:
                    continue
                fn = node.text.strip()
                if fn.split("_")[0].zfill(2) != bid:
                    continue
                stem = xp.stem
                if stem.endswith("_annotation"):
                    stem = stem[: -len("_annotation")]
                for cand in (stem + ".jpg", stem + ".JPG", fn):
                    q = rpath / "images" / ann_dir.name / cand
                    if q.exists():
                        data = np.fromfile(str(q), dtype=np.uint8)
                        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                        if img is not None:
                            return img, str(q)
    import pytest
    pytest.skip(f"khong tim thay anh board {bid} (LEGACY_PKU_ROOT khong co trong Clean-Room)")


def test_branchB_constants_match_plan():
    assert ba.N_FEATURES == 8000
    assert ba.RATIO_TEST == 0.75
    assert ba.RANSAC_THRESH == 5.0
    assert ba.MIN_INLIERS_BOARD_ID == 30
    assert ba.TOP_PER_TEMPLATE_MATCHES == 4000
    assert td.TAU_PX == 3 and 2 <= td.TAU_PX <= 3
    assert td.SMOOTH_WIN == 151
    assert (td.GAIN_LO, td.GAIN_HI) == (0.5, 2.0)
    assert td.STRUCT_DILATE_PX == 7
    assert td.GRAD_EDGE_THRESH == 30.0
    assert td.SEED_K == 2.5
    assert td.MIN_AREA_PX2 == 60
    assert 4.0 <= td.SIGMA_FLOOR <= 6.0, td.SIGMA_FLOOR
    assert td.TOP_K == 8  # Chot 3: TOP_K 6 hoac 8


def test_orb_cache_exists_and_complete():
    p = PROJ / "assets" / "templates" / "orb_cache.npz"
    assert p.exists() and p.stat().st_size > 1_000_000, "cache ~2.6MB"
    z = np.load(str(p), allow_pickle=False)
    bids = [str(b) for b in z["board_ids"]]
    assert bids == ["01", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    n_kp = [int(v) for v in z["n_kp"]]
    assert n_kp[4] < 8000 and all(v == 8000 for i, v in enumerate(n_kp) if i != 4), n_kp
    assert z["keypoints"].shape[0] == 10 and z["descriptors"].shape[0] == 10
    # cache nam trong project (zero-touch)
    assert str(p.resolve()).startswith(str(NEW_PROJECT_DIR.resolve()))


def test_match_template_lowe_sorted_capped():
    rng = np.random.default_rng(0)
    a = (rng.integers(0, 256, size=(50, 32))).astype(np.uint8)
    b = a.copy()
    m = ba.match_template(a, b)
    assert len(m) > 0
    dists = [x.distance for x in m]
    assert dists == sorted(dists), "phai sort theo distance"
    assert len(m) <= ba.TOP_PER_TEMPLATE_MATCHES
    assert ba.match_template(None, b) == []
    assert ba.match_template(a, None) == []


def test_estimate_h_identity_and_degenerate():
    kp = np.float32([[0, 0], [100, 0], [100, 100], [0, 100], [50, 50]])
    import types
    ms = []
    for i in range(5):
        d = cv2.DMatch()
        d.queryIdx, d.trainIdx, d.distance = i, i, 1.0
        ms.append(d)
    h, inl = ba.estimate_h(kp, kp, ms)
    assert h is not None and h.shape == (3, 3) and h.dtype == np.float64
    assert inl >= 4
    assert np.allclose(h / h[2, 2], np.eye(3), atol=1e-4), h
    h2, inl2 = ba.estimate_h(kp, kp, ms[:3])
    assert h2 is None and inl2 == 0, " <4 matches phai tra (None,0)"


def test_project_points_roundtrip_zero_error():
    # H tong hop: shift+rotate+scale nhe; H_inv nguoc lai phai roundtrip ~0px
    ang, dx, dy = 2.0, 25.0, -17.0
    cx, cy = 1500.0, 1200.0
    m = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    m[0, 2] += dx
    m[1, 2] += dy
    h = np.vstack([m, [0.0, 0.0, 1.0]])
    h_inv = np.linalg.inv(h)
    pts = np.float32([[100, 200], [1500, 1200], [2900, 1500], [10, 10]])
    fwd = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), h).reshape(-1, 2)
    back = ba.project_points(fwd, h_inv)
    err = np.linalg.norm(back - pts, axis=1).max()
    assert err < 1e-3, f"roundtrip err={err}"
    # Chot 1: project_points phai dung perspectiveTransform + H_inv
    src = inspect.getsource(ba.project_points)
    assert "perspectiveTransform" in src and "h_inv" in src


def test_identify_align_single_image_board01_full():
    img, _ = _first_defect_of_board("01")
    cache = ba.load_cache()
    r = ba.identify_and_align(img, cache)  # full mode mac dinh
    assert r["board_id"] == "01", r["scores"]
    assert r["inliers"] >= 30
    assert r["H"] is not None and r["H_inv"] is not None
    assert r["H"].shape == (3, 3)
    ident = r["H"] @ r["H_inv"]
    ident = ident / ident[2, 2]
    assert np.allclose(ident, np.eye(3), atol=1e-4), "H @ H_inv phai ~ I"
    th, tw = r["tpl_shape"]
    assert r["warped"].shape[:2] == (th, tw)
    assert r["mode"] == "full"
    # roundtrip tren anh that: template -> input -> template
    c = np.float32([[r['warped'].shape[1] / 2, r['warped'].shape[0] / 2]])
    to_input = ba.project_points(c, r["H_inv"])
    back = cv2.perspectiveTransform(
        np.asarray(to_input, np.float32).reshape(-1, 1, 2), r["H"]).reshape(-1, 2)
    assert float(np.linalg.norm(back - c)) < 1.0


def test_tolerance_diff_identical_zero_tau_behavior():
    rng = np.random.default_rng(1)
    tpl = (rng.integers(60, 200, size=(200, 200))).astype(np.uint8)
    tf = tpl.astype(np.float32)
    out_same = td.tolerance_diff(tf, tf, tau=3,
                                 structure_mask=np.ones_like(tpl, bool))
    assert float(np.abs(out_same).max()) == 0.0, "anh giong het -> res=0"
    # dich 1px trong tau=3 -> res ~0; dich 10px vuot tau -> res >0
    shifted1 = np.roll(tf, 1, axis=1)
    r1 = td.tolerance_diff(shifted1, tf, tau=3,
                           structure_mask=np.ones_like(tpl, bool))
    shifted10 = np.roll(tf, 10, axis=1)
    r10 = td.tolerance_diff(shifted10, tf, tau=3,
                            structure_mask=np.ones_like(tpl, bool))
    assert float(r1.mean()) < float(r10.mean()), "tau phai dung sai lech nho"
    # khong vong lap Python: dung erode/dilate
    src = inspect.getsource(td.tolerance_diff)
    assert "erode" in src and "dilate" in src


def test_structure_mask_superset_strict_copper():
    img, _ = _first_defect_of_board("01")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    tpl_p = LOCAL_TEMPLATES / "01.JPG"
    data = np.fromfile(str(tpl_p), dtype=np.uint8)
    tpl = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    tpl_r = cv2.resize(tpl, (w, h), interpolation=cv2.INTER_AREA)
    strict = td.copper_mask_from_template(tpl_r)
    struct = td.structure_mask_from_template(tpl_r)
    # structure = dilate7(copper UNION edge) -> phu gan het strict (cho phep lech bien 2%)
    inter = np.logical_and(strict, struct).sum()
    assert inter / max(strict.sum(), 1) > 0.98, "structure phai bao het copper"


def test_extract_blobs_sigma_floor_blocks_false_Z():
    # anh sach: bg_std ~0.3, hat bui 2 muc xam; khong floor -> Z>5 bao gia
    rng = np.random.default_rng(2)
    base = np.full((300, 300), 100.0, np.float32) + rng.normal(0, 0.3, (300, 300))
    base[150, 150] += 2.0  # hat bui
    blobs = td.extract_blobs(np.maximum(base - 100.0, 0), min_area=1,
                             sigma_floor=5.0, seed_k=2.5, top_k=8)
    for b in blobs:
        assert b["z_diff"] < 3.5, f"GOOD sach Z={b['z_diff']} phai <3.5 (floor)"
        assert b["sigma_floor"] == 5.0
    # cong thuc Chot 2 co mat trong source
    src = inspect.getsource(td.extract_blobs)
    assert "max(bg_std" in src and "sigma_floor" in src


def test_extract_blobs_topk_sorted_min_area():
    diff = np.zeros((200, 200), np.float32)
    diff[10:20, 10:20] = 50.0   # area 100
    diff[50:70, 50:70] = 30.0   # area 400 (mean thap hon)
    diff[150:152, 150:152] = 200.0  # area 4 <60 -> loai
    blobs = td.extract_blobs(diff, min_area=60, sigma_floor=5.0,
                             seed_k=0.5, top_k=8)
    assert all(b["area"] >= 60 for b in blobs)
    zs = [b["z_diff"] for b in blobs]
    assert zs == sorted(zs, reverse=True), "phai sort Z giam"
    assert len(blobs) <= 8
    many = np.zeros((400, 400), np.float32)
    for i in range(12):
        many[i * 30:i * 30 + 15, 10:25] = 40.0 + i  # 12 blob area 225
    capped = td.extract_blobs(many, min_area=60, sigma_floor=5.0,
                              seed_k=0.5, top_k=8)
    assert len(capped) == 8, "TOP_K=8 phai cat"


def test_run_branch_b_end_to_end_fields():
    img, _ = _first_defect_of_board("01")
    cache = ba.load_cache()
    r = ba.identify_and_align(img, cache)
    data = np.fromfile(str(LOCAL_TEMPLATES / "01.JPG"), dtype=np.uint8)
    tpl = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if tpl.shape != r["warped"].shape[:2]:
        tpl = cv2.resize(tpl, (r["warped"].shape[1], r["warped"].shape[0]),
                         interpolation=cv2.INTER_AREA)
    out = td.run_branch_b(r["warped"], tpl)
    assert {"blobs", "diff_map", "structure_mask", "warped_corr", "tpl_gray"} <= set(out)
    assert len(out["blobs"]) <= 8
    for b in out["blobs"]:
        assert {"centroid_tpl", "area", "bbox_tpl", "z_diff",
                "mean_res", "seed_thr", "bg_mean", "bg_std",
                "sigma_floor"} <= set(b)
        assert b["area"] >= 60 and b["z_diff"] >= 0


def _call_names_of(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute):
                calls.add(f.attr)
            elif isinstance(f, ast.Name):
                calls.add(f.id)
    return calls


def test_no_ecc_call_no_patchcore_sahi_in_src():
    for p in PROJ.glob("src/**/*.py"):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        assert "patchcore" not in txt.lower(), f"PatchCore lot vao {p}"
        assert "sahi" not in txt.lower(), f"SAHI lot vao {p}"
        calls = _call_names_of(p)
        assert "findTransformECC" not in calls, f"ECC call trong {p}"
        assert "findTransformECC(" not in txt, f"ECC call string trong {p}"
    # Docstring duoc phep NHAC inputMask/CV_8UC1 de canh bao crash OpenCV 4.11
    # (verifier_rules: neu dung ECC thi mask phai la None/CV_8UC1). Chi fail
    # khi co call ECC that kem mask float32 — da assert khong co call o tren.
    assert "findTransformECC(" not in (PROJ / "src" / "reference" / "board_id_and_align.py").read_text(encoding="utf-8")
    assert "findTransformECC(" not in (PROJ / "src" / "reference" / "tolerance_diff.py").read_text(encoding="utf-8")


def test_yolo_day2_fixes_still_present():
    from src.models.yolo_detector import YOLODetector
    import src.models.yolo_detector as yd
    assert str(inspect.signature(YOLODetector.__init__)).count("device") >= 1
    d = YOLODetector.__init__.__defaults__ if hasattr(YOLODetector.__init__, "__defaults__") else None  # noqa
    src_g = inspect.getsource(YOLODetector.spatial_center_gating)
    assert "_box_iou" in src_g and "require_overlap" in src_g
    # require_overlap=False chi xet R; True giu them box giao diff_box
    dets = [{"bbox": [300, 300, 340, 340], "class_id": 0,
             "class_name": "open_circuit", "conf": 0.5}]
    kept_f = YOLODetector.spatial_center_gating(
        dets, center_xy=(900, 900), radius=80.0, crop_origin=(100, 100),
        require_overlap=False)
    assert kept_f == []
    kept_t = YOLODetector.spatial_center_gating(
        dets, center_xy=(900, 900), radius=80.0, crop_origin=(100, 100),
        require_overlap=True, diff_box=[400.0, 400.0, 450.0, 450.0])
    assert len(kept_t) == 1
    assert yd.DEFAULT_IMGSZ == 3072 and yd.DEFAULT_CONF == 0.20
    assert "rect=True" in inspect.getsource(YOLODetector.predict)
    src_batch_full = inspect.getsource(YOLODetector.predict_batch)

    assert src_batch_full.count(".predict(") == 1 or inspect.getsource(YOLODetector.predict_batch).count("_predict_with_oom_fallback") == 1


def test_terminology_zero_defect_not_unseen():
    bad = []
    for p in PROJ.glob("src/**/*.py"):
        if "unseen boards" in p.read_text(encoding="utf-8").lower():
            bad.append(str(p))
    assert not bad, f"cam cum 'unseen boards': {bad}"
    found = any("Zero-defect-training" in p.read_text(encoding="utf-8")
                for p in PROJ.glob("src/**/*.py"))
    assert found, "thieu thuat ngu Zero-defect-training"


def test_zero_touch_writes_guarded_day3():
    # board_id_and_align.py co build_cache ghi .npz -> BAT BUOC guard
    txt_ba = (PROJ / "src" / "reference" / "board_id_and_align.py").read_text(encoding="utf-8")
    assert "assert_writable_ok" in txt_ba
    assert "rmtree" not in txt_ba and "unlink()" not in txt_ba
    # tolerance_diff.py thuan tuy (khong ghi file) -> KHONG bat buoc guard;
    # chi cam xoa/ghi legacy
    txt_td = (PROJ / "src" / "reference" / "tolerance_diff.py").read_text(encoding="utf-8")
    assert "rmtree" not in txt_td and "unlink()" not in txt_td, "tolerance_diff.py"
    assert "LEGACY_PROJECT_DIR" not in txt_td and "Test_github" not in txt_td
    import pytest
    with pytest.raises(PermissionError):
        assert_writable_ok(LEGACY_PROJECT_DIR / "data" / "x.json")  # PORTABLE: van ngoai project -> phai raise
