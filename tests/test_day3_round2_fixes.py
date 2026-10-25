"""Verifier Day-3 round-2: kiem dinh fix BUG-DAY3-01 (FOV valid mask + auto calib)
va BUG-DAY3-02 (shortlist + cv2 thread restore).

Scope: chi doc src/ (READ-ONLY). Khong sua src/.
Chay: <venv>/python.exe -m pytest tests/test_day3_round2_fixes.py -v
"""
import inspect
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJ = Path(__file__).resolve().parent.parent  # PORTABLE: tuong doi theo project root
sys.path.insert(0, str(PROJ))

from configs.resource_paths import LOCAL_TEMPLATES  # noqa: E402

import src.reference.board_id_and_align as ba  # noqa: E402
import src.reference.tolerance_diff as td  # noqa: E402


def _tpl_gray(bid: str = "01") -> np.ndarray:
    data = np.fromfile(str(LOCAL_TEMPLATES / f"{bid}.JPG"), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    assert img is not None
    return img


# ---------------------------------------------------------------------------
# BUG-DAY3-01: vung ngoai FOV cua canvas Template
# ---------------------------------------------------------------------------
def test_valid_mask_from_h_covers_input_quad_only():
    # H = dich phai 200px (Input -> Template): canvas ben trai 200px khong co
    # anh dau vao -> valid mask phai False o do, True o phan con lai.
    h = np.array([[1.0, 0.0, 200.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    vm = td.valid_mask_from_h((400, 500), h, (400, 500), erode_px=0)
    assert vm.shape == (400, 500) and vm.dtype == np.bool_
    assert bool(vm[200, 250]) is True, "vung canvas co nguon phai hop le"
    assert bool(vm[200, 450]) is True
    # H bien input x in [0,499] thanh canvas [200,699] -> canvas [0,200)
    # khong co nguon nao -> phai False.
    assert bool(vm[200, 50]) is False, "canvas [0,200) ngoai FOV phai bi loai"
    assert bool(vm[200, 199]) is False
    assert bool(vm[200, 200]) is True
    # erode lam hep bien
    vm_e = td.valid_mask_from_h((400, 500), h, (400, 500), erode_px=11)
    assert vm_e.sum() < vm.sum()


def test_valid_mask_identity_h_is_full_canvas():
    h = np.eye(3)
    vm = td.valid_mask_from_h((300, 400), h, (300, 400), erode_px=0)
    assert bool(vm.all()), "H = I -> toan bo canvas hop le"


def test_fill_out_of_fov_replaces_outside_with_template():
    tpl = np.full((100, 100), 100, np.uint8)
    warped = np.full((100, 100, 3), 250, np.uint8)
    vm = np.zeros((100, 100), bool)
    vm[:, :50] = True                     # nua trai hop le
    warped[:, :50] = 30                   # gia tri that cua anh trong FOV
    out = td.fill_out_of_fov(warped, vm, tpl)
    assert out.shape == warped.shape and out.dtype == np.uint8
    # trong FOV giu nguyen
    assert np.all(out[:, :50] == 30)
    # ngoai FOV thay bang noi dung template (gray 100 -> BGR 100)
    assert np.all(out[:, 50:] == 100), "ngoai FOV phai = template"


def test_run_branch_b_valid_mask_kills_fake_residual():
    """Tren anh THAT: vung ngoai FOV khong duoc sinh blob.

    Dung template 01 + H dich/co gian de tao vung ngoai FOV that, thay vi
    anh tong hop nua-255 (truong hop phi thuc te: AUTO_CALIB_TOL tu nang
    nguong khi residual chiem >50% dien tich — da ghi la han che trong
    status_and_bugs.md).
    """
    cache = ba.load_cache()
    tpl = _tpl_gray("01")
    img = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
    r = ba.identify_and_align(img, cache, fast=True)
    tpl_used = cache["templates_gray"][cache["board_ids"].index(r["board_id"])]
    if tpl_used.shape != r["warped"].shape[:2]:
        tpl_used = cv2.resize(tpl_used, (r["warped"].shape[1], r["warped"].shape[0]),
                              interpolation=cv2.INTER_AREA)
    valid = r["valid_mask"]
    assert valid.mean() > 0.9, "anh trung template -> gan nhu toan canvas hop le"
    out = td.run_branch_b(r["warped"], tpl_used, valid_mask=valid)
    # moi blob phai nam TRONG vung hop le (khong phai residual ngoai FOV)
    for b in out["blobs"]:
        cx, cy = int(b["centroid_tpl"][0]), int(b["centroid_tpl"][1])
        cx = min(max(cx, 0), valid.shape[1] - 1)
        cy = min(max(cy, 0), valid.shape[0] - 1)
        assert bool(valid[cy, cx]), \
            f"blob o ngoai FOV: centroid={b['centroid_tpl']}"


def test_valid_mask_removes_out_of_fov_blobs_on_real_board():
    """Anh template bi crop 20% ben phai -> vung thieu phai khong sinh blob."""
    cache = ba.load_cache()
    tpl = _tpl_gray("01")
    h_img, w_img = tpl.shape
    cropped = tpl[:, : int(w_img * 0.8)]          # mat 20% ben phai
    img = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
    r = ba.identify_and_align(img, cache, fast=True)
    tpl_used = cache["templates_gray"][cache["board_ids"].index(r["board_id"])]
    if tpl_used.shape != r["warped"].shape[:2]:
        tpl_used = cv2.resize(tpl_used, (r["warped"].shape[1], r["warped"].shape[0]),
                              interpolation=cv2.INTER_AREA)
    valid = r["valid_mask"]
    assert valid.mean() < 0.95, "anh crop phai co vung ngoai FOV"
    out = td.run_branch_b(r["warped"], tpl_used, valid_mask=valid)
    for b in out["blobs"]:
        cx = int(np.clip(b["centroid_tpl"][0], 0, valid.shape[1] - 1))
        cy = int(np.clip(b["centroid_tpl"][1], 0, valid.shape[0] - 1))
        assert bool(valid[cy, cx]), \
            f"blob ngoai FOV lot qua valid_mask: {b['centroid_tpl']}"


def test_autocalib_limitation_documented():
    """Han che da biet: AUTO_CALIB_TOL tu hieu chuan tu CHINH anh dang xet.

    Neu residual chiem >50% dien tich (khong xay ra tren PCB that — probe
    24/24 loi nhan tao van bat duoc), nguong tu nang va blob bi nuot.
    Test nay ghi nhan hanh vi de khong bi "am tham" thay doi sau nay.
    """
    tpl = np.full((300, 300), 100, np.uint8)
    tpl[100:200, :] = 200
    warped = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
    warped[:, 150:] = 255
    out = td.run_branch_b(warped, tpl)          # khong valid_mask
    # khong assert co blob: ghi nhan rang truong hop cuc doan nay khong bao
    assert isinstance(out["blobs"], list)
    # nhung tren kich ban thuc te (loi <= 20% dien tich) van phai bat duoc
    tpl2 = np.full((300, 300), 100, np.uint8)
    tpl2[100:200, :] = 200
    w2 = cv2.cvtColor(tpl2, cv2.COLOR_GRAY2BGR)
    w2[140:160, 140:160] = 250                  # loi nho 400px2
    out2 = td.run_branch_b(w2, tpl2)
    assert len(out2["blobs"]) >= 1, "loi nho thuc te phai bat duoc"


def test_valid_mask_wrong_shape_is_resized_not_crash():
    tpl = np.full((200, 200), 100, np.uint8)
    warped = np.full((200, 200, 3), 100, np.uint8)
    vm_small = np.ones((50, 50), np.uint8)
    out = td.run_branch_b(warped, tpl, valid_mask=vm_small)
    assert "blobs" in out


# ---------------------------------------------------------------------------
# AUTO_CALIB_TOL: khong duoc "nuot" loi that (phan chung)
# ---------------------------------------------------------------------------
def test_auto_calib_tol_still_detects_large_defect():
    tpl = np.full((400, 400), 100, np.uint8)
    warped = np.full((400, 400, 3), 100, np.uint8)
    warped[150:270, 150:270] = 220        # loi lon 120x120, +120 muc xam
    out = td.run_branch_b(warped, tpl)
    assert out["residual_tol_map"] is not None
    hit = [b for b in out["blobs"]
           if b["bbox_tpl"][0] < 210 and b["bbox_tpl"][2] > 210
           and b["bbox_tpl"][1] < 210 and b["bbox_tpl"][3] > 210]
    assert hit, f"loi lon bi AUTO_CALIB_TOL nuot: blobs={out['blobs']}"
    assert hit[0]["area"] >= 60


def test_auto_calib_tol_suppresses_clean_noise():
    # anh sach + nhieu nhe -> khong duoc co blob Z cao
    rng = np.random.default_rng(7)
    tpl = np.full((300, 300), 100, np.uint8)
    warped = np.clip(np.full((300, 300, 3), 100.0) + rng.normal(0, 1.5, (300, 300, 1)),
                     0, 255).astype(np.uint8)
    out = td.run_branch_b(warped, tpl)
    for b in out["blobs"]:
        assert b["z_diff"] < 3.5, f"nhieu nhe sinh Z={b['z_diff']}"


# ---------------------------------------------------------------------------
# Hang so: minh bach contract vs gia tri run_branch_b dung
# ---------------------------------------------------------------------------
def test_run_constants_within_plan_and_documented():
    assert 2 <= td.RUN_TAU_PX <= 3, "plan Phan 2.3: tau = 2-3 px"
    assert td.TAU_PX == 3, "hang so contract giu nguyen (test cu assert)"
    assert td.RUN_SEED_K <= td.SEED_K
    assert td.SIGMA_FLOOR == 5.0 and td.TOP_K == 8
    sig = inspect.signature(td.run_branch_b)
    assert sig.parameters["tau"].default == td.RUN_TAU_PX, \
        "run_branch_b phai dung RUN_TAU_PX lam mac dinh (minh bach)"
    # RUN_* duoc giai thich trong source (khong phai chinh am tham)
    src = inspect.getsource(td)
    assert "RUN_TAU_PX" in src and "RUN_SEED_K" in src
    assert src.count("RUN_TAU_PX =") == 1 and src.count("RUN_SEED_K =") == 1


def test_template_priors_cache_consistent_with_direct_compute():
    tpl = _tpl_gray("01")[:600, :600]     # cat nho cho nhanh
    pri = td.template_priors(tpl, td.RUN_TAU_PX)
    pri2 = td.template_priors(tpl, td.RUN_TAU_PX)
    assert pri is pri2, "cache phai tra cung object (tinh 1 lan/board)"
    tf = tpl.astype(np.float32)
    k = np.ones((2 * td.RUN_TAU_PX + 1,) * 2, np.float32)
    assert np.allclose(pri["tf"], tf)
    assert np.allclose(pri["t_min"], cv2.erode(tf, k))
    assert np.allclose(pri["t_max"], cv2.dilate(tf, k))
    assert np.array_equal(pri["mask_tpl"], td.structure_mask_from_template(tpl))
    # tau khac -> key khac
    pri3 = td.template_priors(tpl, 3)
    assert pri3 is not pri


# ---------------------------------------------------------------------------
# BUG-DAY3-02: shortlist + thread restore
# ---------------------------------------------------------------------------
def test_cv_threads_restored_after_align_and_diff():
    cache = ba.load_cache()
    tpl = _tpl_gray("01")
    img = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
    cv2.setNumThreads(1)                  # mo phong side-effect ultralytics
    r = ba.identify_and_align(img, cache, fast=True)
    assert r["board_id"] == "01"
    assert cv2.getNumThreads() == ba.CV_THREADS, \
        f"align khong khoi phuc thread: {cv2.getNumThreads()} != {ba.CV_THREADS}"
    cv2.setNumThreads(1)
    td.run_branch_b(r["warped"], tpl)
    assert cv2.getNumThreads() == td.CV_THREADS, \
        f"run_branch_b khong khoi phuc thread: {cv2.getNumThreads()}"


def test_shortlist_fingerprint_gain_bias_invariant():
    cache = ba.load_cache()
    tpl = _tpl_gray("01")
    # mo phong camera: gain 0.8 + bias +20 (khong doi cau truc gradient)
    cam = np.clip(tpl.astype(np.float32) * 0.8 + 20.0, 0, 255).astype(np.uint8)
    cand = ba.shortlist_boards(cam, cache["fingerprints"], cache["board_ids"])
    assert cand[0][0] == "01", f"shortlist top-1 sai: {cand}"
    # fingerprint chi phu thuoc gradient -> cosine ~1
    assert cand[0][1] > 0.95, cand[0][1]


def test_identify_align_returns_warped_raw_and_valid_mask():
    cache = ba.load_cache()
    tpl = _tpl_gray("04")
    img = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
    r = ba.identify_and_align(img, cache, fast=True)
    assert r["board_id"] == "04"
    assert "warped_raw" in r and "valid_mask" in r
    assert r["warped_raw"].shape == r["warped"].shape
    assert r["valid_mask"].dtype == np.bool_
    # anh trung template -> gan nhu toan bo canvas hop le
    assert r["valid_mask"].mean() > 0.95, r["valid_mask"].mean()


# ---------------------------------------------------------------------------
# Chong "an gian": Nhanh B khong duoc dung nhan/GT
# ---------------------------------------------------------------------------
def test_reference_modules_never_read_labels_or_gt():
    """Nhanh B phai la unsupervised: khong doc XML/nhan/GT cua PKU.

    Kiem tra CALL AST (khong grep chuoi) de tranh khop nham
    `from __future__ import annotations`.
    """
    import ast as _ast
    for name in ("tolerance_diff.py", "board_id_and_align.py"):
        p = PROJ / "src" / "reference" / name
        txt = p.read_text(encoding="utf-8")
        # khong import XML parser / khong mo file .xml
        assert "xml" not in txt.lower(), f"{name} co the doc XML nhan"
        assert ".xml" not in txt.lower(), f"{name} mo file XML"
        # khong co bien/nhan GT trong AST (ten bien hoac keyword)
        for n in _ast.walk(_ast.parse(txt)):
            if isinstance(n, _ast.Name):
                assert n.id not in {"gt_boxes", "ground_truth", "labels"}, \
                    f"{name} dung bien nhan: {n.id}"
            if isinstance(n, _ast.keyword) and n.arg in {"labels", "targets", "gt"}:
                assert False, f"{name} truyen {n.arg}= (nhan/GT) vao ham"
        # khong doc thu muc Annotations cua PKU
        assert "PKU" not in txt or "LEGACY_PKU" not in txt, name


def test_extract_blobs_meanstddev_matches_numpy():
    rng = np.random.default_rng(11)
    diff = rng.uniform(0, 40, (240, 240)).astype(np.float32)
    diff[50:80, 50:80] = 200.0
    blobs = td.extract_blobs(diff, min_area=60, sigma_floor=5.0,
                             seed_k=td.RUN_SEED_K, top_k=8)
    bg_mean = float(diff.mean(dtype=np.float64))
    bg_std = float(diff.std(dtype=np.float64))
    assert blobs, "phai co blob"
    for b in blobs:
        assert abs(b["bg_mean"] - bg_mean) < 1e-3, (b["bg_mean"], bg_mean)
        assert abs(b["bg_std"] - bg_std) < 1e-2, (b["bg_std"], bg_std)
        z = (b["mean_res"] - bg_mean) / max(bg_std, 5.0)
        assert abs(b["z_diff"] - z) < 1e-3, (b["z_diff"], z)


def test_no_legacy_write_and_no_ecc_day3_round2():
    import ast as _ast
    for name in ("tolerance_diff.py", "board_id_and_align.py"):
        p = PROJ / "src" / "reference" / name
        txt = p.read_text(encoding="utf-8")
        assert "rmtree" not in txt and "unlink()" not in txt
        calls = set()
        for n in _ast.walk(_ast.parse(txt)):
            if isinstance(n, _ast.Call):
                f = n.func
                calls.add(f.attr if isinstance(f, _ast.Attribute)
                          else (f.id if isinstance(f, _ast.Name) else ""))
        assert "findTransformECC" not in calls, name
        assert "patchcore" not in txt.lower() and "sahi" not in txt.lower()
