"""Ngay 6 — UI Streamlit demo Dual-Path Industrial AOI (3 khung hinh).

3 khung hinh (plan.md Phan 5, Ngay 6):
  (1) Anh dau vao + BBox YOLO (da fuse, toa do anh goc).
  (2) Diff heatmap Nhanh B (khong gian anh goc).
  (3) Ket luan 4 muc: FAIL / REWORK / REVIEW / PASS.

Pipeline dung Y NGUYEN trong `scripts/evaluate.py::run_dualpath`:
  Nhanh B (identify_and_align fast + run_branch_b) -> crop 640 quanh tam blob
  qua H_inv (Chot 1) -> Nhanh A predict_batch 1 LAN GPU duy nhat (Chot 3) +
  gating R=80 -> Box Fusion IoU>=0.3, S_final=min(1.0,S_A+0.15)
  -> quy tac 4 muc (logic.decide). Z_diff co sigma_floor (Chot 2).

Khong co baseline ablation trong luong quyet dinh chinh
(chi Nhanh A + Nhanh B + fusion; 2 baseline doi chieu nam o evaluate.py).
Thuat ngu: 01/04 la "Zero-defect-training Boards" (co Golden Sample nhung
chua co du lieu loi) — khong goi la "unseen".

Chay portable (may moi, khong can o D:):
  2_run_ui_demo.bat
  # hoac thu cong: .venv\Scripts\python.exe -m streamlit run src/ui/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.resource_paths import (  # noqa: E402
    LOCAL_BASE_PT,
    LOCAL_DAY2_PT,
    LOCAL_TEMPLATES,
    NEW_PROJECT_DIR,
    default_weights,
    find_pku_root as find_pku_root_portable,
)

import src.reference.board_id_and_align as ba  # noqa: E402
import src.reference.tolerance_diff as td  # noqa: E402
from src.models.ensemble import CONF_BOOST, FUSION_IOU, GATE_R, fuse_and_decide  # noqa: E402
from src.models.yolo_detector import (  # noqa: E402
    CENTER_GATE_RADIUS,
    CLASS_NAMES,
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    YOLODetector,
)

# ---------------------------------------------------------------------------
# Hang so UI (khoa voi plan, khong tu y doi nguong quyet dinh o day)
# ---------------------------------------------------------------------------
DAY2_PT = LOCAL_DAY2_PT  # = NEW_PROJECT_DIR / assets/weights/yolo11n_day2.pt
# PORTABLE: chi hien option co file that; mac dinh = default_weights()
# (day2 neu co, khong thi base). May moi luon chay duoc offline.
WEIGHT_OPTIONS: dict[str, Path] = {
    label: p for label, p in {
        "yolo11n_day2.pt (fine-tune 15 epoch, khuyen nghi)": DAY2_PT,
        "yolo11n_base.pt (goc 50 epoch)": LOCAL_BASE_PT,
    }.items() if p.exists()
} or {"yolo11n_base.pt (goc 50 epoch)": LOCAL_BASE_PT}

CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 255),        # open_circuit — do
    1: (0, 165, 255),      # short — cam
    2: (0, 255, 255),      # mouse_bite — vang
    3: (255, 0, 255),      # spur — tim
    4: (255, 100, 0),      # spurious_copper — lam
    5: (0, 255, 0),        # missing_hole — xanh la
}

LEVEL_STYLE: dict[int, tuple[str, str]] = {
    1: ("FAIL", "Mau do — loi nghiem trong (open_circuit / short / missing_hole, S>=0.55). Dung day chuyen, cach ly board."),
    2: ("REWORK", "Mau cam — loi co the sua (mouse_bite / spur / spurious_copper, S>=0.35). Chuyen tram sua chua."),
    3: ("REVIEW", "Mau vang — can ky su xem lai (3a cuu recall Z>=2.0 / 3b open-set Z>=3.5 & Area>=100)."),
    4: ("PASS", "Mau xanh — khong phat hien loi. Cho luu hanh."),
}

# 6 anh dai dien 6 lop + 2 GOOD (duong dan that, da verify ton tai o Ngay 5).
SAMPLE_FILES: list[tuple[str, str]] = [
    ("01 / missing_hole (b1)", "Missing_hole/missing_hole01.jpg"),
    ("01 / mouse_bite (b2)", "Mouse_bite/mouse_bite01.jpg"),
    ("01 / open_circuit (b0)", "Open_circuit/open_circuit01.jpg"),
    ("01 / short (b1)", "Short/short01.jpg"),
    ("01 / spur (b3)", "Spur/spur01.jpg"),
    ("01 / spurious_copper (b4)", "Spurious_copper/spurious_copper01.jpg"),
    ("GOOD 01 (Golden Template)", "__GOOD_01__"),
    ("GOOD 04 (Golden Template)", "__GOOD_04__"),
]


# ---------------------------------------------------------------------------
# IO helpers (Unicode-safe, chi doc)
# ---------------------------------------------------------------------------
def read_bgr(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot decode image: {path}")
    return img


def read_gray(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot decode image: {path}")
    return img


def decode_upload(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Khong doc duoc anh upload (chi nhan JPG/PNG).")
    return img


# Canh ngan toi thieu de ORB+RANSAC on dinh (anh chuan 1586x3034).
INPUT_MIN_SIDE = 640


def validate_input_image(img_bgr: np.ndarray | None) -> str | None:
    """Kiem tra nhanh anh dau vao truoc khi align.

    Tra ve thong bao loi tieng Viet neu anh khong du chuan, None neu OK.
    Anh la (khong phai PCB / qua nho) phai fail RO o day, khong ha nguong
    MIN_INLIERS de "cho qua".
    """
    if img_bgr is None:
        return ("Chua co anh dau vao (giai ma that bai). "
                "Hay upload lai file JPG/PNG hoac chon 1 trong 8 anh mau.")
    if (not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3
            or img_bgr.shape[2] != 3):
        return ("Anh dau vao khong dung dinh dang (can anh mau 3 kenh). "
                "Hay upload file JPG/PNG chup mach PCB.")
    h, w = int(img_bgr.shape[0]), int(img_bgr.shape[1])
    if h <= 0 or w <= 0:
        return "Anh rong (kich thuoc = 0). Hay upload lai file JPG/PNG."
    if min(h, w) < INPUT_MIN_SIDE:
        return (f"Anh qua nho ({w}x{h}, canh ngan {min(h, w)}px < "
                f"{INPUT_MIN_SIDE}px). He thong can anh chup gan full-board "
                "de ORB tim du keypoint. Hay chup lai gan hon hoac thu anh mau board 01.")
    return None


def shortlist_rank_text(gray_u8: np.ndarray, cache: dict, top_n: int = 4) -> str:
    """Chuoi rank van tay gradient top-N de hien khi board-ID that bai."""
    try:
        cand = ba.shortlist_boards(gray_u8, cache["fingerprints"],
                                   cache["board_ids"],
                                   top_k=max(1, int(top_n)))
        return ", ".join(f"{bid} ({cos:.3f})" for bid, cos in cand)
    except Exception as e:
        return f"(khong tinh duoc rank: {type(e).__name__})"


def render_rows_as_markdown(rows: list[dict], empty_text: str) -> None:
    """Ve bang chi tiet bang st.markdown (khong dung st.table).

    Ly do: st.table keo pyarrow doi numpy>=2.0 trong khi venv dung chung
    dang giu numpy 1.26.4 (torch/ultralytics) -> crash ImportError.
    Render markdown thu cong khong phu thuoc pyarrow.
    """
    import streamlit as st
    if not rows:
        st.write(empty_text)
        return
    cols = list(rows[0].keys())

    def cell(v):
        if v is None:
            return "-"
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            return f"{v:.3f}"
        if isinstance(v, list):
            parts = [f"{float(x):.1f}" if isinstance(x, (int, float)) else str(x)
                     for x in v]
            return "[" + ", ".join(parts) + "]"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(cell(r.get(c)) for c in cols) + " |")
    st.markdown("\n".join(lines))


def resolve_sample(rel: str, pku_root: Path | None) -> Path:
    if rel == "__GOOD_01__":
        return LOCAL_TEMPLATES / "01.JPG"
    if rel == "__GOOD_04__":
        return LOCAL_TEMPLATES / "04.JPG"
    # PORTABLE: pku_root (LOCAL -> DEMO bundled -> LEGACY old-machine) truoc,
    # sau do DEMO bundled. May moi khong co legacy van chay duoc.
    from configs.resource_paths import DEMO_ROOT
    roots: list[Path | None] = [pku_root, DEMO_ROOT]
    for cls_dir in ("Missing_hole", "Mouse_bite", "Open_circuit",
                    "Short", "Spur", "Spurious_copper"):
        for root in roots:
            if root is None:
                continue
            try:
                cand = root / "images" / cls_dir / Path(rel).name
            except Exception:
                continue
            if cand.exists():
                return cand
    raise FileNotFoundError(f"Khong tim thay anh mau: {rel}")


def find_pku_root() -> Path | None:
    # PORTABLE: dung chung lookup LOCAL -> DEMO -> LEGACY (old machine).
    # Ghi de ham cu chi biet LOCAL/LEGACY de may moi thay DEMO bundled.
    try:
        return find_pku_root_portable()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pipeline don anh (GIONG HET run_dualpath trong evaluate.py)
# ---------------------------------------------------------------------------
def get_template_gray(board_id: str, like_shape: tuple[int, int]) -> np.ndarray:
    tpl = read_gray(LOCAL_TEMPLATES / f"{board_id}.JPG")
    if tpl.shape != like_shape:
        tpl = cv2.resize(tpl, (like_shape[1], like_shape[0]),
                         interpolation=cv2.INTER_AREA)
    return tpl


def run_dualpath_single(img_bgr: np.ndarray, yolo: YOLODetector,
                        cache: dict, fast: bool = True) -> dict:
    """Chay Dual-Path tren 1 anh BGR uint8. Tra ve dict day du cho UI ve.

    Thu tu: align fast -> run_branch_b (valid_mask) -> project centroid qua
    H_inv (Chot 1) -> crop 640 -> predict_batch 1 call (Chot 3) + gating R=80
    -> fuse_and_decide (IoU>=0.3 boost +0.15) -> logic 4 muc.
    """
    t_total = time.perf_counter()
    h_img, w_img = img_bgr.shape[:2]

    t0 = time.perf_counter()
    r = ba.identify_and_align(img_bgr, cache, fast=fast)
    t_align = (time.perf_counter() - t0) * 1000.0

    tpl = get_template_gray(r["board_id"], r["warped"].shape[:2])

    t0 = time.perf_counter()
    out = td.run_branch_b(r["warped"], tpl,
                          valid_mask=r.get("valid_mask"))
    t_diff = (time.perf_counter() - t0) * 1000.0

    blobs = out["blobs"]
    if blobs:
        cents = ba.project_points(
            np.float32([b["centroid_tpl"] for b in blobs]), r["H_inv"])
    else:
        cents = np.zeros((0, 2), dtype=np.float32)

    crops: list[np.ndarray] = []
    origins: list[tuple[int, int]] = []
    for (px, py) in cents:
        x0 = int(np.clip(float(px) - 320, 0, max(w_img - 640, 0)))
        y0 = int(np.clip(float(py) - 320, 0, max(h_img - 640, 0)))
        x1, y1 = min(x0 + 640, w_img), min(y0 + 640, h_img)
        crop = img_bgr[y0:y1, x0:x1]
        if crop.shape[0] != 640 or crop.shape[1] != 640:
            # Anh upload nho hon 640px: dem nen den cho du 640 (giu hop dong).
            pad = np.zeros((640, 640, 3), np.uint8)
            pad[:crop.shape[0], :crop.shape[1]] = crop
            crop = pad
        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        origins.append((x0, y0))

    t0 = time.perf_counter()
    batch = yolo.predict_batch(crops) if crops else []
    t_roi = (time.perf_counter() - t0) * 1000.0

    yolo_dets_all: list[dict] = []
    kept_per_crop: list[int] = []
    for dets_one, (px, py), (x0, y0) in zip(batch, cents, origins):
        kept = YOLODetector.spatial_center_gating(
            dets_one, center_xy=(float(px), float(py)),
            radius=float(CENTER_GATE_RADIUS),
            crop_origin=(float(x0), float(y0)))
        kept_per_crop.append(len(kept))
        for d in kept:
            # Dich det tu he crop ve he anh goc TRUOC khi fusion
            # (neu quen, IoU luon = 0 -> mat het TP; loi Day 5 da sua).
            dd = dict(d)
            dd["bbox"] = [d["bbox"][0] + x0, d["bbox"][1] + y0,
                          d["bbox"][2] + x0, d["bbox"][3] + y0]
            yolo_dets_all.append(dd)

    res = fuse_and_decide(yolo_dets_all, blobs, r["H_inv"])
    t_total_ms = (time.perf_counter() - t_total) * 1000.0
    return {
        "board_id": r["board_id"], "inliers": r["inliers"],
        "mode": r.get("mode", "fast"),
        "escalated": bool(r.get("shortlist_escalated", False)),
        "H": r["H"], "H_inv": r["H_inv"],
        "warped": r.get("warped"), "warped_raw": r.get("warped_raw"),
        "blobs": blobs, "diff_map": out["diff_map"],
        "structure_mask": out.get("structure_mask"),
        "centroids_input": [(float(x), float(y)) for x, y in cents],
        "origins": origins, "n_crops": len(crops),
        "kept_per_crop": kept_per_crop,
        "yolo_dets": yolo_dets_all,
        "result": res,
        "t_align_ms": t_align, "t_diff_ms": t_diff, "t_roi_ms": t_roi,
        "t_total_ms": t_total_ms,
    }


# ---------------------------------------------------------------------------
# Ve hinh (BGR uint8 -> RGB uint8 cho st.image)
# ---------------------------------------------------------------------------
def draw_boxes_rgb(img_bgr: np.ndarray, items: list[dict]) -> np.ndarray:
    vis = img_bgr.copy()
    for it in items:
        if it.get("conf") is None or it.get("bbox") is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in it["bbox"]]
        cid = it.get("class_id")
        color = CLASS_COLORS_BGR.get(int(cid) if cid is not None else -1,
                                     (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        label = f"{it.get('class_name', '?')} {float(it['conf']):.2f} [{it.get('source', '?')}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        y1t = max(0, y1 - th - 10)
        cv2.rectangle(vis, (x1, y1t), (x1 + tw + 8, y1t + th + 10), color, -1)
        cv2.putText(vis, label, (x1 + 4, y1t + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    # Danh dau tam blob (da qua H_inv) bang vong tron trang.
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def mark_centroids_rgb(img_rgb: np.ndarray, centroids: list[tuple[float, float]],
                       img_shape: tuple[int, int]) -> np.ndarray:
    vis = img_rgb.copy()
    h, w = img_shape[:2]
    for (px, py) in centroids:
        x, y = int(round(px)), int(round(py))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(vis, (x, y), 12, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(vis, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
    return vis


def diff_heatmap_input_space(diff_tpl: np.ndarray, h_inv: np.ndarray,
                             img_shape: tuple[int, int]) -> np.ndarray:
    """Chieu diff_map (khong gian Template) ve khong gian anh goc roi JET."""
    h_img, w_img = int(img_shape[0]), int(img_shape[1])
    try:
        diff_in = cv2.warpPerspective(
            np.asarray(diff_tpl, dtype=np.float32), np.asarray(h_inv),
            (w_img, h_img), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    except cv2.error:
        diff_in = np.zeros((h_img, w_img), np.float32)
    mx = float(np.max(diff_in)) if diff_in.size else 0.0
    if not np.isfinite(mx) or mx <= 1e-9:
        norm = np.zeros_like(diff_in, dtype=np.uint8)
    else:
        norm = np.clip(diff_in / mx * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def overlay_heatmap(img_bgr: np.ndarray, heat_rgb: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    base = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    heat = np.asarray(heat_rgb, dtype=np.float32)
    if heat.shape[:2] != base.shape[:2]:
        heat = cv2.resize(heat, (base.shape[1], base.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    out = base * (1.0 - alpha) + heat * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# So lieu verify san (doc tu results/, co ghi nguon) — chi de tham khao
# ---------------------------------------------------------------------------
def load_verified_numbers() -> dict | None:
    for cand in (NEW_PROJECT_DIR / "results" / "dualpath_day5.json",
                 Path("results/dualpath_day5.json")):
        try:
            if cand.exists():
                import json
                return {"path": str(cand),
                        "data": json.loads(cand.read_text(encoding="utf-8"))}
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------
def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Dual-Path Industrial AOI — Demo",
                       page_icon="🔍", layout="wide")
    st.title("🔍 Dual-Path Industrial AOI — Demo hoi dong")
    st.caption(
        "Nhanh A (YOLO11n Full-Res, rect=True + half=True) + Nhanh B "
        "(Golden-Template Safety-Net: ORB Homography H/H_inv, Tolerance-Aware "
        "Diff, Z_diff voi sigma_floor, TOP-8 batch ROI 1 lan GPU) + "
        "Box Fusion (IoU>=0.3, S_final=min(1.0, S_A+0.15)) + quyet dinh 4 muc. "
        "Tap test 01/04 la **Zero-defect-training Boards** (co Golden Sample, "
        "chua co du lieu loi)."
    )

    # -- Sidebar: cau hinh (giong evaluate.py) --------------------------------
    with st.sidebar:
        st.header("⚙️ Cau hinh")
        wlabel = st.selectbox("Trong so YOLO", list(WEIGHT_OPTIONS.keys()), index=0)
        wpath = WEIGHT_OPTIONS[wlabel]
        if not wpath.exists():
            st.warning(f"Khong thay {wpath.name}, dung ban con lai.")
            wpath = LOCAL_BASE_PT if wpath != LOCAL_BASE_PT else DAY2_PT
        conf = st.slider("conf YOLO", 0.05, 0.60, float(DEFAULT_CONF), 0.05)
        device_opt = st.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
        # PORTABLE: cho phep ha imgsz tren may yeu (VRAM 4GB/CPU).
        # Auto = giu 3072, tu ha xuong 2560/1920 theo VRAM (giu rect+half).
        imgsz_opt = st.selectbox("imgsz Nhanh A", ["Auto (3072, tu ha khi yeu)",
                                                   "3072 (full-res chuan)",
                                                   "2560 (VRAM 4-6GB)",
                                                   "1920 (CPU / VRAM <4GB)"],
                                 index=0)
        imgsz_map = {"Auto (3072, tu ha khi yeu)": None,
                     "3072 (full-res chuan)": 3072,
                     "2560 (VRAM 4-6GB)": 2560,
                     "1920 (CPU / VRAM <4GB)": 1920}
        imgsz_req = imgsz_map[imgsz_opt]
        fast = st.checkbox("Align fast (~45ms)", value=True,
                           help="fast=False chinh xac hon nhung cham (~5.7s).")
        st.divider()
        st.subheader("So lieu verify san")
        ver = load_verified_numbers()
        if ver is not None:
            st.caption(f"Nguon: `{ver['path']}` (output evaluate.py, khong bia).")
            try:
                d = ver["data"]
                for sc in ("standard", "camera"):
                    r = d.get(sc, {}).get("dualpath", {})
                    if r:
                        st.write(f"**{sc}**: AP50={r.get('AP50')} "
                                 f"R={r.get('recall')} P={r.get('precision')} "
                                 f"FAR={r.get('far')} lat={r.get('latency_ms')}ms "
                                 f"levels={r.get('levels')}")
            except Exception as e:  # hien loi, khong che dau
                st.caption(f"(Khong doc duoc so lieu: {e})")
        else:
            st.caption("(Chua co results/dualpath_day5.json — chay evaluate.py truoc.)")
        st.divider()
        st.caption(f"YOLO imgsz={DEFAULT_IMGSZ}, rect=True + half(cuba). "
                   f"Fusion IoU>={FUSION_IOU}, boost +{CONF_BOOST}, gate R={GATE_R}px.")

    # -- Nguon anh: upload hoac mau that --------------------------------------
    pku_root = find_pku_root()
    left, right = st.columns([1, 1])
    with left:
        up = st.file_uploader("📤 Upload anh mach (JPG/PNG)", type=["jpg", "jpeg", "png"])
    with right:
        sample_label = st.selectbox("📁 hoac chon anh mau that (board 01/04)",
                                    [s[0] for s in SAMPLE_FILES], index=2)

    img_bgr: np.ndarray | None = None
    src_name = ""
    if up is not None:
        try:
            img_bgr = decode_upload(up.read())
            src_name = up.name
        except Exception as e:
            st.error(f"Khong doc duoc anh upload: {e}")
            return
    else:
        rel = dict(SAMPLE_FILES)[sample_label]
        try:
            img_bgr = read_bgr(resolve_sample(rel, pku_root))
            src_name = sample_label
        except Exception as e:
            st.error(f"Khong tai duoc anh mau: {e}")
            return

    run = st.button("▶️ Chay Dual-Path", type="primary", use_container_width=True)

    # -- Tai tai nguyen nang 1 lan --------------------------------------------
    @st.cache_resource(show_spinner="Dang tai ORB cache + YOLO (1 lan)...")
    def _load_resources(wpath_s: str, device_s: str, conf_s: float,
                        imgsz_s: str):
        cache = ba.load_cache()
        import torch
        dev = device_s
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        req = {"auto": None, "3072": 3072, "2560": 2560,
               "1920": 1920}.get(imgsz_s, None)
        yolo = YOLODetector(Path(wpath_s), conf_threshold=float(conf_s),
                            imgsz=req if req else int(DEFAULT_IMGSZ),
                            device=dev)
        if req is not None and yolo.imgsz != req:
            # May yeu tu ha them (khong bao gio tang vuot yeu cau).
            pass
        return cache, yolo, dev

    try:
        cache, yolo, device_used = _load_resources(
            str(wpath), str(device_opt), float(conf),
            {"Auto (3072, tu ha khi yeu)": "auto",
             "3072 (full-res chuan)": "3072",
             "2560 (VRAM 4-6GB)": "2560",
             "1920 (CPU / VRAM <4GB)": "1920"}[imgsz_opt])
    except Exception as e:
        st.error(f"Khong tai duoc model/cache: {type(e).__name__}: {e}")
        return
    st.caption(f"Nguon: **{src_name}** ({img_bgr.shape[1]}x{img_bgr.shape[0]}) · "
               f"weights=`{Path(wpath).name}` · device=`{device_used}` · "
               f"imgsz thuc te=`{yolo.imgsz}` (auto-adapt VRAM/CPU) · "
               f"conf={float(conf):.2f} · align={'fast' if fast else 'full'}")
    if getattr(yolo, "adapt_note", ""):
        st.caption(f"🖥️ {yolo.adapt_note}")

    if not run:
        st.info("Nhan **▶️ Chay Dual-Path** de suy luan (GPU 1 lan cho batch ROI). "
                "Coder chi smoke-test 1-2 anh; eval full 36 anh la viec cua Verifier.")
        return

    # -- Guard anh dau vao (BUG-DAY6-01) ---------------------------------------
    # Anh la (khong phai PCB / qua nho) phai fail RO o day, khong ha nguong
    # MIN_INLIERS_BOARD_ID=30 de "cho qua".
    bad = validate_input_image(img_bgr)
    if bad is not None:
        st.error(bad)
        st.info("Goi y loai tru: chon 1 trong 8 anh mau (GOOD 01/04 hoac "
                "6 lop board 01) de kiem tra pipeline truoc khi upload anh tu chup.")
        return

    # -- Suy luan --------------------------------------------------------------
    with st.spinner("Dang chay Dual-Path (align → diff → batch ROI → fusion → 4 muc)..."):
        try:
            out = run_dualpath_single(img_bgr, yolo, cache, fast=fast)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            st.error(f"Pipeline loi: {msg}")
            # BUG-DAY6-01: khi board-ID fail (anh la / ngoai 10 board co template),
            # hien rank van tay + goi y + log scores/kich thuoc de tai hien.
            try:
                h0, w0 = int(img_bgr.shape[0]), int(img_bgr.shape[1])
                st.caption(f"Anh dau vao: {w0}x{h0} ({src_name}). Nguong board-ID "
                           f"giu nguyen MIN_INLIERS=30 — anh la phai fail ro, khong 'cho qua'.")
                gray0 = (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                         if img_bgr.ndim == 3 else img_bgr)
                st.caption("Rank van tay gradient (top-4): "
                           f"{shortlist_rank_text(gray0, cache)}")
            except Exception:
                pass
            st.info("Goi y loai tru: thu GOOD 01/04 hoac 1 trong 6 anh mau board 01 "
                    "truoc. Neu anh tu chup, hay gui lai kich thuoc + nguon chup de kiem tra "
                    "(anh ngoai 10 board 01/04/05-12, anh qua nho/xoay/crop manh, hoac "
                    "khong phai PCB deu khong nhan dien duoc).")
            return

    res = out["result"]
    items = res.get("items", [])
    level = int(res.get("level", 4))
    lname, ldesc = LEVEL_STYLE.get(level, ("?", ""))
    banner = {1: "error", 2: "warning", 3: "info", 4: "success"}.get(level, "info")

    # -- 3 khung hinh -----------------------------------------------------------
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("(1) Anh + BBox YOLO (da fuse)")
        boxed = draw_boxes_rgb(img_bgr, items)
        boxed = mark_centroids_rgb(boxed, out["centroids_input"], img_bgr.shape)
        st.image(boxed, use_container_width=True,
                 caption=f"{len([i for i in items if i.get('conf') is not None])} box co conf "
                         f"(vong tron trang = tam blob qua H_inv, R={GATE_R:.0f}px)")
    with c2:
        st.subheader("(2) Diff heatmap Nhanh B")
        heat = diff_heatmap_input_space(out["diff_map"], out["H_inv"], img_bgr.shape)
        ov = overlay_heatmap(img_bgr, heat)
        ov = mark_centroids_rgb(ov, out["centroids_input"], img_bgr.shape)
        st.image(ov, use_container_width=True,
                 caption=f"{len(out['blobs'])} blob (Area>=60, TOP-8 theo Z_diff, sigma_floor=5.0)")
    with c3:
        st.subheader("(3) Ket luan 4 muc")
        msg = f"### {lname} (level {level})\n{ldesc}"
        getattr(st, banner)(msg)
        st.write(f"**Board**: `{out['board_id']}` (inliers={out['inliers']}, "
                 f"{out['mode']}{', escalated' if out['escalated'] else ''})")
        st.write(f"**Latency**: align {out['t_align_ms']:.0f}ms · "
                 f"diff {out['t_diff_ms']:.0f}ms · ROI-batch {out['t_roi_ms']:.0f}ms · "
                 f"**tong {out['t_total_ms']:.0f}ms**")
        for line in res.get("reasons", [])[:6]:
            st.write(f"- {line}")

    # -- Chi tiet (cho hoi dong hoi sau) ---------------------------------------
    with st.expander("🔬 Chi tiet Nhanh B (tung blob: Z_diff, Area, centroid)"):
        if out["blobs"]:
            rows = []
            for b in out["blobs"]:
                cx_t, cy_t = b["centroid_tpl"]
                rows.append({
                    "centroid_tpl": f"({cx_t:.0f}, {cy_t:.0f})",
                    "area": round(float(b["area"]), 1),
                    "Z_diff": round(float(b["z_diff"]), 2),
                    "mean_res": round(float(b.get("mean_res", 0.0)), 2),
                })
            # BUG-DAY6-02: khong dung st.table (keo pyarrow doi numpy>=2.0,
            # xung dot numpy 1.26.4 cua venv) -> render markdown thu cong.
            render_rows_as_markdown(rows, "")
        else:
            st.write("Khong co blob (anh sach hoac residual = 0) → ung vien PASS.")
        st.caption(f"diff_map shape (Template): {np.shape(out['diff_map'])} · "
                   f"tau=2 (RUN_TAU_PX) · seed_k=1.25 (RUN_SEED_K) · "
                   f"sigma_floor=5.0 (Chot 2) · valid_mask ngoai FOV da loai.")
    with st.expander("🎯 Chi tiet Nhanh A + Fusion (tung box)"):
        if items:
            rows = []
            for it in items:
                bb = it.get("bbox")
                rows.append({
                    "class": it.get("class_name"), "conf": it.get("conf"),
                    "source": it.get("source"), "Z": it.get("z_diff"),
                    "area": it.get("area"),
                    "bbox": ([round(float(v), 1) for v in bb]
                             if bb is not None else None),
                })
            # BUG-DAY6-02: nhu tren — render markdown, khong dung st.table.
            render_rows_as_markdown(rows, "")
        else:
            st.write("Khong co evidence (YOLO khong giu box nao + khong blob) → PASS.")
        st.caption(f"predict_batch 1 call cho {out['n_crops']} crop 640x640 "
                   f"(giu lai {out['kept_per_crop']}) · gating R={CENTER_GATE_RADIUS:.0f}px · "
                   f"fusion IoU>={FUSION_IOU} boost +{CONF_BOOST}.")
    with st.expander("📜 Ly do quyet dinh (logic.decide, khong che dau)"):
        for d in res.get("details", []):
            st.write(f"- L{d.get('level')} {d.get('conclusion')}/{d.get('verdict')}: "
                     f"{d.get('why')} "
                     f"(cls={d.get('class_name')}, conf={d.get('conf')}, "
                     f"src={d.get('source')}, Z={d.get('z_diff')}, area={d.get('area')})")
        if not res.get("details"):
            st.write("Khong co evidence nao → PASS / NONE (mac dinh muc 4).")


if __name__ == "__main__":
    main()
