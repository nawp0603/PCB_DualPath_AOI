"""Ngay 1 — Dual-Path Industrial AOI: baseline / ablation evaluation harness.

Pham vi (plan.md Phan 5, dong Ngay 1):
  * Dung split board-disjoint: train {05,06,08,09,10,11,12}, val {07},
    test = Zero-defect-training Boards {01, 04} (khong goi la "unseen").
  * Xuat 4 dong baseline / ablation TACH BIET theo tung kich ban:
      1. YOLO full-board @640 (plain, rect=True)
      2. SAHI slice 640 overlap 0.20 (chi baseline ablation, KHONG dung
         trong pipeline chinh)
      3. PatchCore image-level AUROC/FPR (chi baseline ablation, KHONG dung
         trong pipeline chinh)
      4. YOLO full-board @3072 rect=True half=True (Nhanh A chuan)
  * Hai kich ban TACH BIET, khong tron so (plan.md 4.2):
      - Scenario 1 "standard": anh PKU goc (can san tinh).
      - Scenario 2 "camera": mo phong nhieu camera thuc te (dich chuyen
        +-25px, xoay +-2 do, bien dang phoi canh, sang/vignette, nhieu hat)
        ap len anh test 01/04 + template GOOD 01/04.

Quy tac trung thuc so lieu: khong ghi bat ky con so F1/Precision muc tieu
nao thanh ket qua da do. Moi con so in ra deu tu chay thuc te.

Zero-touch: moi duong dan ngoai project moi la STRICT READ-ONLY (doc qua
configs/resource_paths.py). Output chi ghi trong project moi (results/).

3 chot chan ky thuat lien quan file nay:
  * Nhanh A bat buoc rect=True, half=True (cuda) khi goi model.predict.
  * PatchCore/SAHI chi do baseline o day, cam dua vao logic quyet dinh.
  * Coder chi smoke-test 1-2 anh (--max-images 2). Eval full la viec Verifier.

Vi du:
  python scripts/evaluate.py --help
  python scripts/evaluate.py --scenario standard --configs yolo640 --max-images 2
  python scripts/evaluate.py --scenario both --out-json results/baseline_day1.json --out-csv results/baseline_day1.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Import resource_paths (READ-ONLY guard + PKU roots + weights/templates)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.resource_paths import (  # noqa: E402
    DEMO_ROOT,
    LOCAL_BASE_PT,
    LOCAL_DAY2_PT,
    LOCAL_TEMPLATES,
    LEGACY_PATCHCORE_CKPT,
    LEGACY_PKU_ROOT,
    LOCAL_PKU_ROOT,
    assert_writable_ok,
    default_weights,
    ensure_pku_dataset,
    find_pku_root,
)

# PORTABLE: mac dinh = weights fine-tune day2 (neu co), khong thi base.
# Giu LEGACY_PKU_ROOT import (old-machine fallback + Verifier test).
_DEFAULT_WEIGHTS = default_weights()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# Ngay 5: Nhanh B + hop nhat (duoc import src/, KHONG sua src/ — quyen cua VERIFIER)
import src.reference.board_id_and_align as ba  # noqa: E402
import src.reference.tolerance_diff as td  # noqa: E402
from src.models.yolo_detector import YOLODetector  # noqa: E402
from src.models.ensemble import fuse_and_decide  # noqa: E402

# Thu tu class phai khop weights YOLO (da kiem tra tu best.pt):
# {0: open_circuit, 1: short, 2: mouse_bite, 3: spur,
#  4: spurious_copper, 5: missing_hole}
CLASS_ID = {
    "open_circuit": 0,
    "short": 1,
    "mouse_bite": 2,
    "spur": 3,
    "spurious_copper": 4,
    "missing_hole": 5,
}
CLASS_NAMES = {v: k for k, v in CLASS_ID.items()}

TRAIN_BOARDS = {"05", "06", "08", "09", "10", "11", "12"}
VAL_BOARDS = {"07"}
TEST_BOARDS = {"01", "04"}  # Zero-defect-training Boards

DEFAULT_CONF = 0.20  # plan.md 2.2: conf=0.20
DEFAULT_IOU = 0.5
SAHI_SLICE = 640
SAHI_OVERLAP = 0.20


# ---------------------------------------------------------------------------
# IO helpers (Unicode-safe, read-only friendly)
# ---------------------------------------------------------------------------
def read_bgr(path: str | Path) -> np.ndarray:
    """Doc anh BGR uint8, an toan voi path Unicode (chi doc, khong ghi)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot decode image: {path}")
    return img


def board_id_from_xml_filename(filename: str) -> str:
    """"01_missing_hole_01.jpg" -> "01"."""
    return Path(filename).stem.split("_")[0].zfill(2)


def resolve_defect_image(ann_dir_name: str, xml_path: Path, xml_filename_field: str) -> Path | None:
    """Resolve anh loi theo stem cua file XML (ten trong XML lech voi file dia).

    Thu tu uu tien: <stem>.jpg, <stem>.JPG, ten goc trong XML.
    PORTABLE: quet LOCAL_PKU_ROOT -> DEMO_ROOT (bundled 01/04) ->
    LEGACY_PKU_ROOT (old machine only). May moi dung DEMO_ROOT.
    """
    stem = xml_path.stem
    if stem.endswith("_annotation"):
        stem = stem[: -len("_annotation")]
    from configs.resource_paths import DEMO_ROOT as _DEMO
    roots = [LOCAL_PKU_ROOT, _DEMO, LEGACY_PKU_ROOT]
    candidates = [stem + ".jpg", stem + ".JPG", xml_filename_field,
                  Path(xml_filename_field).stem + ".jpg"]
    for root in roots:
        try:
            if not root.is_dir():
                continue
        except Exception:
            continue
        for cand in candidates:
            q = root / "images" / ann_dir_name / cand
            if q.exists():
                return q
    return None


def collect_defect_items(pku_root: Path, boards: set[str] = TEST_BOARDS) -> list[dict[str, Any]]:
    """Quet Annotations, tra ve item defect thuoc boards chi dinh.

    Moi item: {path, board, gt: [(cls_id, [x1,y1,x2,y2])], xml}.
    """
    items: list[dict[str, Any]] = []
    ann_root = pku_root / "Annotations"
    if not ann_root.is_dir():
        raise FileNotFoundError(f"Thieu thu muc Annotations: {ann_root}")
    for ann_dir in sorted(p for p in ann_root.iterdir() if p.is_dir()):
        for xp in sorted(ann_dir.glob("*_annotation.xml")):
            try:
                r = ET.parse(xp).getroot()
            except ET.ParseError:
                continue
            fn_el = r.find("filename")
            if fn_el is None or not fn_el.text:
                continue
            fname = fn_el.text.strip()
            bid = board_id_from_xml_filename(fname)
            if bid not in boards:
                continue
            img_p = resolve_defect_image(ann_dir.name, xp, fname)
            if img_p is None:
                continue
            gt: list[tuple[int, list[float]]] = []
            for o in r.findall("object"):
                n = o.find("name")
                b = o.find("bndbox")
                if n is None or b is None or not n.text:
                    continue
                cid = CLASS_ID.get(n.text.strip().lower())
                if cid is None:
                    continue
                try:
                    box = [float(b.find("xmin").text), float(b.find("ymin").text),
                           float(b.find("xmax").text), float(b.find("ymax").text)]
                except (TypeError, ValueError):
                    continue
                gt.append((cid, box))
            if gt:
                items.append({"path": str(img_p), "board": bid,
                              "xml": str(xp), "gt": gt})
    items.sort(key=lambda d: d["path"])
    return items


def collect_good_items(boards: set[str] = TEST_BOARDS) -> list[dict[str, Any]]:
    """Anh GOOD = Golden Template 01/04 (ban copy trong assets/templates).

    Day la chuan doi chieu cong nghiep (Golden Sample), khong phai "an gian".
    """
    items: list[dict[str, Any]] = []
    for bid in sorted(boards):
        p = LOCAL_TEMPLATES / f"{bid}.JPG"
        if p.exists():
            items.append({"path": str(p), "board": bid, "gt": []})
    return items


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def box_iou(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    ua = (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])) \
        + (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])) - inter
    return inter / ua if ua > 0 else 0.0


def voc_ap(dets: list[tuple[int, float, list[float], int]],
           gts: dict[int, list[tuple[int, list[float]]]],
           thr: float = 0.5) -> tuple[float, dict[str, float]]:
    """VOC AP50 kieu 11-diem lien tuc (giong _bench/sahi_fullboard.py)."""
    by_cls: dict[int, list[tuple[float, int, list[float]]]] = {}
    for img_i, conf, box, cls in dets:
        by_cls.setdefault(cls, []).append((conf, img_i, box))
    gt_cls: dict[int, dict[int, list[list[Any]]]] = {}
    for img_i, lst in gts.items():
        for cls, box in lst:
            gt_cls.setdefault(cls, {}).setdefault(img_i, []).append([box, False])
    aps: list[float] = []
    per: dict[str, float] = {}
    for cls in sorted(gt_cls):
        d = sorted(by_cls.get(cls, []), key=lambda t: -t[0])
        npos = sum(len(v) for v in gt_cls[cls].values())
        if npos == 0:
            continue
        tp = np.zeros(len(d))
        fp = np.zeros(len(d))
        for j, (_, img_i, box) in enumerate(d):
            cands = gt_cls[cls].get(img_i, [])
            best, bi = 0.0, -1
            for m, cd in enumerate(cands):
                if cd[1]:
                    continue
                v = box_iou(box, cd[0])
                if v > best:
                    best, bi = v, m
            if best >= thr and bi >= 0:
                cands[bi][1] = True
                tp[j] = 1
            else:
                fp[j] = 1
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        rec = ctp / max(npos, 1)
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for q in range(len(mpre) - 2, -1, -1):
            mpre[q] = max(mpre[q], mpre[q + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
        aps.append(ap)
        per[CLASS_NAMES.get(cls, str(cls))] = round(ap, 4)
    return (float(np.mean(aps)) if aps else 0.0), per


def pr_counts(dets: list[tuple[int, float, list[float], int]],
              gts: dict[int, list[tuple[int, list[float]]]],
              thr: float = 0.5) -> tuple[int, int, int]:
    npos = sum(len(v) for v in gts.values())
    used: set[tuple[int, int, int]] = set()
    tp = fp = 0
    for img_i, _, box, cls in sorted(dets, key=lambda t: -t[1]):
        cands = [(m, bb) for m, (kk, bb) in enumerate(gts.get(img_i, [])) if kk == cls]
        best, bi = 0.0, None
        for m, bb in cands:
            key = (img_i, cls, m)
            if key in used:
                continue
            v = box_iou(box, bb)
            if v > best:
                best, bi = v, key
        if best >= thr and bi is not None:
            used.add(bi)
            tp += 1
        else:
            fp += 1
    return tp, fp, npos


def compute_auroc(scores: list[float], labels: list[int]) -> float:
    if not scores or len(set(labels)) < 2:
        return float("nan")
    order = sorted(zip(scores, labels), key=lambda p: -p[0])
    total_pos = sum(1 for l in labels if l == 1)
    total_neg = sum(1 for l in labels if l == 0)
    if total_pos == 0 or total_neg == 0:
        return float("nan")
    tps = fps = 0
    prev_fpr, prev_tpr, auroc = 0.0, 0.0, 0.0
    # Duyet tung diem theo thu tu score giam dan (trapezoidal chinh xac).
    for _, lab in order:
        if lab == 1:
            tps += 1
        else:
            fps += 1
        tpr, fpr = tps / total_pos, fps / total_neg
        auroc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr, prev_tpr = fpr, tpr
    return float(auroc)


# ---------------------------------------------------------------------------
# Camera perturbation (Scenario 2) — tong hop, co warp GT boxes
# ---------------------------------------------------------------------------
def perturb_image_and_boxes(img_bgr: np.ndarray, boxes: list[list[float]],
                            seed: int) -> tuple[np.ndarray, list[list[float]]]:
    """Mo phong nhieu camera: shift +-25px, xoay +-2do, perspective nhe,
    doi sang cuc bo + vignette + nhieu hat. Tra ve (anh, boxes da warp)."""
    rng = np.random.default_rng(seed)
    h, w = img_bgr.shape[:2]

    dx, dy = float(rng.uniform(-25, 25)), float(rng.uniform(-25, 25))
    ang = float(rng.uniform(-2.0, 2.0))
    cx, cy = w / 2.0, h / 2.0
    m_aff = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    m_aff[0, 2] += dx
    m_aff[1, 2] += dy
    h_aff = np.vstack([m_aff, [0.0, 0.0, 1.0]])

    j = 0.008  # bien dang phoi canh ~0.8% kich thuoc
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-j, j, size=(4, 2)).astype(np.float32) \
        * np.float32([w, h])
    h_persp, _ = cv2.findHomography(src, dst)
    if h_persp is None:
        h_persp = np.eye(3)
    h_all = h_persp @ h_aff

    warped = cv2.warpPerspective(img_bgr, h_all, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

    # Photometric: gain/bias + vignette + gaussian noise.
    gain = float(rng.uniform(0.85, 1.15))
    bias = float(rng.uniform(-10.0, 10.0))
    f = warped.astype(np.float32) * gain + bias
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    rx, ry = (xs - w / 2) / (w / 2), (ys - h / 2) / (h / 2)
    rr = np.sqrt(rx ** 2 + ry ** 2) / np.sqrt(2.0)
    vig = 1.0 - 0.15 * np.clip(rr, 0, 1) ** 2
    f *= vig[..., None]
    f += rng.normal(0, float(rng.uniform(3.0, 5.0)), size=f.shape)
    out = np.clip(f, 0, 255).astype(np.uint8)

    new_boxes: list[list[float]] = []
    if boxes:
        pts = []
        for b in boxes:
            pts.append([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
        arr = np.float32(pts).reshape(-1, 1, 2)
        wpts = cv2.perspectiveTransform(arr, h_all.astype(np.float32)).reshape(-1, 4, 2)
        for q in wpts:
            x1, y1 = float(np.clip(q[:, 0].min(), 0, w - 1)), float(np.clip(q[:, 1].min(), 0, h - 1))
            x2, y2 = float(np.clip(q[:, 0].max(), 0, w - 1)), float(np.clip(q[:, 1].max(), 0, h - 1))
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                new_boxes.append([x1, y1, x2, y2])
    return out, new_boxes


# ---------------------------------------------------------------------------
# Runners: YOLO plain (rect+half), SAHI, PatchCore — chi baseline
# ---------------------------------------------------------------------------
class YoloRunner:
    """Wrapper YOLO Ultralytics dung cho baseline. Bat buoc rect=True,
    half=True khi chay cuda (Chot 3 — chong Square Padding).

    PORTABLE: CPU -> half=False + cap imgsz 1920 (tranh crash
    "Half precision not supported on CPU"); CUDA OOM -> empty_cache +
    thu lai 2560/1920 (graceful, giu rect+half)."""

    def __init__(self, weights: Path, device: str, conf: float):
        from ultralytics import YOLO
        import torch
        from src.models.yolo_detector import resolve_device_and_imgsz
        self.conf = float(conf)
        dev, half, _auto, note = resolve_device_and_imgsz(
            None if device == "auto" else device, 3072)
        self.device = dev
        self.half = half
        if note and ("fallback" in note or "VRAM" in note):
            print(f"[YoloRunner] {note}")
        self.model = YOLO(str(weights))

    def infer_bgr(self, img_bgr: np.ndarray, imgsz: int) -> tuple[list[tuple[int, float, list[float]]], float]:
        from src.models.yolo_detector import CPU_FALLBACK_IMGSZ
        t0 = time.perf_counter()
        eff_imgsz = int(imgsz)
        if self.device == "cpu" and eff_imgsz > CPU_FALLBACK_IMGSZ:
            eff_imgsz = CPU_FALLBACK_IMGSZ
        # Ultralytics nhan BGR ndarray truc tiep; rect+half ap dung tai day.
        try:
            r = self.model.predict(source=img_bgr, imgsz=eff_imgsz, conf=self.conf,
                                   device=self.device, verbose=False,
                                   rect=True, half=self.half)[0]
        except Exception as e:
            import torch as _t
            if self.device == "cuda" and ("out of memory" in str(e).lower()
                                          or "OOM" in str(type(e).__name__)):
                _t.cuda.empty_cache()
                fell_back = False
                for fb in (2560, 1920):
                    if fb >= eff_imgsz:
                        continue
                    try:
                        print(f"[YoloRunner] OOM o imgsz={eff_imgsz}, thu lai {fb}.")
                        r = self.model.predict(
                            source=img_bgr, imgsz=fb, conf=self.conf,
                            device=self.device, verbose=False,
                            rect=True, half=self.half)[0]
                        fell_back = True
                        break
                    except Exception:
                        _t.cuda.empty_cache()
                        continue
                if not fell_back:
                    raise
            else:
                raise
        dt = (time.perf_counter() - t0) * 1000.0
        out: list[tuple[int, float, list[float]]] = []
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                out.append((int(b.cls[0]), float(b.conf[0]),
                            [float(v) for v in b.xyxy[0]]))
        return out, dt


_SAHI_MODEL = None


def sahi_infer(img_bgr: np.ndarray, weights: Path, device: str,
               conf: float) -> tuple[list[tuple[int, float, list[float]]], float]:
    """SAHI slice 640 overlap 0.20 — CHI BASELINE ABLATION."""
    global _SAHI_MODEL
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    if _SAHI_MODEL is None:
        _SAHI_MODEL = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=str(weights),
            confidence_threshold=float(conf), device=device)
    t0 = time.perf_counter()
    res = get_sliced_prediction(img_bgr, _SAHI_MODEL,
                                slice_height=SAHI_SLICE, slice_width=SAHI_SLICE,
                                overlap_height_ratio=SAHI_OVERLAP,
                                overlap_width_ratio=SAHI_OVERLAP, verbose=0)
    dt = (time.perf_counter() - t0) * 1000.0
    out: list[tuple[int, float, list[float]]] = []
    for o in res.object_prediction_list:
        b = o.bbox
        out.append((int(o.category.id), float(o.score.value),
                    [float(b.minx), float(b.miny), float(b.maxx), float(b.maxy)]))
    return out, dt


class PatchCoreScorer:
    """PatchCore image-level scorer — CHI BASELINE ABLATION.

    Tai hien toi thieu logic anomaly_detector.py cua project cu
    (resnet18 layer2+layer3, memory_bank, max patch distance) de project moi
    doc lap, chi doc .ckpt READ-ONLY, khong import src project cu."""

    def __init__(self, ckpt_path: Path, device: str):
        import torch
        from torchvision.models import resnet18, ResNet18_Weights
        import torch.nn as nn
        import torch.nn.functional as F
        self.torch = torch
        self.F = F
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        self.memory_bank = ckpt["memory_bank"].to(dtype=torch.float32).to(self.device)
        self.threshold = float(ckpt.get("threshold", 0.0))
        self.input_size = list(ckpt.get("input_size", [256, 256]))

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT).to(self.device)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone
        self.feats: dict[str, Any] = {}
        backbone.layer2.register_forward_hook(self._hook("l2"))
        backbone.layer3.register_forward_hook(self._hook("l3"))
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1).to(self.device)

    def _hook(self, name: str):
        def fn(_m, _i, out):
            self.feats[name] = out
        return fn

    def score_bgr(self, img_bgr: np.ndarray) -> tuple[float, float]:
        import numpy as _np  # local de ro rang
        torch = self.torch
        t0 = time.perf_counter()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        _w, _h = int(self.input_size[0]), int(self.input_size[1])
        img_rs = cv2.resize(img_rgb, (_w, _h))
        img_rs = img_rs.astype(_np.float32) / 255.0
        mean = _np.array([0.485, 0.456, 0.406], dtype=_np.float32)
        std = _np.array([0.229, 0.224, 0.225], dtype=_np.float32)
        img_rs = (img_rs - mean) / std
        t = torch.from_numpy(img_rs.transpose(2, 0, 1)[None, ...]).to(self.device)
        self.feats = {}
        with torch.no_grad():
            _ = self.backbone(t)
            f2 = self.pool(self.feats["l2"])
            f3 = self.pool(self.feats["l3"])
            f3u = self.F.interpolate(f3, size=f2.shape[2:], mode="bilinear", align_corners=False)
            feats = self.F.normalize(torch.cat([f2, f3u], dim=1), p=2, dim=1)
            b, c, hf, wf = feats.shape
            patches = feats.reshape(b, c, hf * wf).transpose(1, 2).reshape(-1, c)
            dists = torch.cdist(patches, self.memory_bank)
            score = float(dists.min(dim=1).values.max().item())
        dt = (time.perf_counter() - t0) * 1000.0
        return score, dt


# ---------------------------------------------------------------------------
# Evaluation drivers (tach biet scenario, khong tron so)
# ---------------------------------------------------------------------------
def eval_detection(items: list[dict[str, Any]], runner, tag: str,
                   iou_thr: float, perturb: bool,
                   seed_base: int) -> dict[str, Any]:
    dets: list[tuple[int, float, list[float], int]] = []
    gts: dict[int, list[tuple[int, list[float]]]] = {}
    lats: list[float] = []
    n_det = 0
    for i, it in enumerate(items):
        img = read_bgr(it["path"])
        gt_boxes = [list(b) for _, b in it["gt"]]
        if perturb:
            img, warped = perturb_image_and_boxes(img, gt_boxes, seed_base + i)
            gt_use = [(c, wb) for (c, _), wb in zip(it["gt"], warped)]
            # box bi warp mat han -> coi nhu khong con GT (hiem).
        else:
            gt_use = list(it["gt"])
        out, dt = runner(img)
        lats.append(dt)
        n_det += len(out)
        for cls, conf, box in out:
            dets.append((i, conf, box, cls))
        gts[i] = gt_use
    ap, per = voc_ap(dets, gts, thr=iou_thr)
    tp, fp, npos = pr_counts(dets, gts, thr=iou_thr)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / npos if npos else 0.0
    return {"tag": tag, "AP50": round(ap, 4),
            "recall": round(rec, 4), "precision": round(prec, 4),
            "tp": tp, "fp": fp, "npos": npos, "n_det": n_det,
            "latency_ms": round(float(np.mean(lats)), 1) if lats else 0.0,
            "per_class": per, "num_images": len(items)}


def eval_patchcore(defect_items: list[dict[str, Any]], good_items: list[dict[str, Any]],
                   scorer: PatchCoreScorer, perturb: bool,
                   seed_base: int) -> dict[str, Any]:
    scores: list[float] = []
    labels: list[int] = []
    lats: list[float] = []
    per_image: list[dict[str, Any]] = []
    idx = 0
    for it in defect_items + good_items:
        img = read_bgr(it["path"])
        if perturb:
            img, _ = perturb_image_and_boxes(img, [], seed_base + idx)
        s, dt = scorer.score_bgr(img)
        lats.append(dt)
        lab = 0 if not it["gt"] else 1
        scores.append(s)
        labels.append(lab)
        per_image.append({"file": Path(it["path"]).name, "label": lab,
                          "score": round(s, 5)})
        idx += 1
    auroc = compute_auroc(scores, labels)
    thr = scorer.threshold
    fp = sum(1 for s, l in zip(scores, labels) if l == 0 and s > thr)
    tn = sum(1 for s, l in zip(scores, labels) if l == 0 and s <= thr)
    tp = sum(1 for s, l in zip(scores, labels) if l == 1 and s > thr)
    fn = sum(1 for s, l in zip(scores, labels) if l == 1 and s <= thr)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"tag": "patchcore", "auroc": round(float(auroc), 4) if auroc == auroc else None,
            "threshold": thr, "fpr": round(fpr, 4), "tp": tp, "tn": tn,
            "fp": fp, "fn": fn, "num_images": len(scores),
            "latency_ms": round(float(np.mean(lats)), 1) if lats else 0.0,
            "per_image": per_image}


# ---------------------------------------------------------------------------
# Ngay 5: duong pip ca 2 nhanh (Nhanh A @3072 + Nhanh B + Box Fusion + quy tac 4 muc)
# ---------------------------------------------------------------------------
def run_dualpath(defect_items: list[dict[str, Any]],
                 good_items: list[dict[str, Any]],
                 yolo: YOLODetector, device: str,
                 perturb: bool, seed_base: int,
                 cache: dict[str, Any] | None = None) -> dict[str, Any]:
    """Chay pipeline Dual-Path toan bo tren 1 scenario:
      Nhanh B (align + tolerance diff) -> cat crop 640 quanh tam blob -> Nhanh A
      predict_batch (1 call GPU) -> Box Fusion IoU>=0.3 (S_final=min(1.0,S_A+0.15))
      -> quy tac 4 muc (logic.decide).

    Tra ve AP50/Recall/Precision tren anh loi + FAR tren anh GOOD (bat ky ket
    luan khac PASS = bao gia). So lieu lay tu chay thuc te, khong lam tron.
    """
    if cache is None:
        cache = ba.load_cache()
    templates = {bid: None for bid in cache["board_ids"]}

    dets: list[tuple[int, float, list[float], int]] = []
    gts: dict[int, list[tuple[int, list[float]]]] = {}
    far_triggers = 0
    far_total = len(good_items)
    lats: list[float] = []
    levels: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
    scenario_error: str | None = None

    for i, it in enumerate(defect_items + good_items):
        img = read_bgr(it["path"])
        gt_boxes = [list(b) for _, b in it["gt"]]
        if perturb:
            img, warped_boxes = perturb_image_and_boxes(img, gt_boxes,
                                                        seed_base + i)
            gt_use = [(c, wb) for (c, _), wb in zip(it["gt"], warped_boxes)]
        else:
            gt_use = list(it["gt"])
        gts[i] = gt_use

        t0 = time.perf_counter()
        try:
            r = ba.identify_and_align(img, cache, fast=True)
            bid = r["board_id"]
            tpl = templates.get(bid)
            if tpl is None:
                tpl_p = LOCAL_TEMPLATES / f"{bid}.JPG"
                data = np.fromfile(str(tpl_p), dtype=np.uint8)
                tpl = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                if tpl.shape != r["warped"].shape[:2]:
                    tpl = cv2.resize(tpl, (r["warped"].shape[1],
                                           r["warped"].shape[0]),
                                     interpolation=cv2.INTER_AREA)
                templates[bid] = tpl
            out = td.run_branch_b(r["warped"], tpl)
            cents = ba.project_points(
                np.float32([b["centroid_tpl"] for b in out["blobs"]]),
                r["H_inv"]) if out["blobs"] else np.zeros((0, 2))
            H, W = img.shape[:2]
            crops, origins = [], []
            for (px, py) in cents:
                x0 = int(np.clip(px - 320, 0, max(W - 640, 0)))
                y0 = int(np.clip(py - 320, 0, max(H - 640, 0)))
                crops.append(cv2.cvtColor(img[y0:y0 + 640, x0:x0 + 640],
                                          cv2.COLOR_BGR2RGB))
                origins.append((x0, y0))
            batch = yolo.predict_batch(crops) if crops else []
            yolo_dets_all = []
            for dets_one, (px, py), (x0, y0) in zip(batch, cents, origins):
                kept = YOLODetector.spatial_center_gating(
                    dets_one, center_xy=(float(px), float(py)),
                    radius=80.0, crop_origin=(float(x0), float(y0)))
                # Dich toa do det tu he toa do crop ve he toa do anh goc truoc
                # khi Box Fusion / tinh AP50 (det tra ve o toa do crop, con
                # diff_box/centroid o toa do anh goc -> neu khong dich, IoU luon
                # = 0 va khong det nao tro thanh TP).
                for d in kept:
                    d = dict(d)
                    d["bbox"] = [d["bbox"][0] + x0, d["bbox"][1] + y0,
                                 d["bbox"][2] + x0, d["bbox"][3] + y0]
                    yolo_dets_all.append(d)
            res = fuse_and_decide(yolo_dets_all, out["blobs"], r["H_inv"])
            # Detection-level det (co conf) de tinh AP50/Recall/Precision.
            for itm in res["items"]:
                if itm.get("conf") is not None:
                    dets.append((i, float(itm["conf"]),
                                 [float(v) for v in itm["bbox"]],
                                 int(itm["class_id"])))
            levels[res["level"]] += 1
            err = None
        except Exception as e:  # ghi ro, khong che dau
            err = f"{type(e).__name__}: {e}"
            if scenario_error is None:
                scenario_error = err
            res = None
        dt = (time.perf_counter() - t0) * 1000.0
        lats.append(dt)

        if not it["gt"]:  # anh GOOD -> FAR: bat ky ket luan khac PASS
            if res is not None and res["level"] < 4:
                far_triggers += 1

    ap, per = voc_ap(dets, gts, thr=DEFAULT_IOU)
    tp, fp, npos = pr_counts(dets, gts, thr=DEFAULT_IOU)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / npos if npos else 0.0
    far = far_triggers / far_total if far_total else 0.0
    return {"tag": f"Dual-Path (Nhanh A @{yolo.imgsz} + Nhanh B + Fusion)",
            "AP50": round(ap, 4), "recall": round(rec, 4),
            "precision": round(prec, 4),
            "tp": tp, "fp": fp, "npos": npos, "n_det": len(dets),
            "far": round(far, 4), "far_triggers": far_triggers,
            "far_total": far_total,
            "latency_ms": round(float(np.mean(lats)), 1) if lats else 0.0,
            "num_images": len(defect_items + good_items),
            "num_defect": len(defect_items), "num_good": len(good_items),
            "per_class": per, "levels": levels,
            "error": scenario_error}


def run_scenario(name: str, defect_items: list[dict[str, Any]],
                 good_items: list[dict[str, Any]], args,
                 yolo: YoloRunner, device: str) -> dict[str, Any]:
    """Chay 4 dong baseline cho 1 scenario (standard HOAC camera)."""
    perturb = (name == "camera")
    seed_base = 20261023 if perturb else 0
    rows: dict[str, Any] = {"scenario": name, "num_defect": len(defect_items),
                            "num_good": len(good_items)}
    want = set(args.configs)

    if "yolo640" in want:
        rows["yolo640"] = eval_detection(
            defect_items, lambda im, _y=yolo: _y.infer_bgr(im, 640),
            "YOLO full-board @640 (rect=True)", args.iou, perturb, seed_base)
    if "sahi640" in want:
        try:
            rows["sahi640"] = eval_detection(
                defect_items,
                lambda im: sahi_infer(im, args.weights, device, args.conf),
                "SAHI slice=640 ov=0.20 [ABLATION]", args.iou, perturb, seed_base + 7919)
        except Exception as e:  # thieu sahi -> ghi ro, khong che dau
            rows["sahi640"] = {"tag": "SAHI slice=640 [ABLATION]",
                               "error": f"{type(e).__name__}: {e}"}
    if "patchcore" in want:
        if args.anomaly is None:
            rows["patchcore"] = {"tag": "patchcore [ABLATION]", "error": "no --anomaly ckpt"}
        else:
            try:
                scorer = PatchCoreScorer(Path(args.anomaly), device)
                rows["patchcore"] = eval_patchcore(defect_items, good_items, scorer,
                                                   perturb, seed_base + 31337)
            except Exception as e:
                rows["patchcore"] = {"tag": "patchcore [ABLATION]",
                                     "error": f"{type(e).__name__}: {e}"}
    if "yolo3072" in want:
        rows["yolo3072"] = eval_detection(
            defect_items, lambda im, _y=yolo: _y.infer_bgr(im, args.highres),
            f"YOLO full-board @{args.highres} rect=True half=True (Nhanh A)",
            args.iou, perturb, seed_base + 101)
    if "dualpath" in want:
        try:
            dp_yolo = YOLODetector(args.weights, conf_threshold=args.conf,
                                   imgsz=args.highres, device=device)
            cache = ba.load_cache()
            rows["dualpath"] = run_dualpath(defect_items, good_items, dp_yolo,
                                            device, perturb, seed_base, cache)
        except Exception as e:
            rows["dualpath"] = {
                "tag": "Dual-Path (Nhanh A + Nhanh B + Fusion) [ERROR]",
                "error": f"{type(e).__name__}: {e}"}
    return rows


def print_table(res: dict[str, Any]) -> None:
    print("\n===== BASELINE / ABLATION (so thuc do, khong muc tieu) =====")
    for sc in ("standard", "camera"):
        if sc not in res:
            continue
        r = res[sc]
        label = "Kich ban 1 (Standard — anh PKU goc)" if sc == "standard" \
            else "Kich ban 2 (Camera Perturbation — Zero-defect-training 01/04)"
        print(f"\n--- {label} --- "
              f"[{r.get('num_defect', '?')} anh loi + {r.get('num_good', '?')} GOOD]")
        for key in ("yolo640", "sahi640", "patchcore", "yolo3072", "dualpath"):
            if key not in r:
                continue
            v = r[key]
            if "error" in v:
                print(f"  {key:10s} ERROR: {v['error']}")
            elif key == "patchcore":
                print(f"  {key:10s} AUROC={v['auroc']} FPR={v['fpr']} "
                      f"(TP={v['tp']} TN={v['tn']} FP={v['fp']} FN={v['fn']}) "
                      f"{v['latency_ms']:.0f}ms/anh")
            elif key == "dualpath":
                err_txt = v.get("error")
                print(f"  {key:10s} AP50={v['AP50']:.4f} R={v['recall']:.3f} "
                      f"P={v['precision']:.3f} FAR={v['far']:.4f} "
                      f"(TP={v['tp']} FP={v['fp']} FAR_trig={v['far_triggers']}/{v['far_total']}) "
                      f"{v['latency_ms']:.0f}ms/anh levels={v['levels']}"
                      + (f" ERROR={err_txt}" if err_txt else ""))
            else:
                print(f"  {key:10s} AP50={v['AP50']:.4f} R={v['recall']:.3f} "
                      f"P={v['precision']:.3f} TP={v['tp']} FP={v['fp']} "
                      f"npos={v['npos']} det={v['n_det']} {v['latency_ms']:.0f}ms/anh")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ngay 1: baseline YOLO/SAHI/PatchCore board-disjoint 01/04.")
    p.add_argument("--weights", type=Path, default=None,
                   help="YOLO .pt (mac dinh portable: yolo11n_day2.pt neu co, "
                        "khong thi yolo11n_base.pt trong assets/weights/).")
    p.add_argument("--anomaly", type=Path, default=LEGACY_PATCHCORE_CKPT,
                   help="PatchCore .ckpt READ-ONLY (mac dinh: legacy model.ckpt). "
                        "Dung --anomaly '' de bo qua PatchCore.")
    p.add_argument("--scenario", choices=["standard", "camera", "both"], default="both",
                   help="Kich ban danh gia (mac dinh: both, xuat tach biet).")
    p.add_argument("--configs", nargs="+",
                   choices=["yolo640", "sahi640", "patchcore", "yolo3072",
                            "dualpath"],
                   default=["yolo640", "sahi640", "patchcore", "yolo3072",
                            "dualpath"],
                   help="Chon dong baseline can chay (smoke-test: chi 1 config).")
    p.add_argument("--highres", type=int, default=3072,
                   help="imgsz Nhanh A (3072 hoac 3584, mac dinh 3072).")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--iou", type=float, default=DEFAULT_IOU)
    p.add_argument("--max-images", type=int, default=None,
                   help="Gioi han so anh loi (smoke-test 1-2 anh). None = full 36 anh.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--out-json", type=Path, default=Path("results/baseline_day1.json"))
    p.add_argument("--out-csv", type=Path, default=Path("results/baseline_day1.csv"))
    p.add_argument("--no-save", action="store_true", help="Khong ghi file ket qua.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    # PORTABLE weights default: day2 neu co, khong thi base (khong hardcode
    # base lam mac dinh nua de may moi dung luon ban fine-tune tot nhat).
    if args.weights is None:
        args.weights = default_weights()
    # --anomaly '' (chuoi rong) = bo qua PatchCore. Luu y argparse type=Path
    # bien '' thanh Path('.') nen phai bat ca hai truong hop.
    if args.anomaly is not None and str(args.anomaly) in ("", "."):
        args.anomaly = None
    if args.anomaly is not None:
        args.anomaly = Path(args.anomaly)
    args.weights = Path(args.weights)

    if not args.weights.exists():
        raise FileNotFoundError(f"Thieu weights YOLO: {args.weights}")
    if args.anomaly is not None and not Path(args.anomaly).exists():
        print(f"[WARN] Khong thay PatchCore ckpt {args.anomaly} -> bo qua dong patchcore.")
        args.anomaly = None

    pku_root = ensure_pku_dataset()
    print(f"[evaluate] PKU root: {pku_root}")
    defect_items = collect_defect_items(pku_root, TEST_BOARDS)
    good_items = collect_good_items(TEST_BOARDS)
    print(f"[evaluate] Test Zero-defect-training Boards 01/04: "
          f"{len(defect_items)} anh loi "
          f"({sum(len(i['gt']) for i in defect_items)} boxes), "
          f"{len(good_items)} GOOD templates")
    if not defect_items:
        raise SystemExit("Khong thu thap duoc anh test 01/04 — kiem tra PKU root.")
    if args.max_images is not None:
        defect_items = defect_items[: max(0, int(args.max_images))]
        print(f"[evaluate] SMOKE-TEST mode: chi chay {len(defect_items)} anh loi.")

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[evaluate] device={device}, conf={args.conf}, iou={args.iou}, "
          f"configs={args.configs}, highres={args.highres}")
    yolo = YoloRunner(args.weights, device, args.conf)

    res: dict[str, Any] = {
        "meta": {"conf": args.conf, "iou": args.iou, "highres": args.highres,
                 "weights": str(args.weights),
                 "anomaly": str(args.anomaly) if args.anomaly else None,
                 "boards": "test=01,04 (Zero-defect-training Boards); "
                           "train=05,06,08,09,10,11,12; val=07",
                 "note": "PatchCore/SAHI chi baseline ablation, khong thuoc pipeline chinh."},
    }
    scenarios = ["standard", "camera"] if args.scenario == "both" else [args.scenario]
    for sc in scenarios:
        res[sc] = run_scenario(sc, defect_items, good_items, args, yolo, device)

    print_table(res)

    if not args.no_save:
        out_json = assert_writable_ok(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[evaluate] Da ghi JSON: {out_json}")
        out_csv = assert_writable_ok(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scenario", "config", "tag", "AP50", "precision", "recall",
                        "tp", "fp", "tn", "fn", "npos", "n_det", "auroc", "fpr",
                        "latency_ms", "num_images", "error", "far", "far_triggers",
                        "far_total", "levels"])
            for sc in scenarios:
                for key in ("yolo640", "sahi640", "patchcore", "yolo3072",
                            "dualpath"):
                    v = res[sc].get(key)
                    if v is None:
                        continue
                    w.writerow([sc, key, v.get("tag", ""),
                                v.get("AP50", ""), v.get("precision", ""),
                                v.get("recall", ""), v.get("tp", ""),
                                v.get("fp", ""), v.get("tn", ""),
                                v.get("fn", ""), v.get("npos", ""),
                                v.get("n_det", ""), v.get("auroc", ""),
                                v.get("fpr", ""), v.get("latency_ms", ""),
                                v.get("num_images", ""), v.get("error", ""),
                                v.get("far", ""), v.get("far_triggers", ""),
                                v.get("far_total", ""), v.get("levels", "")])
        print(f"[evaluate] Da ghi CSV: {out_csv}")
    print("\n[evaluate] DONE")


if __name__ == "__main__":
    main()
