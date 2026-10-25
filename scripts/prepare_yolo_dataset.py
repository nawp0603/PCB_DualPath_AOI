"""Ngay 2 — Chuan bi dataset YOLO board-disjoint (project moi, doc lap).

Nguon doc READ-ONLY (qua configs/resource_paths.py):
  * Anh loi + XML VOC : LEGACY_PKU_ROOT/images|Annotations (khong sua)
  * Golden Templates  : LOCAL_TEMPLATES (ban copy trong project moi)

Dau ra — CHI trong project moi (data/yolo_day2/), TUYET DOI khong ghi de
sang thu muc dataset cua project cu:
  data/yolo_day2/{train,val}/{images,labels}/ + data.yaml + stats.json

Hai diem FIX theo plan.md 2.2 (khac phuc loi Data Pipeline):
  1. clip_bbox_to_tile: CAT (clip) toa do bbox ve bien [0, tile_size] thay
     vi tra ve None. Ban cu lam mat nhan loi o ria tile -> false negatives
     co y. Chi loai bbox khi phan giao con lai qua nho (< 25% dien tich goc
     hoac canh < 8px).
  2. Them 25% tile nen sach (hard-negative background tiles, label rong) vao
     tap train de triet tieu bao gia (FP) tren crop 640x640.

Split board-disjoint: train {05,06,08,09,10,11,12}, val {07}.
Board 01/04 la Zero-defect-training Boards — KHONG co mat trong train/val.

Vi du:
  python scripts/prepare_yolo_dataset.py
  python scripts/prepare_yolo_dataset.py --bg-ratio 0.25 --tile-size 640
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.resource_paths import (  # noqa: E402
    LEGACY_PKU_ROOT,
    LOCAL_TEMPLATES,
    NEW_PROJECT_DIR,
    assert_writable_ok,
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

SEED = 42

TRAIN_BOARDS = {"05", "06", "08", "09", "10", "11", "12"}
VAL_BOARDS = {"07"}
# 01/04 = Zero-defect-training Boards: chi lam Golden Template + test, cam train.

TILE_SIZE = 640
STRIDE = 512
BG_RATIO = 0.25          # 25% tile nen sach trong train (plan.md 2.2)
MIN_KEEP_RATIO = 0.25    # giu bbox bi cat ria neu con >= 25% dien tich goc
MIN_KEEP_SIDE = 8.0      # va canh con lai >= 8px
VAL_BG_TILES = 10        # vai tile sach board 07 de val co tin hieu FP

CLASS_NAME_MAP = {
    "open_circuit": 0,
    "short": 1,
    "mouse_bite": 2,
    "spur": 3,
    "spurious_copper": 4,
    "missing_hole": 5,
}
CLASS_NAMES = {v: k for k, v in CLASS_NAME_MAP.items()}

OUT_ROOT = NEW_PROJECT_DIR / "data" / "yolo_day2"


# ---------------------------------------------------------------------------
# IO (Unicode-safe)
# ---------------------------------------------------------------------------
def read_image_rgb(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot decode image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_tile_png(img_rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    assert_writable_ok(path)
    path.write_bytes(buf.tobytes())


def write_label(labels: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_writable_ok(path)
    path.write_text("\n".join(labels), encoding="utf-8")


# ---------------------------------------------------------------------------
# Parse + split helpers
# ---------------------------------------------------------------------------
def board_id_from_xml_filename(filename: str) -> str:
    return Path(filename).stem.split("_")[0].zfill(2)


def parse_xml_annotation(xml_path: Path) -> dict[str, Any] | None:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return None
    fn_el = root.find("filename")
    if fn_el is None or not fn_el.text:
        return None
    filename = fn_el.text.strip()
    objects: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        bb_el = obj.find("bndbox")
        if name_el is None or bb_el is None or not name_el.text:
            continue
        cid = CLASS_NAME_MAP.get(name_el.text.strip().lower())
        if cid is None:
            continue
        try:
            xmin = int(float(bb_el.find("xmin").text or 0))
            ymin = int(float(bb_el.find("ymin").text or 0))
            xmax = int(float(bb_el.find("xmax").text or 0))
            ymax = int(float(bb_el.find("ymax").text or 0))
        except (TypeError, ValueError):
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        objects.append({"class_id": cid, "class_name": CLASS_NAMES[cid],
                        "bbox": [xmin, ymin, xmax, ymax]})
    if not objects:
        return None
    return {"filename": filename,
            "board_id": board_id_from_xml_filename(filename),
            "objects": objects}


def resolve_defect_image(class_folder: str, xml_path: Path) -> Path | None:
    """Resolve anh loi theo stem XML (ten trong XML lech voi file tren dia)."""
    stem = xml_path.stem
    if stem.endswith("_annotation"):
        stem = stem[: -len("_annotation")]
    img_root = LEGACY_PKU_ROOT / "images" / class_folder
    for cand in (stem + ".jpg", stem + ".JPG"):
        q = img_root / cand
        if q.exists():
            return q
    return None


def choose_tile_around_bbox(img_w: int, img_h: int, bbox: list[int],
                            tile_size: int) -> tuple[int, int]:
    xmin, ymin, xmax, ymax = bbox
    cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
    x0 = max(0, min(cx - tile_size // 2, max(0, img_w - tile_size)))
    y0 = max(0, min(cy - tile_size // 2, max(0, img_h - tile_size)))
    return x0, y0


# ---------------------------------------------------------------------------
# FIX 1 (plan.md 2.2): clip thay vi drop
# ---------------------------------------------------------------------------
def clip_bbox_to_tile(bbox: list[int], x0: int, y0: int,
                      tile_size: int) -> list[float] | None:
    """Cat bbox ve bien tile, tra ve YOLO normalized [cx, cy, w, h].

    Khac ban cu (tra None khi bbox cham ria tile): cat phan giao voi tile va
    chi loai khi phan con lai qua nho (area < 25% goc hoac canh < 8px) de
    tranh false negatives co y o ria tile.
    """
    xmin, ymin, xmax, ymax = bbox
    ix1, iy1 = max(xmin, x0), max(ymin, y0)
    ix2, iy2 = min(xmax, x0 + tile_size), min(ymax, y0 + tile_size)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw < MIN_KEEP_SIDE or ih < MIN_KEEP_SIDE:
        return None
    orig_area = max(1, (xmax - xmin)) * max(1, (ymax - ymin))
    if (iw * ih) / orig_area < MIN_KEEP_RATIO:
        return None
    cx = (ix1 + ix2) / 2.0 - x0
    cy = (iy1 + iy2) / 2.0 - y0
    return [cx / tile_size, cy / tile_size, iw / tile_size, ih / tile_size]


# ---------------------------------------------------------------------------
# FIX 2 (plan.md 2.2): 25% background tiles
# ---------------------------------------------------------------------------
def sample_background_tiles(n_needed: int, boards: set[str], prefix: str,
                            tile_size: int, stride: int,
                            rng: random.Random) -> list[tuple[np.ndarray, str]]:
    """Cat tile sach tu Golden Templates (vung khong loi theo dinh nghia).

    Golden Template la anh board chuan khong loi nen moi tile cat ra deu la
    hard-negative hop le cho viec triet FP.
    """
    pool: list[tuple[np.ndarray, str]] = []
    for bid in sorted(boards):
        tpl = LOCAL_TEMPLATES / f"{bid}.JPG"
        if not tpl.exists():
            print(f"  [WARN] Thieu template {bid}.JPG, bo qua")
            continue
        img = read_image_rgb(tpl)
        h, w = img.shape[:2]
        for y0 in range(0, max(1, h - tile_size + 1), stride):
            for x0 in range(0, max(1, w - tile_size + 1), stride):
                xa, ya = min(x0, w - tile_size), min(y0, h - tile_size)
                pool.append((img[ya:ya + tile_size, xa:xa + tile_size], bid))
    rng.shuffle(pool)
    return pool[:n_needed]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ngay 2: chuan bi data/yolo_day2 (clip fix + 25% bg).")
    p.add_argument("--tile-size", type=int, default=TILE_SIZE)
    p.add_argument("--stride", type=int, default=STRIDE)
    p.add_argument("--bg-ratio", type=float, default=BG_RATIO)
    p.add_argument("--val-bg-tiles", type=int, default=VAL_BG_TILES)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", type=Path, default=OUT_ROOT)
    return p


def main() -> None:
    args = build_parser().parse_args()
    ts, stride = int(args.tile_size), int(args.stride)
    rng = random.Random(int(args.seed))
    out_root: Path = assert_writable_ok(Path(args.out))

    train_img = out_root / "train" / "images"
    train_lbl = out_root / "train" / "labels"
    val_img = out_root / "val" / "images"
    val_lbl = out_root / "val" / "labels"

    stats: dict[str, Any] = {
        "tile_size": ts, "seed": int(args.seed), "bg_ratio": float(args.bg_ratio),
        "boards": {"train": sorted(TRAIN_BOARDS), "val": sorted(VAL_BOARDS),
                   "heldout_zero_defect_training": ["01", "04"]},
        "clip_rule": f"keep if inter_area >= {MIN_KEEP_RATIO} orig and side >= {MIN_KEEP_SIDE}px",
        "train_defect_tiles": 0, "val_defect_tiles": 0,
        "train_bg_tiles": 0, "val_bg_tiles": 0,
        "clip_kept_partial": 0, "clip_dropped_tiny": 0,
        "per_class_train": {n: 0 for n in CLASS_NAMES.values()},
        "per_class_val": {n: 0 for n in CLASS_NAMES.values()},
    }

    # ---- 1. Tile loi: 1 tile / bbox, label gom moi bbox sau clip ----
    ann_root = LEGACY_PKU_ROOT / "Annotations"
    if not ann_root.is_dir():
        raise FileNotFoundError(f"Thieu Annotations READ-ONLY: {ann_root}")
    for ann_dir in sorted(p for p in ann_root.iterdir() if p.is_dir()):
        for xp in sorted(ann_dir.glob("*_annotation.xml")):
            info = parse_xml_annotation(xp)
            if info is None:
                continue
            bid = info["board_id"]
            if bid in TRAIN_BOARDS:
                split, img_d, lbl_d = "train", train_img, train_lbl
            elif bid in VAL_BOARDS:
                split, img_d, lbl_d = "val", val_img, val_lbl
            else:
                continue  # 01/04 held-out
            img_p = resolve_defect_image(ann_dir.name, xp)
            if img_p is None:
                continue
            img = read_image_rgb(img_p)
            h, w = img.shape[:2]
            stem = xp.stem[:-len("_annotation")] if xp.stem.endswith("_annotation") else xp.stem
            for obj in info["objects"]:
                x0, y0 = choose_tile_around_bbox(w, h, obj["bbox"], ts)
                tile = img[y0:y0 + ts, x0:x0 + ts]
                if tile.shape[0] != ts or tile.shape[1] != ts:
                    continue
                tstem = f"{obj['class_name']}_{bid}_{stem}_x{x0}_y{y0}"
                save_tile_png(tile, img_d / f"{tstem}.png")
                lines: list[str] = []
                for other in info["objects"]:
                    yb = clip_bbox_to_tile(other["bbox"], x0, y0, ts)
                    if yb is None:
                        stats["clip_dropped_tiny"] += 1
                        continue
                    # dem bbox bi cat ria nhung van giu (khac ban cu: cu drop han)
                    ox1, oy1, ox2, oy2 = other["bbox"]
                    if ox1 < x0 or oy1 < y0 or ox2 > x0 + ts or oy2 > y0 + ts:
                        stats["clip_kept_partial"] += 1
                    cx, cy, bw, bh = yb
                    lines.append(f"{other['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    if split == "train":
                        stats["per_class_train"][other["class_name"]] += 1
                    else:
                        stats["per_class_val"][other["class_name"]] += 1
                write_label(lines, lbl_d / f"{tstem}.txt")
                if split == "train":
                    stats["train_defect_tiles"] += 1
                else:
                    stats["val_defect_tiles"] += 1

    # ---- 2. FIX 2: 25% background tiles cho train (+ vai tile cho val) ----
    n_bg_train = int(round(stats["train_defect_tiles"] * float(args.bg_ratio)))
    for i, (tile, bid) in enumerate(sample_background_tiles(
            n_bg_train, TRAIN_BOARDS, "bg", ts, stride, rng)):
        tstem = f"bg_{bid}_n{i:04d}"
        save_tile_png(tile, train_img / f"{tstem}.png")
        write_label([], train_lbl / f"{tstem}.txt")
        stats["train_bg_tiles"] += 1

    for i, (tile, bid) in enumerate(sample_background_tiles(
            int(args.val_bg_tiles), VAL_BOARDS, "bg", ts, stride, rng)):
        tstem = f"bg_{bid}_v{i:04d}"
        save_tile_png(tile, val_img / f"{tstem}.png")
        write_label([], val_lbl / f"{tstem}.txt")
        stats["val_bg_tiles"] += 1

    # ---- 3. data.yaml + stats.json ----
    data_yaml = {"path": str(out_root).replace("\\", "/"),
                 "train": "train/images", "val": "val/images",
                 "names": {i: CLASS_NAMES[i] for i in sorted(CLASS_NAMES)}}
    import yaml
    yaml_p = assert_writable_ok(out_root / "data.yaml")
    yaml_p.parent.mkdir(parents=True, exist_ok=True)
    yaml_p.write_text(yaml.dump(data_yaml, allow_unicode=True, sort_keys=False),
                      encoding="utf-8")
    stats_p = assert_writable_ok(out_root / "stats.json")
    stats_p.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    n_tr = stats["train_defect_tiles"] + stats["train_bg_tiles"]
    n_va = stats["val_defect_tiles"] + stats["val_bg_tiles"]
    print(f"[prepare] train: {stats['train_defect_tiles']} defect + "
          f"{stats['train_bg_tiles']} bg = {n_tr} tiles")
    print(f"[prepare] val:   {stats['val_defect_tiles']} defect + "
          f"{stats['val_bg_tiles']} bg = {n_va} tiles")
    print(f"[prepare] clip kept-partial={stats['clip_kept_partial']} "
          f"dropped-tiny={stats['clip_dropped_tiny']}")
    print(f"[prepare] wrote {yaml_p} + {stats_p}")
    print("[prepare_yolo_dataset] DONE")


if __name__ == "__main__":
    main()
