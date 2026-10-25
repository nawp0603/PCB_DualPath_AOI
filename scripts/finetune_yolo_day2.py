"""Ngay 2 — Fine-tune YOLO11n nhe 15-20 epoch tren data/yolo_day2.

Dau vao : assets/weights/yolo11n_base.pt (ban copy cua best.pt goc 50 epoch,
          KHONG ghi de) + data/yolo_day2/data.yaml (clip fix + 25% bg tiles).
Dau ra  : assets/weights/yolo11n_day2.pt (file MOI).

Doc quyen GPU: CODER chi smoke-test (VD --epochs 1 --max-train-images 8).
Train that 15-20 epoch la viec cua VERIFIER.

Vi du:
  python scripts/finetune_yolo_day2.py --help
  python scripts/finetune_yolo_day2.py --epochs 1 --max-train-images 8 --out data/smoke_day2.pt   # smoke-test
  python scripts/finetune_yolo_day2.py --epochs 15                                               # train that (Verifier)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.resource_paths import (  # noqa: E402
    LOCAL_BASE_PT,
    NEW_PROJECT_DIR,
    assert_writable_ok,
)

import yaml  # noqa: E402

BASE_PT = LOCAL_BASE_PT
DATA_YAML = NEW_PROJECT_DIR / "data" / "yolo_day2" / "data.yaml"
DEFAULT_OUT = NEW_PROJECT_DIR / "assets" / "weights" / "yolo11n_day2.pt"

EXPECTED_NAMES = {
    0: "open_circuit",
    1: "short",
    2: "mouse_bite",
    3: "spur",
    4: "spurious_copper",
    5: "missing_hole",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ngay 2: fine-tune YOLO -> yolo11n_day2.pt.")
    p.add_argument("--weights", type=Path, default=BASE_PT)
    p.add_argument("--data", type=Path, default=DATA_YAML)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--project", type=Path,
                   default=NEW_PROJECT_DIR / "runs" / "finetune_day2")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-train-images", type=int, default=None,
                   help="Smoke-test: gioi han so anh train (VD 8). Train that: bo qua.")
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_parser().parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Thieu weights goc: {weights}")
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Thieu {data_yaml} — chay scripts/prepare_yolo_dataset.py truoc.")
    out_path: Path = assert_writable_ok(Path(args.out))
    # Ultralytics noi project tuong doi vao runs_dir (runs/detect/...) lam lac
    # best.pt -> bat buoc resolve absolute trong project moi.
    project: Path = assert_writable_ok(Path(args.project).resolve())

    data_arg = str(data_yaml)
    if args.max_train_images is not None:
        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        src_train = (data_yaml.parent / cfg["train"]).resolve()
        sub = data_yaml.parent / "train_smoke"
        sub_img, sub_lbl = sub / "images", sub / "labels"
        sub_img.mkdir(parents=True, exist_ok=True)
        sub_lbl.mkdir(parents=True, exist_ok=True)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        all_imgs = sorted(p for p in src_train.glob("*")
                          if p.is_file() and p.suffix.lower() in exts)
        import random
        chosen = random.Random(int(args.seed)).sample(
            all_imgs, min(int(args.max_train_images), len(all_imgs)))
        for im in chosen:
            shutil.copy2(im, sub_img / im.name)
            lb = src_train.parent / "labels" / (im.stem + ".txt")
            if lb.exists():
                shutil.copy2(lb, sub_lbl / lb.name)
        cfg["train"] = "train_smoke/images"
        sub_yaml = data_yaml.parent / "data_smoke.yaml"
        assert_writable_ok(sub_yaml)
        sub_yaml.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False),
                            encoding="utf-8")
        data_arg = str(sub_yaml)
        print(f"[finetune] SMOKE subset: {len(chosen)} anh -> {sub_yaml}")

    import torch
    from ultralytics import YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu" \
        if args.device == "auto" else args.device
    print(f"[finetune] device={device} epochs={args.epochs} imgsz={args.imgsz} "
          f"batch={args.batch} lr0={args.lr0}")
    print(f"[finetune] weights={weights} data={data_arg}")

    model = YOLO(str(weights))
    model.train(data=data_arg, epochs=int(args.epochs), imgsz=int(args.imgsz),
                batch=int(args.batch), lr0=float(args.lr0), seed=int(args.seed),
                device=device, project=str(project), verbose=False)

    cands = sorted(project.glob("**/weights/best.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise SystemExit("Khong tim thay best.pt sau train")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cands[0], out_path)

    check = YOLO(str(out_path))
    assert check.names == EXPECTED_NAMES, f"Class names mismatch: {check.names}"
    print(f"[finetune] Saved {out_path}, classes={check.names}")
    print("[finetune_yolo_day2] DONE")


if __name__ == "__main__":
    main()
