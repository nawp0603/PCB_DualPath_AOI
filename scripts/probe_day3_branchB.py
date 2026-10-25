"""Ngay 3 FIX — probe chan doan Nhanh B (CPU-only, KHONG dung GPU/YOLO).

Muc dich:
  * BUG-DAY3-01: do residual tai vung GT theo tung lop loi, tach ro nguyen nhan
    miss: mask phu, nguong seed, do lon blob, sai lech tam blob, nhieu camera.
  * BUG-DAY3-02: do latency tung buoc cua align (ORB detect / match / RANSAC).

Chay:
  <venv>/python.exe scripts/probe_day3_branchB.py --max-images 6 --classes short,spurious_copper
  <venv>/python.exe scripts/probe_day3_branchB.py --timing

Khong ghi file nao; chi in stdout. So lieu chi de chan doan — Verifier do lai
chinh thuc bang tests/bench_day3_branchB.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.resource_paths import LEGACY_PKU_ROOT, LOCAL_TEMPLATES  # noqa: E402
import src.reference.board_id_and_align as ba  # noqa: E402
import src.reference.tolerance_diff as td  # noqa: E402

CLASS_NAMES = {0: "open_circuit", 1: "short", 2: "mouse_bite", 3: "spur",
               4: "spurious_copper", 5: "missing_hole"}


def read_gray(p: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def read_bgr(p: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def collect() -> list[dict]:
    """Doc XML 01/04 (giong evaluate.py) — chi doc."""
    import xml.etree.ElementTree as ET
    items = []
    ann_root = LEGACY_PKU_ROOT / "Annotations"
    for ann_dir in sorted(p for p in ann_root.iterdir() if p.is_dir()):
        for xp in sorted(ann_dir.glob("*_annotation.xml")):
            try:
                r = ET.parse(xp).getroot()
            except ET.ParseError:
                continue
            fn = r.find("filename")
            if fn is None or not fn.text:
                continue
            bid = Path(fn.text.strip()).stem.split("_")[0].zfill(2)
            if bid not in ("01", "04"):
                continue
            stem = xp.stem
            if stem.endswith("_annotation"):
                stem = stem[: -len("_annotation")]
            img = None
            for cand in (stem + ".jpg", stem + ".JPG", fn.text.strip()):
                q = LEGACY_PKU_ROOT / "images" / ann_dir.name / cand
                if q.exists():
                    img = q
                    break
            if img is None:
                continue
            gt = []
            for o in r.findall("object"):
                n, b = o.find("name"), o.find("bndbox")
                if n is None or b is None or not n.text:
                    continue
                cid = {"open_circuit": 0, "short": 1, "mouse_bite": 2, "spur": 3,
                       "spurious_copper": 4, "missing_hole": 5}.get(n.text.strip().lower())
                if cid is None:
                    continue
                box = [float(b.find("xmin").text), float(b.find("ymin").text),
                       float(b.find("xmax").text), float(b.find("ymax").text)]
                gt.append((cid, box))
            if gt:
                items.append({"path": str(img), "board": bid, "gt": gt})
    items.sort(key=lambda d: d["path"])
    return items


def perturb(img, boxes, seed):
    """Sao y evaluate.py::perturb_image_and_boxes (de so sanh dung seed)."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    dx, dy = float(rng.uniform(-25, 25)), float(rng.uniform(-25, 25))
    ang = float(rng.uniform(-2.0, 2.0))
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
    m[0, 2] += dx
    m[1, 2] += dy
    h_aff = np.vstack([m, [0.0, 0.0, 1.0]])
    j = 0.008
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-j, j, size=(4, 2)).astype(np.float32) * np.float32([w, h])
    hp, _ = cv2.findHomography(src, dst)
    h_all = (hp if hp is not None else np.eye(3)) @ h_aff
    warped = cv2.warpPerspective(img, h_all, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    gain, bias = float(rng.uniform(0.85, 1.15)), float(rng.uniform(-10.0, 10.0))
    f = warped.astype(np.float32) * gain + bias
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    rx, ry = (xs - w / 2) / (w / 2), (ys - h / 2) / (h / 2)
    rr = np.sqrt(rx ** 2 + ry ** 2) / np.sqrt(2.0)
    f *= (1.0 - 0.15 * np.clip(rr, 0, 1) ** 2)[..., None]
    f += rng.normal(0, float(rng.uniform(3.0, 5.0)), size=f.shape)
    out = np.clip(f, 0, 255).astype(np.uint8)
    nb = []
    if boxes:
        pts = np.float32([[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
                          for b in boxes]).reshape(-1, 1, 2)
        q = cv2.perspectiveTransform(pts, h_all.astype(np.float32)).reshape(-1, 4, 2)
        for p in q:
            x1, y1 = float(np.clip(p[:, 0].min(), 0, w - 1)), float(np.clip(p[:, 1].min(), 0, h - 1))
            x2, y2 = float(np.clip(p[:, 0].max(), 0, w - 1)), float(np.clip(p[:, 1].max(), 0, h - 1))
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                nb.append([x1, y1, x2, y2])
    return out, nb


def residual_gray(corr_f32, tpl_f32, tau=3):
    k = (2 * tau + 1, 2 * tau + 1)
    ker = np.ones(k, np.float32)
    tmin, tmax = cv2.erode(tpl_f32, ker), cv2.dilate(tpl_f32, ker)
    return np.maximum(tmin - corr_f32, 0.0) + np.maximum(corr_f32 - tmax, 0.0)


def robust_sigma(res: np.ndarray, mask: np.ndarray | None = None) -> float:
    v = res[mask] if mask is not None else res.ravel()
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return 1.4826 * mad


def gt_in_template(gt_boxes, h):
    out = []
    for b in gt_boxes:
        c = np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
        q = cv2.perspectiveTransform(c.reshape(-1, 1, 2), h).reshape(-1, 2)
        out.append([float(q[:, 0].min()), float(q[:, 1].min()),
                    float(q[:, 0].max()), float(q[:, 1].max())])
    return out


def probe_recall(args):
    cache = ba.load_cache()
    templates_g = {b: read_gray(LOCAL_TEMPLATES / f"{b}.JPG") for b in cache["board_ids"]}
    items = collect()
    classes = {CLASS_NAMES[c] for c in CLASS_NAMES} if not args.classes else set(args.classes)
    if args.max_images:
        items = items[: args.max_images]
    print(f"[probe] items={len(items)} classes={sorted(classes)}")

    stats: dict[str, dict] = {}
    for idx, it in enumerate(items):
        img = read_bgr(Path(it["path"]))
        gt0 = [list(b) for _, b in it["gt"]]
        cls0 = [CLASS_NAMES[c] for c, _ in it["gt"]]
        img_c, gt_c = perturb(img, gt0, 20261023 + 101 + idx)
        for tag, im, gt, cls in (("standard", img, gt0, cls0), ("camera", img_c, gt_c, cls0)):
            r = ba.identify_and_align(im, cache, fast=True)
            if r["board_id"] != it["board"]:
                print(f"  [!] board mismatch {it['path']}")
                continue
            tpl_g = templates_g[r["board_id"]]
            if r["warped"].shape[:2] != tpl_g.shape:
                tpl_g = cv2.resize(tpl_g, (r["warped"].shape[1], r["warped"].shape[0]),
                                   interpolation=cv2.INTER_AREA)
            wg = cv2.cvtColor(r["warped"], cv2.COLOR_BGR2GRAY).astype(np.float32)
            tf = tpl_g.astype(np.float32)
            gtb = gt_in_template(gt, r["H"])

            struct = td.structure_mask_from_template(tpl_g)
            corr = td.correct_lowfreq_gain(wg, tf)
            corr_dn = cv2.GaussianBlur(corr, (3, 3), 0.8)
            res_raw = residual_gray(corr, tf)
            res_dn = residual_gray(corr_dn, tf)
            rsig = robust_sigma(res_raw)
            rsig_dn = robust_sigma(res_dn)
            cur = td.run_branch_b(r["warped"], tpl_g)
            blobs = cur["blobs"]
            cents = (ba.project_points(np.float32([b["centroid_tpl"] for b in blobs]),
                                       r["H_inv"]) if blobs else np.zeros((0, 2)))

            for gi, (gbox, cname) in enumerate(zip(gtb, cls)):
                if cname not in classes:
                    continue
                x1, y1, x2, y2 = [int(round(v)) for v in gbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(expw := res_raw.shape[1] - 1, x2), min(res_raw.shape[0] - 1, y2)
                reg = res_raw[y1:y2 + 1, x1:x2 + 1]
                regd = res_dn[y1:y2 + 1, x1:x2 + 1]
                mreg = struct[y1:y2 + 1, x1:x2 + 1]
                inbox = [bi for bi, p in enumerate(cents)
                         if gbox[0] - 3 <= p[0] <= gbox[2] + 3 and
                         gbox[1] - 3 <= p[1] <= gbox[3] + 3]
                hit = len(inbox) > 0
                rank = (inbox[0] + 1) if inbox else 0  # vi tri blob trung GT (1-based)
                # blob nao gan GT nhat (khoang cach centroid -> tam GT)
                cx, cy = (gbox[0] + gbox[2]) / 2, (gbox[1] + gbox[3]) / 2
                dmin = min([float(np.hypot(p[0] - cx, p[1] - cy)) for p in cents] + [1e9])
                s = stats.setdefault(cname, {"n": 0, "hit": 0, "maskcov": [],
                                             "resmax": [], "resmax_dn": [], "resmean": [],
                                             "hit_if_nomask": 0, "hit_if_dn": 0,
                                             "area_gt": [], "dmin": [], "rsig": [],
                                             "rsig_dn": []})
                s["n"] += 1
                s["hit"] += int(hit)
                s["maskcov"].append(float(mreg.mean()) if mreg.size else 0.0)
                s["resmax"].append(float(reg.max()) if reg.size else 0.0)
                s["resmax_dn"].append(float(regd.max()) if regd.size else 0.0)
                s["resmean"].append(float(reg.mean()) if reg.size else 0.0)
                s["area_gt"].append(float(reg.size))
                s["dmin"].append(dmin)
                s["rsig"].append(rsig)
                s["rsig_dn"].append(rsig_dn)
                # gia thuyet: neu KHONG ap structure mask, blob co bat duoc khong?
                diff_nomask = residual_gray(corr_dn, tf)
                bl_nm = td.extract_blobs(diff_nomask, td.MIN_AREA_PX2, td.SIGMA_FLOOR,
                                         td.SEED_K, td.TOP_K)
                hit_nm = False
                for b in bl_nm:
                    p = ba.project_points(np.float32([b["centroid_tpl"]]), r["H_inv"])[0]
                    if gbox[0] - 3 <= p[0] <= gbox[2] + 3 and gbox[1] - 3 <= p[1] <= gbox[3] + 3:
                        hit_nm = True
                        break
                s["hit_if_nomask"] += int(hit_nm)
                s["hit_if_dn"] += int(hit)

    print(f"\n{'class':<17}{'n':>4}{'hit':>5}{'hitNomask':>11}{'maskcov':>9}"
          f"{'resmax':>8}{'resmaxDN':>9}{'rsig':>7}{'rsigDN':>7}{'areaGT':>8}{'dmin':>8}")
    for c in sorted(stats):
        s = stats[c]
        print(f"{c:<17}{s['n']:>4}{s['hit']:>5}{s['hit_if_nomask']:>11}"
              f"{np.mean(s['maskcov']):>9.2f}{np.mean(s['resmax']):>8.1f}"
              f"{np.mean(s['resmax_dn']):>9.1f}{np.mean(s['rsig']):>7.2f}"
              f"{np.mean(s['rsig_dn']):>7.2f}{np.mean(s['area_gt']):>8.0f}"
              f"{np.median(s['dmin']):>8.1f}")


def probe_timing():
    cache = ba.load_cache()
    img = read_bgr(Path(collect()[0]["path"]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"[timing] input={gray.shape}")
    for scale in (1.0, 0.5, 0.25):
        sm = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for nf in (8000, 2500, 1200):
            orb = cv2.ORB_create(nfeatures=nf)
            t0 = time.perf_counter()
            kp, des = orb.detectAndCompute(sm, None)
            t1 = time.perf_counter()
            if des is None:
                continue
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            tot_match = 0.0
            for ti in range(3):
                t2 = time.perf_counter()
                _ = bf.knnMatch(des, cache["descriptors"][ti], k=2)
                tot_match += (time.perf_counter() - t2) * 1000
            print(f"  scale={scale} nf={nf} kp={len(kp)} detect={1000*(t1-t0):.1f}ms "
                  f"match1tpl_avg={tot_match/3:.1f}ms")
    # full path breakdown
    t0 = time.perf_counter()
    _ = ba.identify_and_align(img, cache, fast=False)
    print(f"  full total={1000*(time.perf_counter()-t0):.1f}ms")
    t0 = time.perf_counter()
    _ = ba.identify_and_align(img, cache, fast=True)
    print(f"  fast total={1000*(time.perf_counter()-t0):.1f}ms")
    t0 = time.perf_counter()
    _ = ba.identify_and_align(img, cache, fast=True)
    print(f"  fast total (2nd)={1000*(time.perf_counter()-t0):.1f}ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--classes", type=str, default="")
    ap.add_argument("--timing", action="store_true")
    args = ap.parse_args()
    args.classes = [c for c in args.classes.split(",") if c]
    if args.timing:
        probe_timing()
    else:
        probe_recall(args)


if __name__ == "__main__":
    main()
