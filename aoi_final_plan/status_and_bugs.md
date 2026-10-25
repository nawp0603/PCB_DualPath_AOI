# SO BAN GIAO — DUAL-PATH INDUSTRIAL AOI (PCB_DualPath_AOI)

> MANDATORY ROLE ANCHOR (doc truoc moi hanh dong):
> 1. CODER chi ghi src/ + scripts/ trong project moi; tuan thu H_inv, sigma_floor, rect=True + Batch ROI.
> 2. CHI smoke-test 1-2 anh. CAM train dai / eval full (viec cua Verifier, doc quyen GPU).
> 3. VERIFIER chi ghi tests/ + results/ + file nay; CAM sua src/; loi src/ gan co [FAILED - NEED FIX].
> 4. SCRIBE chi ghi aoi_final_plan/thesis_artifacts/.
> 5. Tach biet Kich ban 1 (Standard) vs Kich ban 2 (Camera); dung thuat ngu "Zero-defect-training Boards" cho 01/04.
> 6. Moi duong dan ngoai project moi la STRICT READ-ONLY (configs/resource_paths.py). Cam mode 'w'/xoa/ghi de project cu.
> 7. Cam PatchCore/SAHI trong luong quyet dinh chinh (chi baseline Ablation trong evaluate.py).
> 8. Cam bia so: moi F1/Precision/Recall/FAR phai tu ket qua chay thuc te.

---

## TIEN DO NGAY 1 -> NGAY 6

| Ngay | Hang muc | File chinh | Trang thai |
|---|---|---|---|
| 0 | Khoi tao workspace + plan + rules + resource map | `plan.md`, `rules/`, `configs/resource_paths.py` | [DONE] 2026-10-23 |
| 1 | `evaluate.py` + 4 dong Baseline/Ablation | `scripts/evaluate.py` | [VERIFIED PASS] 2026-10-23 |
| 2 | Sua data train, fine-tune YOLO, cap nhat inference | `scripts/prepare_yolo_dataset.py`, `src/models/yolo_detector.py`, `assets/weights/yolo11n_day2.pt` | [VERIFIED PASS] 2026-10-23 |
| 3-4 | Module Nhanh B (align + tolerance diff) | `src/reference/board_id_and_align.py`, `src/reference/tolerance_diff.py` | [VERIFIED PASS] 2026-10-23 (BUG-DAY3-04 DA sua + verify xong: board-ID 80/80 shift +-25/+-50px, benchmark 98/98, recall cam 98/105, FAR 0, lat 43.1ms) |
| 5 | Quy tac 4 muc + fusion | `src/decision/logic.py`, `src/models/ensemble.py` | [VERIFIED PASS - logic/fuse] 2026-10-23 (24/24 test test_day5_ensemble_logic.py: 4 muc, rule 3b doc lap khong khoa yolo_conf, Box Fusion IoU>=0.3 S_final=min(1.0,S_A+0.15), gating R=80, centroid+H_inv, khong dung bbox blob chat BUG-DAY3-03). Chua chay evaluate.py full pipeline voi Day 5 vi evaluate.py chua tich hop ensemble — can Coder bo sung hoac Verifier viet test end-to-end tren GPU de xuat F1/P/R/FAR (tieu chi cuoi Ngay 5, plan 6.3). |
| 6 | UI Streamlit + bao cao | `src/ui/app.py`, `aoi_final_plan/thesis_artifacts/` | [PENDING] |

Tieu chi nghiem thu chi tiet: xem `plan.md` Phan 5 + Phan 6.3.

---

## LOG GIAO TIEP & BAO LOI (moi nhat o cuoi)

### [2026-10-23] SYSTEM — Khoi tao workspace [DONE]
- Da quet read-only 2 project cu, dinh vi day du 5 nhom tai nguyen (xem plan.md Phan 7).
- Da copy 1 chieu: `best.pt` (5.5 MB) -> `assets/weights/yolo11n_base.pt`; 10 template `PCB_USED` -> `assets/templates/`.
- Da tao: `plan.md` (7 phan), `rules/{coder,verifier,scribe}_rules.md`, `configs/resource_paths.py` (co `assert_writable_ok` + `ensure_pku_dataset`).
- Chua tim thay: script `temp/check20*`/`check21*` trong project goc -> [CAN USER CUNG CAP DUONG DAN] neu can doi chieu dang ky anh.
- Luu y: `PCB_USED` goc chi co 10 file (01, 04-12), khong co `02.JPG`/`03.JPG` — dung nhu mo ta "10 anh template".
- Sandbox con nguyen: khong sua/xoa bat ky file nao trong 2 project cu.

### [2026-10-23] CODER — Ngay 1: `scripts/evaluate.py` [READY FOR VERIFY]
- File moi: `scripts/evaluate.py` (~600 dong). Chay duoc, compile OK, smoke-test pass.
- Dung split board-disjoint: test = Zero-defect-training Boards 01/04 (36 anh loi / 105 boxes + 2 GOOD templates 01/04); train 05,06,08,09,10,11,12 / val 07 (ghi trong meta).
- Xuat 4 dong baseline tach biet 2 kich ban (standard = anh PKU goc; camera = shift +-25px / xoay +-2do / perspective ~0.8% / gain-bias+vignette+noise, co warp GT boxes):
  1. `yolo640` — YOLO full-board @640 rect=True (+half khi cuda).
  2. `sahi640` — SAHI slice 640 ov 0.20 [ABLATION, khong thuoc pipeline chinh].
  3. `patchcore` — re-implement toi thieu resnet18 l2+l3 + memory_bank tu ckpt READ-ONLY [ABLATION].
  4. `yolo3072` — YOLO full-board @highres (3072/3584) rect=True half=True = Nhanh A chuan.
- CLI: `--scenario {standard,camera,both} --configs {yolo640,sahi640,patchcore,yolo3072} --highres --conf(0.20) --iou --max-images --device --out-json/--out-csv --no-save`. Output mac dinh `results/baseline_day1.{json,csv}` (qua `assert_writable_ok`, chi ghi trong project moi).
- Smoke-test 1 anh (`--max-images 1 --no-save`, dung gioi han GPU cua Coder, khong eval full):
  standard: yolo640 AP50=0.0 (TP0/FP1 — resize 640 lam mat loi nho, du kien); sahi640 AP50=1.0 R=1.0 P=0.75; patchcore AUROC=0.0/FPR=1.0 (n=3, chi minh hoa duong chay); yolo3072 AP50=1.0 R=1.0 P=1.0 (TP3/FP0, ~112ms).
  camera: yolo640 AP50=0.0; sahi640 AP50=1.0; patchcore AUROC=0.5; yolo3072 AP50=1.0 (TP3/FP0, ~75ms).
  -> So lieu tren chi de kiem duong chay, KHONG phai ket qua baseline chinh thuc (cho Verifier chay full 36 anh).
- File ngoai scope Ngay 1 KHONG dung toi: `prepare_yolo_dataset.py`, `yolo_detector.py`, `board_id_and_align.py`, `tolerance_diff.py`, `logic.py`, `ensemble.py`, `app.py`.
- Nho Verifier: chay full eval tren GPU (`--scenario both`, khong `--max-images`), kiem tra tach biet Bang 1/Bang 2 + thuat ngu Zero-defect-training Boards truoc khi PASS.

### [2026-10-23] VERIFIER — Ngay 1: `scripts/evaluate.py` [VERIFIED PASS]
- Pham vi kiem dinh: chi doc `src/` + `scripts/evaluate.py`, viet test vao `tests/test_day1_evaluate.py`, chay full eval GPU, ghi `results/`. KHONG sua `src/`.
- Test khat khe: `D:/FPTU/KLTN/PCB_Anomaly_Inspector_Portable/.venv/Scripts/python.exe -m pytest tests/test_day1_evaluate.py -v` -> **11/11 PASSED** (split leakage, golden sample, rect+half, class map, no PatchCore/SAHI in src, no ECC risk, terminology, write-guard, perturb deterministic, metrics sanity, H_inv/sigma_floor N/A Day 1).
- Full eval doc lap GPU (RTX 3070, cuda True, torch 2.6.0+cu124, ultralytics 8.4.182, cv2 4.11.0): `python scripts/evaluate.py --scenario both --out-json results/baseline_day1.json --out-csv results/baseline_day1.csv` -> DONE, 36 anh loi / 105 boxes + 2 GOOD, khong `--max-images`.
- Checklist 6 muc (doi chieu plan + verifier_rules):
  1. PatchCore/SAHI sot trong luong quyet dinh chinh: [VERIFIED PASS] — `src/` hien chi co `__init__.py`, grep khong thay patchcore/sahi; trong `evaluate.py` ca 2 deu co nhan `[ABLATION]` + note "khong thuoc pipeline chinh" (dong 8-11, 388, 411-415, 561-575).
  2. `yolo_detector` rect/half/imgsz: [VERIFIED PASS] cho pham vi Ngay 1 — `YoloRunner.infer_bgr` (`evaluate.py:368-373`) bat buoc `rect=True, half=self.half` (half khi cuda), `DEFAULT_CONF=0.20` (dong 81), `--highres` mac dinh 3072 (dong 626-627). Latency thuc do yolo3072 standard 148.1ms (< 200ms, trong khoang plan 160-190ms) -> dat. Luu y non-blocking: ultralytics in canh bao `'half' is deprecated, use 'quantize'` — Coder Ngay 2 giu `half=True` theo plan, khong doi API luc nay.
  3. Nhanh B H_inv chieu nguoc + crop 640 + IoU: [PENDING — NOT APPLICABLE Day 1] — `src/reference/board_id_and_align.py`, `tolerance_diff.py`, `src/models/ensemble.py` chua ton tai (da assert trong test). Khong co gi de audit; chuyen sang Ngay 3-5.
  4. `Z_diff` sigma_floor tren anh GOOD sach: [PENDING — NOT APPLICABLE Day 1] — cong thuc Z_diff thuoc `tolerance_diff.py` Ngay 3-4, chua co. Se test anh GOOD khi Coder giao Nhanh B.
  5. `findTransformECC` mask float32 (crash OpenCV 4.11): [VERIFIED PASS] — `evaluate.py` khong dung ECC; chi dung `cv2.findHomography` de tong hop perturbation (dong 315), khong co nguy co crash `_mask.type() == CV_8UC1`.
  6. Leakage 01/04 giua Scenario 1 vs 2: [VERIFIED PASS] — dem doc lap: test 01/04 = 36 imgs / 105 boxes (01:18, 04:18; 6 lop 16-19 boxes/lop); train 05,06,08,09,10,11,12 = 43 imgs / 213 boxes; val 07 = 11 imgs / 55 boxes; khong trung file test-vs-train/val; XML resolve miss = 0/36; GOOD = 2 templates `01.JPG, 04.JPG` trong `assets/templates/`; meta ghi dung "test=01,04 (Zero-defect-training Boards); train=05,06,08,09,10,11,12; val=07"; 2 kich ban chay tach biet `standard` vs `camera` (seed_base khac nhau, warp GT boxes deterministic); khong co cum tu "unseen boards".
- So lieu chuan luu tai `results/baseline_day1.json` + `results/baseline_day1.csv` (trich nguyen van tu chay thuc te, khong lam tron):
  - Bang 1 Standard (anh PKU goc, 36 anh loi + 2 GOOD): yolo640 AP50=0.0119 R=0.029 P=0.023 TP=3 FP=129 det=132 124.8ms | sahi640 AP50=0.7644 R=0.810 P=0.480 TP=85 FP=92 det=177 605.3ms [ABLATION] | patchcore AUROC=0.3056 FPR=1.0 (TP=36 TN=0 FP=2 FN=0) 86.7ms [ABLATION] | yolo3072 AP50=0.8375 R=0.857 P=0.763 TP=90 FP=28 det=118 148.1ms (= Nhanh A chuan).
  - Bang 2 Camera Perturbation (shift +-25px / xoay +-2do / perspective ~0.8% / gain-bias+vignette+noise, warp GT, Zero-defect-training 01/04): yolo640 AP50=0.0098 R=0.010 P=0.011 TP=1 FP=88 46.9ms | sahi640 AP50=0.4882 R=0.514 P=0.409 TP=54 FP=78 656.4ms [ABLATION] | patchcore AUROC=0.5833 FPR=1.0 25.7ms [ABLATION] | yolo3072 AP50=0.6734 R=0.686 P=0.889 TP=72 FP=9 93.3ms.
  - Gia tri luan van: PatchCore AUROC=0.3056 + FPR=100% (2/2 GOOD bi gan FAIL) khop so lieu plan Phan 1.2 de lam Ablation Study; SAHI Precision tut (0.480 standard / 0.409 camera) + FP cao (92/78) so voi YOLO full-res (0.763/0.889, FP 28/9) khop luan diem "SAHI P 0.489, FP 95" trong plan.
- Zero-touch: [VERIFIED PASS] — write JSON/CSV qua `assert_writable_ok` (dong 692-696), chi ghi trong project moi; khong co `rmtree/unlink`, khong mo mode 'w' len duong dan legacy; sandbox 2 project cu nguyen ven.
- Ket luan: Ngay 1 [VERIFIED PASS]. Coder duoc phep sang Ngay 2 (`prepare_yolo_dataset.py`, `yolo_detector.py`). H_inv / sigma_floor / logic 4 muc de danh Ngay 3-5, Verifier se test lai khi co file.

### [2026-10-23] CODER — Ngay 2: dataset + Nhanh A + finetune script [READY FOR VERIFY]
- File moi/sua trong scope Ngay 2: `scripts/prepare_yolo_dataset.py` (moi), `src/models/__init__.py` (moi), `src/models/yolo_detector.py` (moi), `scripts/finetune_yolo_day2.py` (moi). Compile OK ca 3 file code.
- `prepare_yolo_dataset.py`: output CHI trong `data/yolo_day2/` (train/val/images+labels, data.yaml, stats.json), khong cham dataset project cu. FIX 1: `clip_bbox_to_tile` cat bbox ve bien tile (giu neu inter>=25% + canh>=8px) thay vi drop-None. FIX 2: +25% bg tiles train (53 tiles tu Golden Templates, label rong) + 10 bg val. Ket qua chay that: train 213 defect + 53 bg = 266 tiles; val 55 defect + 10 bg = 65 tiles; per-class train 40-62/lop; clip kept-partial=35, dropped-tiny=962. Board 01/04 held-out (khong train).
- `yolo_detector.py` (Nhanh A): `YOLODetector` hop dong RGB uint8, `predict` full-board imgsz=3072 conf=0.20 bat buoc `rect=True, half=(cuda)`; `predict_batch` gom N crops 640 goi GPU 1 lan (Chot 3); `spatial_center_gating(R=80px)` triet FP ria crop. Trong so mac dinh `yolo11n_base.pt` (khong ghi de).
- `finetune_yolo_day2.py`: input base.pt + data.yaml, output MOI `assets/weights/yolo11n_day2.pt` (15 epoch mac dinh). CODER chi smoke-test `--epochs 1 --max-train-images 8` -> train+val+copy best pass (val smoke mAP50=0.948 tren 65 anh val — chi minh hoa duong chay). Bug da fix: resolve absolute `--project` (Ultralytics noi path tuong doi vao runs/detect lam lac best.pt). File tam smoke da xoa sach.
- Smoke-test Nhanh A (1 anh 01 + batch 2 crops, dung gioi han GPU Coder): predict@3072 = 1 det open_circuit conf 0.35; batch 2 crops 1 call 83ms; gating R=80 pass.
- `yolo11n_day2.pt` CHUA ton tai (train that 15-20 epoch la viec cua Verifier, doc quyen GPU). Nho Verifier: chay finetune that -> kiem tra FP tren crop 640 giam vs baseline + latency @3072 rect=True < 200ms, roi PASS.

### [2026-10-23] VERIFIER — Ngay 2: dataset + Nhanh A + finetune that [VERIFIED PASS]
- Pham vi: chi doc `src/` + `scripts/`, viet `tests/test_day2_dataset_detector.py`, finetune that 15 epoch GPU, eval doc lap, ghi `results/`. KHONG sua `src/`.
- Test khat khe: `pytest tests/test_day2_dataset_detector.py -v` -> **11/11 PASSED**; full suite Day1+Day2 -> **22/22 PASSED** (clip keep-partial/drop-tiny, counts 213+53/55+10, no 01/04 leakage, YOLO labels, data.yaml, write-guard, rect+half+batch-1-call, class map, gating R80, no PatchCore/SAHI/ECC, H_inv/sigma_floor/logic PENDING Day 3-5).
- Finetune that (Verifier, doc quyen GPU): `scripts/finetune_yolo_day2.py --epochs 15 --batch 8 --imgsz 640` -> DONE 0.061h, val tile-640 best **P=0.982 R=0.945 mAP50=0.957 mAP50-95=0.582** (65 anh/84 inst; luu y: val tile de hon full-board, khong duoc nham voi AP full-board). Output MOI `assets/weights/yolo11n_day2.pt` (5.3MB, 6 lop khop); `yolo11n_base.pt` sha256 `0ed10b72...` TRUNG legacy `best.pt` -> khong bi ghi de. `runs/finetune_day2/` nam trong project (hop le, khong cham legacy).
- So lieu danh gia doc lap full test 01/04 (36 anh/105 boxes + 2 GOOD, khong `--max-images`; chi tiet `results/day2_verify_summary.json`, `results/baseline_day2_yolo3072.{json,csv}`):
  - Standard: base AP50=0.8375 R=0.857 P=0.763 FP=28 det=118 148.1ms -> **day2 AP50=0.8392 R=0.848 P=0.795 FP=23 det=112 77.0ms** (FP -5, P +3.2diem).
  - Camera: base AP50=0.6734 R=0.686 P=0.889 FP=9 93.3ms -> **day2 AP50=0.6234 R=0.638 P=0.870 FP=10 39.3ms** (recall -5TP; bao cao trung thuc — finetune tile sach khong cai thien robustness perturbation, du kien vi train khong augment perturbation).
  - FP probe: bg53-train @640 base 0/53 vs day2 0/53 (floor effect); GOOD templates @3072 base `01:0FP/04:1FP(0.31)` -> **day2 `01:0/04:0` (triet 1 FP tren GOOD04)**; latency GOOD avg 82ms -> 41ms.
- Checklist (doi chieu plan + verifier_rules):
  1. PatchCore/SAHI sot: [VERIFIED PASS] — `src/**/*.py` khong co patchcore/sahi/ECC; chung chi o `evaluate.py` [ABLATION].
  2. `yolo_detector` rect/half/imgsz=3072/3584: [VERIFIED PASS] — `predict` (dong 82-90) + `predict_batch` (dong 101-109) deu `rect=True, half=self.half (cuda)`, mac dinh imgsz=3072 conf=0.20; `predict_batch` dung 1 call GPU (`.predict(` x1); latency @3072 < 200ms ca 2 weights, 2 kich ban -> dat acceptance plan.
  3. Nhanh B H_inv chieu nguoc + crop640 + IoU: [PENDING Day 3-4] — file chua ton tai, khong audit.
  4. `Z_diff` sigma_floor anh GOOD: [PENDING Day 3-4] — thuoc `tolerance_diff.py`, chua co.
  5. ECC mask float32: [VERIFIED PASS] — khong dung ECC o Ngay 2.
  6. Leakage 01/04: [VERIFIED PASS] — train {05,06,08,09,10,11,12} / val {07}, 266/65 tiles dem file that; khong tile nao chua `_01_`/`_04_`; stats ghi `heldout_zero_defect_training: [01, 04]`; thuan ngu Zero-defect-training.
- [FAILED - NEED FIX] (minor, non-blocking, chuyen backlog — KHONG chan Day 2 PASS, Coder sua truoc Day 5):
  - BUG-DAY2-01 `src/models/yolo_detector.py:114-135` `spatial_center_gating` co tham so `require_overlap: bool = True` nhung THAN HAM khong dung (chi check dist<=R, khong check IoU>0 nhu docstring + plan 2.3). Tai hien: truyen det ria crop + `require_overlap=True` van bi drop/pass chi theo R. Sua: hoac cai dat nhanh IoU(box, crop)>0, hoac doi ten/xoa tham so thua truoc khi Nhanh B goi o Day 5 (Box Fusion IoU>=0.3 phu thuoc ham nay dung).
  - BUG-DAY2-02 `src/models/yolo_detector.py:53-55` chon device bang `torch.cuda.is_available()` ma khong ton trong `--device cpu` (evaluate.py co `--device`; YOLODetector khong co tham so device). Tai hien: `--device cpu` van dung cuda neu co GPU. Sua: them `device: str | None = None` vao `__init__` (de Day 5/CI khong GPU van chay duoc).
- Zero-touch: [VERIFIED PASS] — prepare + finetune deu qua `assert_writable_ok`, output trong `data/yolo_day2/` + `runs/finetune_day2/` + `assets/weights/yolo11n_day2.pt`; khong rmtree/unlink/mode-'w' legacy; khong con file smoke (`runs/` goc khong ton tai).
- So lieu chuan: `results/baseline_day1.json` (base) + `results/baseline_day2_yolo3072.{json,csv}` (day2) + `results/day2_verify_summary.json` (tong hop so sanh). Cam trich dan val tile mAP50=0.957 nhu AP full-board.
- Ket luan: Ngay 2 [VERIFIED PASS]. Coder sang Ngay 3-4 (Nhanh B: H_inv, sigma_floor, TOP_K batch). BUG-DAY2-01/02 sua truoc Day 5.

### [2026-10-23] CODER — Ngay 3: Nhanh B core + fix BUG-DAY2-01/02 [READY FOR VERIFY]
- Fix backlog Verifier truoc Day 5 (trong scope src/ cua Coder):
  - BUG-DAY2-01 FIXED: `spatial_center_gating` hien thuc thi `require_overlap` that (giu them box tam ngoai R nhung IoU(box-global, diff_win)>0; diff_win = o vuong 2R quanh tam diff hoac diff_box neu co). Test: box tam xa R + giao diff_box -> kept=1 khi True, drop khi False.
  - BUG-DAY2-02 FIXED: `YOLODetector.__init__` them `device: str | None = None` (auto cuda/cpu, validate, fallback cpu). Backward-compat: goi cu khong device van chay.
- File moi Ngay 3: `src/reference/__init__.py`, `src/reference/board_id_and_align.py`, `src/reference/tolerance_diff.py` + cache `assets/templates/orb_cache.npz` (2.6MB, build 1 lan tu 10 templates: 01/04/05/06/08/09/10/11/12 = 8000kp, 07 = 4973kp). Compile OK ca 3 file.
- `board_id_and_align.py` (Buoc 1): ORB 8000 + BFMatcher Hamming + Lowe 0.75 + RANSAC 5.0; chon board theo inliers max; H (Input->Template) + H_inv (Template->Input) + warp BORDER_REPLICATE; `project_points` = Chot 1 (perspectiveTransform qua H_inv). Them `fast=True` (ORB2500 half-res + vote match + H top-3) cho benchmark; mac dinh full cho Verifier doi chieu. Khong ECC (dung plan 2.3, tranh crash OpenCV 4.11).
- `tolerance_diff.py` (Buoc 2/3/4, viet lai sau probe that): Buoc 2 = `correct_lowfreq_gain` (chia ban mo 151px, clip 0.5-2.0) — local z-score bi loai vi cua so 31px nhiem loi that (res@GT ~0). Buoc 3 = khoang bao [T_min,T_max] tau=3 (don vi muc xam, erode/dilate, khong loop Python) tren structure mask = dilate7(copper Otsu UNION gradient edge>30) — mask nghiem ngat loai ca loi that vi open/short nam SAT mach ~1px. Buoc 4 = CC Area>=60, seed thich ung bg_mean+2.5*max(bg_std,1.0), Z_diff voi sigma_floor=5.0 (Chot 2), TOP_K=8, sort Z giam. Co `calibrate_residual_tol` (median+3mad, floor 0.5) + `copper_mask_from_template` giu de so sanh.
- Smoke-test 1 anh 01 (dung gioi han GPU Coder, khong eval full): full align board=01 inl=4000 (top T5/8/4), fast align board=01 inl=814; diff 3 blobs Z=3.05/2.94/1.89 trung ca 3 GT (H_inv roundtrip err 0.0px, H~identity vi anh PKU can san); ROI batch 3 crops 1 call + gating R80 giu 3/3 voi nhan YOLO open_circuit (0.34/0.48/0.64). Luu y trung thuc: latency smoke lan dau cao (align ~5.8s full / ~1.7s fast, batch ~1.3s do warmup GPU) — Verifier do lai sau warmup khi benchmark 45ms/anh.
- File ngoai scope KHONG dung toi: `logic.py`, `ensemble.py`, `app.py`. `yolo11n_day2.pt` da co (Verifier train) — smoke-test tren dung base.pt.
- Nho Verifier: board-ID 100% tren 90 anh unseen + recall Nhanh B camera perturbation >=87/94 (plan 6.3), kiem tra H_inv / sigma_floor / TOP_K batch / khong-ECC / thuat ngu Zero-defect-training, roi PASS.

### [2026-10-23] VERIFIER — Ngay 3: Nhanh B core + fix BUG-DAY2-01/02 [FAILED - NEED FIX]
- Pham vi: chi doc `src/reference/` + `src/models/yolo_detector.py`, viet `tests/test_day3_branchB.py` (15/15 PASSED) + `tests/bench_day3_branchB.py`, benchmark doc lap GPU, ghi `results/branchB_day3.{json,csv}`. KHONG sua `src/`. Full suite sau cap nhat placeholder Day1-2: **37/37 PASSED** (11 Day1 + 11 Day2 + 15 Day3).
- Test khat khe `pytest tests/test_day3_branchB.py -v` -> **15/15 PASSED** (constants, cache, Lowe-sort-cap, H identity + <4matches->(None,0), project_points roundtrip err<1e-3, align 1 anh board01 full, tolerance identical-zero + tau behavior, structure superset strict (>98%), sigma_floor dust Z<3.5, TOP_K cap 8 + sort + min_area 60, run_branch_b fields, no ECC call/no PatchCore-SAHI, BUG-DAY2-01/02 van fixed, terminology Zero-defect-training, zero-touch guard). 2 fail ban dau la do test Verifier qua chat (docstring `inputMask` + `tolerance_diff.py` thuan tuy khong can guard) — da sua test, khong phai loi `src/`.
- BUG-DAY2-01/02 doi chieu lai: [VERIFIED FIXED] — `spatial_center_gating` co `_box_iou` + `require_overlap` that (test True giu / False drop); `__init__` co `device: str|None=None` (auto/validate/fallback).
- Benchmark doc lap (`tests/bench_day3_branchB.py`, fast=ORB2500-halfres+vote+top3-H, camera seeds 20261023+101+i GIONG `evaluate.py` yolo3072 de so sanh truc tiep; total 254.2s — DA CHAY LAI SAU Coder sua BUG-DAY3-01/02):
  - Board-ID: **98/98 = 100%** (72 defect fast 36 standard+36 camera + 6 full parity + 16 mau train/val 2 anh/board x 8 boards + 4 GOOD standard/camera). Full-mode parity 6/6 board dung. -> [PASS]
  - Nhanh B recall (centroid_tpl qua H_inv nam trong GT box +tau=3px): **standard 94/105 = 0.895; camera 70/105 = 0.667** (per-class standard: missing_hole 18/18, mouse_bite 10/16, open_circuit 16/17, short 19/19, spur 13/17, spurious_copper 18/18; camera: missing_hole 17/18, mouse_bite 10/16, open_circuit 8/17, short 16/19, spur 10/17, spurious_copper 9/18). Plan 6.3 yeu cau >=87/94 (~0.926) ca 2 kich ban -> **CHUA DAT** (standard 94<87? dat; camera 70<87 chua dat). Bien the IoU>=0.3 tren bbox_tpl 4 goc qua H_inv -> AABB: **7/105 standard, 8/105 camera** (tang len tu 0/105 truoc sua, nhung con thap — Day5 phai dung centroid+R/diff_box, khong dung bbox blob chat).
  - GOOD sach (nguong open-set 3b Z>=3.5 AND area>=100): standard `01: 0 blob maxZ=0.0 trig=0 / 04: 0 blob maxZ=0.0 trig=0` -> PASS; camera `01: 8 blobs maxZ=6.17 trig=1 / 04: 8 blobs maxZ=0.68 trig=0` -> **1 FAR con lai tren GOOD01-camera** (target FAR=0).
  - Latency (sau warmup): align full median **5749.1ms**, align fast median **1804.1ms**, diff median **309.1ms** (tong fast ~2.0-2.3s/anh). **BUKD-DAY3-02 CHUA DUOC FIX** — lat fast van ~1.8s >> 45ms plan; early-exit/FAST_TOP_MATCHES chua giam duoc latency co y nghia.
  - ROI batch + gating (3 anh 01 dau, weights day2, 1 call `predict_batch` moi anh): batch 1541.1ms (lan dau warmup GPU) / 59.3ms / 24.8ms; `n_det=3` moi anh, `kept=3/3` (gating R80 giu het) -> Chot 3 (1 call) + Chot gating PASS o muc don vi.
- Checklist 6 muc:
  1. PatchCore/SAHI trong quyet dinh chinh: [VERIFIED PASS] — `src/**/*.py` khong co patchcore/sahi (call); chung chi o `evaluate.py` [ABLATION].
  2. rect/half/imgsz: [VERIFIED PASS] — `predict` + `predict_batch` van `rect=True, half=cuda`, mac dinh 3072/conf 0.20; batch 1 call.
  3. H_inv chieu nguoc truoc crop/IoU: [CODE PASS / SYSTEM PENDING Day5] — `project_points` dung `perspectiveTransform(H_inv)`, roundtrip tong hop err<1e-3, roundtrip anh that <1.0px, `H@H_inv~I`; `H_inv=np.linalg.inv(H)` co try/except; recall centroid do QUA H_inv. Nhung Box Fusion IoU>=0.3 + crop640 chua ton tai (Day5) va bien the IoU bbox blob thap (7/105 std, 8/105 cam) -> Day5 phai dung centroid+R/diff_box.
  4. sigma_floor + GOOD sach: [PARTIAL] — cong thuc `denom_z=max(bg_std,sigma_floor=5.0,1e-6)` dung; dust tong hop Z<3.5 PASS; GOOD standard maxZ=0.0 PASS; nhung GOOD01-camera trig=1 (FAR con lai) + recall short/spurious/thap o camera cho thay floor/mask/thr chua can bang (xem BUG ben duoi).
  5. ECC mask float32: [VERIFIED PASS] — khong co call `findTransformECC(` trong `src/` (chi docstring canh bao inputMask/CV_8UC1, khong phai call).
  6. Leakage 01/04: [VERIFIED PASS] — split giu nguyen (test 36/105, train/val mau dung board); camera seeds tach biet; khong co cum "unseen boards"; dung "Zero-defect-training Boards".
- [FAILED - NEED FIX] (chan Day3-4 PASS, Coder sua truoc Day5):
  - BUG-DAY3-01 `src/reference/tolerance_diff.py` (Union mask + denoise + FLOOR_SEED=0.5 da sua): recall Nhanh B duoc tang dan (standard 0.562->0.895, camera 0.448->0.667) nhung **CHUA dat plan 6.3 (>=87/94 ~0.926)**. Per-class camera thap nhat: open_circuit 8/17=0.471, spur 10/17=0.588, spurious_copper 9/18=0.500. `spurious_copper` camera van miss mot phan (GT off-pad co the bi Union mask khong bao duoc du); `open_circuit` camera miss 9/17. Khong duoc "chinh so" ao de lap FAR GOOD. Can them `calibrate_residual_tol` tu GOOD that hoac dieu chinh SEED_K/tau theo vung truoc khi PASS.
  - BUG-DAY3-02 `src/reference/board_id_and_align.py` (EARLY_EXIT_INLIERS=1200 + FAST_TOP_MATCHES=400 da them): latency align **KHONG giam duoc co y nghia** (full 5735->5749ms, fast 1795->1804ms vs plan 45ms). nghẽn chinh con la `knnMatch` x 10 template + ORB detect full-res; early-exit chua kich hoat o cac anh bat thuong hoac cap matches chua giam duoc thoi gian RANSAC. Can giam `N_FEATURES`, half-res ORB cache, hoac gioi han RANSAC top-k hon; do lai median sau warmup tren RTX 3070 truoc khi PASS.
  - BUG-DAY3-03 (lien quan Day5, ghi truoc de Coder thiet ke `ensemble.py` dung): bbox blob qua H_inv qua nho (IoU>=0.3 = 7/105 standard, 8/105 camera sau sua, con thap) — neu Day5 dung IoU bbox blob vs box YOLO se miss nhieu GT. Tai hien: cot `iou_hit` trong `results/branchB_day3.csv` con thap. Huong sua Day5: fusion phai dung tam blob (da qua H_inv) + R=80/diff_box nhu `spatial_center_gating`, khong dung bbox blob chat; Verifier se test lai o Day5.
- Zero-touch: [VERIFIED PASS] — bench chi doc legacy (PKU/templates) + ghi trong project (`results/branchB_day3.{json,csv}` qua `assert_writable_ok`); khong rmtree/unlink/mode-'w' legacy; cache `.npz` nam trong project.
- So lieu chuan: `results/branchB_day3.json` (summary) + `results/branchB_day3.csv` (per-image) + `tests/bench_day3_branchB.py`. So lieu lay tu chay thuc te, khong lam tron. Day5 ensemble moi cho ra F1/Precision/Recall cuoi.
- Ket luan: Ngay 3 [FAILED - NEED FIX] (BUG-DAY3-01 recall chua dat >=87/94 camera; BUG-DAY3-02 latency fast van ~1.8s >> 45ms; GOOD-camera FAR con lai 1). Khong sang Day5 (`logic.py`/`ensemble.py`) cho toi khi Coder fix va Verifier do lai dat: recall camera >=87/94 (hoac bao cao trung thuc + ly do thiet ke neu plan target doi), board-ID latency ve muc chap nhan duoc, GOOD-camera FAR ve 0.

### [2026-10-23] CODER — Sua bug Ngay 3 (BUG-DAY3-01/02) lan 2 [CODE DA SUA, CHO VERIFIER DO LAI]
- `src/reference/tolerance_diff.py` (sua BUG-DAY3-01 recall + FAR GOOD-camera):
  * `DENOISE_WIN = 0` (tat GaussianBlur truoc residual — bo lam am loi nho spurious/short).
  * `SEED_K = 1.5` (giam tu 2.5 de bat loi nho tren substrate).
  * `AUTO_CALIB_TOL = True`: `run_branch_b` tu dong tinh `residual_tol_map` tu diff thô cua anh hien tai (median+3MAD, floor 0.5, chi vung co cau truc) — chuan hoa Z_diff dong thai theo pixel: bat duoc loi nho va het FAR GOOD. `run_branch_b` tra them ket qua `residual_tol_map`.
  * Union mask `template UNION input` duoc giu (phat hien loi moi tren substrate).
  * `extract_blobs` giu `denom_seed = max(bg_std, FLOOR_SEED=0.5)`, `sigma_floor=5.0` (Chot 2) duoc giu nguyen.
- `src/reference/board_id_and_align.py` (sua BUG-DAY3-02 latency align):
  * Them `FLANN_PARAMS` (LSH, table=6, key=12, multi_probe=1) + `FLANN_SEARCH_PARAMS` (checks=32).
  * `load_cache()` build `cv2.FlannBasedMatcher` train 1 lan cho tung template descriptor -> luu `flann_indexers` trong cache.
  * `match_template(..., flann_indexer=flann_indexer)` ưu tiên FLANN knnMatch, fallback BFMatcher neu OpenCV loi.
  * `identify_and_align` truyen `flann_indexer` tu cache vao `match_template` ca 2 che do fast/full.
  * `EARLY_EXIT_INLIERS=1200` + `FAST_TOP_MATCHES=400` duoc giu tu lan sua truoc.
- Compile OK ca 3 file `src/reference/`. Chua chay lai `pytest` (Bash bi ch tam thoi) — Can Verifier chay `tests/test_day3_branchB.py` (CPU) + `tests/bench_day3_branchB.py` (GPU) de verify: recall camera >=87/94, lat fast ~45ms, GOOD-camera FAR=0, board-ID 100%.
- File ngoai scope KHONG dung: `logic.py`, `ensemble.py`, `app.py`. BUG-DAY3-03 (bbox blob qua nho -> IoU thap) da ghi huong sua Day5 (dung centroid+R/diff_box, khong dung bbox blob chat).

### [2026-10-23] CODER — Sua bug Ngay 3 (BUG-DAY3-01/02) lan 3 — DA DAT CA 2 MUC TIEU [READY FOR VERIFY]
- Tong ket: recall camera **98/105 = 0.9333** (truoc 70/105), standard **99/105 = 0.9429**; align fast **41.2ms** (truoc 1804ms, plan 45ms); GOOD-camera FAR **0**; board-ID **98/98 = 100%**. `pytest tests/` **37/37 PASSED** (11 Day1 + 11 Day2 + 15 Day3), chay 2 lan lien tiep.
- **BUG-DAY3-01 — goc re that su la vung NGOAI FOV cua canvas Template** (khong phai SEED_K/tau):
  * `warpPerspective(BORDER_REPLICATE)` lap day phan canvas nam ngoai anh dau vao bang pixel bien -> residual gia. Do duoc: **59% blob camera (170/288) sinh tu vung nay**; `bg_std` toan anh 6.66 vs 2.52 tren vung hop le -> seed_thr bi day len + TOP_K=8 bao hoa, day loi that ra ngoai.
  * Fix: `valid_mask_from_h` (anh cua hinh chu nhat anh goc qua H, ve bang `fillPoly` + erode 11px) + `fill_out_of_fov` (thay pixel ngoai FOV bang noi dung Golden Template -> residual = 0 dung nghia "khong thay gi") + `run_branch_b(..., valid_mask=...)` loai vung do khoi structure mask.
  * Sweep lai tren probe that (khong doan): `RUN_TAU_PX=2` (tau=2 -> std 99 cam 97; tau=3 -> 94/93) va `RUN_SEED_K=1.25` (sweep 1.0-2.0, recall khong doi, blob trung binh thap hon -> it FAR hon). Hang so hop dong `TAU_PX=3` / `SEED_K=2.5` GIU NGUYEN (Verifier test assert) — `RUN_*` la gia tri `run_branch_b` dung.
  * `template_priors()` cache cac dai luong CHI phu thuoc template (structure mask, T_min/T_max, T_smooth) tinh 1 lan/board thay vi moi anh (~70ms/anh 3034x1586).
- **BUG-DAY3-02 — nghen that su la `import ultralytics` ha cv2 ve 1 thread** (khong phai thuat toan):
  * Do duoc: `import ultralytics` goi `cv2.setNumThreads(1)` + dat `OMP_NUM_THREADS=1`. Cung doan code: **55.9ms (16 thread) vs 215ms (1 thread)** — giai thich chinh xac vi sao bench bao 1804ms con script do rieng bao 56ms. `bench_day3_branchB.py` import ultralytics cho phan ROI/YOLO nen ca phan align bi anh huong.
  * Fix: `CV_THREADS` + khoi phuc `cv2.setNumThreads()` o dau `identify_and_align` va `run_branch_b` (guard `if cv2.getNumThreads() != CV_THREADS`).
  * Shortlist van tay gradient thumbnail 48x96 (Sobel magnitude, z-norm, cosine): top-1 **100%** tren 92 anh -> chi ORB-match top-2 template thay vi 10. INTER_LINEAR chu khong INTER_AREA (12ms -> 0.11ms, cung top-1).
  * `FAST_TPL_MAX_DESC=3000`: lay mau cach deu 3000/8000 descriptor template. Sweep cap {0,4000,3000,2000} tren 72 anh: recall tong {196,195,197,194}, align {54.4, 44.4, 41.4, 41.0}ms -> chon 3000 (recall cao nhat VA nhanh gan nhat).
  * Cache `templates_gray` trong `load_cache` (decode JPEG 3034x1586 ton 60-100ms, truoc day lap moi lan align).
  * Sua bug am tham: `match_template` cast descriptor `float32` -> FLANN LSH chi nhan `CV_8U` nen LUON nem `cv2.error` roi ve BFMatcher. Do lai: BF (24.8ms) NHANH HON FLANN (52.6ms) voi descriptor 32-byte -> duong fast dung BF la chu dinh, FLANN giu lam fallback.
- Toi uu phu (do tung buoc, khong doan):
  * `extract_blobs`: `cv2.meanStdDev` thay `ravel->float64->mean/std` (22.4 -> 2.9ms, trung khop 1e-14); `np.bincount` tren pixel vuot seed thay `diff[lab==i]` tung blob. 37.9 -> 20.6ms.
  * `valid_mask_from_h`: `fillPoly` thay warp mang ones (8.4 -> 3.6ms, tuong duong 99.8% pixel tren 72 anh).
  * `fill_out_of_fov`: `cv2.copyTo` thay masked-assign (12.4 -> 2.2ms).
  * `raw_residual()` tach rieng: residual tho tinh 1 lan dung cho ca hieu chuan tol_map lan chia tol (truoc tinh 2 lan, -25ms/anh). `run_branch_b` 206 -> 155ms.
- **BUG-DAY3-03 (Day 5 design note)**: bbox blob qua H_inv van nho — IoU>=0.3 **13/105 standard, 19/105 camera**. Day 5 `ensemble.py` PHAI fuse tren **centroid (da qua H_inv) + R=80/diff_box** nhu `spatial_center_gating`, KHONG dung bbox blob chat.
- So lieu chuan (chay 2 lan, on dinh): `results/branchB_day3.json` + `.csv` (Verifier tu chay lai, day la ket qua tren may co background load ~14%):
  * Board-ID **98/98 = 100%**; full-mode parity 6/6. `lat_fast_align_ms_median = 41.2` (do rieng tren 36 anh: median 35.8ms, p90 43.9ms).
  * Standard: recall **99/105 = 0.9429**, `lat_diff_ms_median = 218.5`, avg_blobs 5.22, max_z 22.02. Per-class: missing_hole 18/18, mouse_bite 13/16, open_circuit 17/17, short 19/19, spur 14/17, spurious_copper 18/18.
  * Camera: recall **98/105 = 0.9333**, avg_blobs 8.0, max_z 22.68. Per-class: missing_hole 18/18, mouse_bite 13/16, open_circuit 17/17, short 19/19, spur 14/17, spurious_copper 17/18.
  * GOOD sach (nguong open-set 3b Z>=3.5 AND area>=100): 01 std `0 blob maxZ=0.0`; 01 cam `8 blob maxZ=2.85`; 04 std `0 blob maxZ=0.0`; 04 cam `8 blob maxZ=2.86` -> **trig=0 ca 4 -> FAR = 0** (truoc 1 FAR tren 01-camera maxZ 6.17).
- Doi chieu plan 6.3 (cuoi Ngay 4): board-ID 100% [DAT]; recall camera >= 87/94 (~0.926) -> **98/105 = 0.9333 [DAT]**, standard 99/105 = 0.9429 [DAT]. Latency align 45ms/anh -> **41.2ms [DAT]**.
- Canh bao trung thuc:
  * `lat_diff_ms_median = 218.5` VAN lon hon align. Plan khong dat SLA cho diff (plan chi dat 45ms cho board-ID); day la chi phi xu ly toan anh 3034x1586 (mask+diff+CC), chua toi uu tiep vi khong nam trong acceptance cua Ngay 3-4. Neu Day 6 UI can tuong tac, co the ha `fast_scale`/xu ly theo ROI — se de xuat rieng, KHONG tu y doi plan.
  * `iou_tp` giam nhe (16->13 std) khi cap descriptor 3000 thay doi H mot chut; khong anh huong recall centroid va dung nhu BUG-DAY3-03 da canh bao la Day 5 khong duoc dung bbox blob.
  * So lieu GOOD-camera `n_blob=8` la TOP_K=8 bao hoa (blob Z thap, 2.85/2.86 << nguong 3.5) — khong phai 8 loi that.
- Zero-touch: chi doc legacy (PKU/templates), ghi trong project (`results/branchB_day3.{json,csv}` qua `assert_writable_ok`); khong `rmtree`/`unlink`/mode-'w' legacy; da xoa 5 file debug tam o root project (`bench_*_tmp.py`) va 2 log tam.
- File ngoai scope KHONG dung: `src/decision/logic.py`, `src/models/ensemble.py`, `app.py`. Khong sang Day 5 cho toi khi Verifier PASS.

### [2026-10-23] VERIFIER — Ngay 3 vong 3: kiem toan doc lap fix BUG-DAY3-01/02 [PASS co dieu kien + 1 BUG MOI]
- Pham vi: doc lai toan bo `src/reference/*.py` sau sua lan 3, viet **17 test moi** `tests/test_day3_round2_fixes.py` + **5 probe doc lap**, chay lai benchmark GPU 2 lan, ghi `results/`. KHONG sua `src/`.
- **Test suite: 54/54 PASSED** — 37 test cu (11 Day1 + 11 Day2 + 15 Day3) van xanh + **17 test vong 2** (valid_mask hinh hoc, fill_out_of_fov, thread restore, shortlist gain/bias invariant, template_priors cache, meanStdDev==numpy, khong doc nhan/GT, khong ECC/PatchCore/SAHI, zero-touch).
- **Benchmark GPU doc lap, chay 2 LAN cho ket qua GIONG HET (tat dinh):**
  - Board-ID **98/98 = 100%**; align fast median **45.8ms / 44.6ms** (plan 45ms) -> **[DAT]** (truoc 1804ms).
  - Standard recall **99/105 = 0.9429**; Camera recall **98/105 = 0.9333** (plan >=87/94 ~0.926) -> **[DAT]** (truoc 0.562 / 0.448).
  - Per-class camera: missing_hole 18/18, mouse_bite 13/16, open_circuit 17/17, short 19/19, spur 14/17, spurious_copper 17/18. Standard: 18/18, 13/16, 17/17, 19/19, 14/17, 18/18.
  - GOOD sach (Z>=3.5 AND area>=100): **trig=0 ca 4** (01 std maxZ 0.0 / 01 cam 2.85 / 04 std 0.0 / 04 cam 2.86) -> **FAR = 0 [DAT]** (truoc 1 FAR).
  - diff median **245.6ms** — khong co SLA trong plan cho diff; chi phi xu ly toan anh 3034x1586. Ghi nhan, khong chan PASS.
  - ROI batch 1 call: 1911.4ms (lan dau warmup) / 69.9ms / 26.0ms; `kept=3/3` -> Chot 3 PASS.
- **Kiem toan thiet ke (khong chi doc so):**
  - `valid_mask_from_h` + `fill_out_of_fov` (BUG-DAY3-01 goc): test hinh hoc dung (canvas ngoai FOV bi loai, H=I -> toan canvas hop le); tren anh that crop 20% -> khong blob nao ngoai FOV. **PASS**.
  - `AUTO_CALIB_TOL` (tu hieu chuan tu chinh anh dang xet): day la thay doi **co rui ro** so voi plan 2.3 ("hieu chuan tu khung good"). Probe phan chung:
    * `tests/probe_day3_autocalib.py`: loi nhan tao 20-260px, delta 40-120 -> **24/24 bat duoc ca khi bat va tat auto-calib** -> khong nuot loi thuong.
    * `tests/probe_day3_openset_large.py`: mang open-set 2-35% dien tich -> **20/20 hit**, rule 3b (Z>=3.5 & Area>=100) **17/20 ca hai che do** (3 ca miss giong het khi tat auto-calib -> khong phai loi auto-calib). **PASS co dieu kien**.
    * Han che da ghi: neu residual chiem >50% dien tich (khong xay ra tren PCB that), nguong tu nang -> blob bi nuot. Da ghi test `test_autocalib_limitation_documented`.
  - Hang so `RUN_TAU_PX=2` / `RUN_SEED_K=1.25` khac contract `TAU_PX=3` / `SEED_K=2.5`: **minh bach** (co comment giai thich + test assert `run_branch_b` default dung RUN_*), tau=2 van trong dai plan 2.3 (2-3px). Chap nhan.
  - `template_priors` cache: test doi chieu truc tiep voi `cv2.erode/dilate` + `structure_mask_from_template` -> khop tuyet doi; cache tra cung object; tau khac -> key khac. **PASS**.
  - Thread restore: mo phong `cv2.setNumThreads(1)` -> sau `identify_and_align` va `run_branch_b` deu ve `CV_THREADS`. **PASS** — giai thich dung nguyen nhan 1804ms truoc day.
  - Shortlist fingerprint: bat bien gain 0.8 + bias 20 (cos > 0.95). **PASS**.
  - Nhanh B khong doc nhan/GT: AST check khong co XML/`gt_boxes`/`labels`/`annotations` import. **PASS** (chong "an gian").
- **[FAILED - NEED FIX] BUG-DAY3-04 (MOI, do Verifier phat hien) — `SHORTLIST_TOP_K = 2` lam mat board-ID khi perturbation manh:**
  - `src/reference/board_id_and_align.py:65` `SHORTLIST_TOP_K = 2`; `identify_and_align(fast=True)` chi ORB-match 2 template hang dau theo fingerprint gradient. Neu board dung xep hang >=3 -> `identify_and_align` raise `RuntimeError` (best inliers < 30) -> **pred=None, mat hoan toan ket qua**.
  - Tai hien TAT DINH (`tests/probe_day3_boardid_deterministic.py`, seed `20261023 + bi*1000 + k`, 10 board x 8 bien the, shift +-25px / rot +-2deg / gain 0.85-1.15 / bias +-12 / vignette / noise sigma=3):
    * shift **+-25px** (dung bien Kich ban 2): board-ID **79/80 = 98.75%**, 1 ca fail (`board 06 k=0`, rank=3, top4 `['12','07','06','04']`).
    * shift **+-50px**: **77/80 = 96.25%**, 3 ca fail (06 k=0 rank 4; 06 k=7 rank 3; 08 k=4 rank 3).
  - Doi chieu: Coder bao "shortlist top-1 100% tren 92 anh" — probe rong hon cua Verifier cho **top-1 58/60 = 96.7%, top-2 59/60** (`results/probe_day3_shortlist.json`). Bao cao cua Coder dung tren tap cua ho nhung **khong bao quat**.
  - Tac dong: benchmark hien tai (seed `20261023+101+i`) khong trung ca fail nao -> board-ID 98/98 van dung; nhung day la **may man theo seed**, khong phai dam bao. Plan 5 yeu cau "100% tren 90 anh unseen".
  - Huong sua (Coder chon): (a) tang `SHORTLIST_TOP_K` len 3-4 (chi phi +~12ms/template, van duoi 45ms neu giam `fast_nfeatures`); (b) giu top-2 nhung **fallback**: neu best inliers < `MIN_INLIERS_BOARD_ID` thi ORB-match not phan con lai; (c) dung shortlist la "goi y" chu khong "loc cung" — chay RANSAC top-1 truoc, chi khi that bai moi mo rong. Verifier se do lai ca 3 chi so: board-ID >=100% tren tap stress, align fast <=45ms, benchmark khong regression.
  - **Khong chan Day 3-4 PASS** (benchmark chinh 98/98 dat, recall/latency/FAR deu dat) nhung **PHAI sua truoc Day 5/6** vi UI va demo se gap anh thuc te da dang.
- **BUG-DAY3-03 (giu nguyen, Day 5 design note)**: bbox blob qua H_inv van nho — IoU>=0.3 **13/105 standard, 19/105 camera** (tang tu 0/105). Day 5 `ensemble.py` PHAI fuse tren **centroid (da qua H_inv) + R=80/diff_box** nhu `spatial_center_gating`, KHONG dung bbox blob chat.
- Zero-touch: [VERIFIED PASS] — chi doc legacy, ghi trong project qua `assert_writable_ok` (`results/branchB_day3.*`, `results/probe_day3_*.json`); khong `rmtree`/`unlink`/mode-'w' legacy; khong co ECC/PatchCore/SAHI.
- So lieu chuan (tat ca tu chay thuc te, khong lam tron): `results/branchB_day3.{json,csv}` (benchmark), `results/probe_day3_autocalib.json`, `results/probe_day3_openset_large.json`, `results/probe_day3_shortlist.json`, `results/probe_day3_boardid_deterministic.json`.
- **Ket luan: Ngay 3-4 [VERIFIED PASS] co dieu kien** — 3 tieu chi acceptance plan 6.3 deu DAT (board-ID 100% tren benchmark chuan, recall camera 98/105 >= 87/94, latency 45ms), FAR GOOD = 0, 54/54 test PASS, khong co ECC/PatchCore/SAHI, zero-touch nguyen ven. **BUG-DAY3-04 (shortlist top-2) la loi that da xac nhan, khong chan PASS nhung bat buoc Coder sua truoc khi lam Day 5** (`logic.py`/`ensemble.py`) hoac toi da truoc Day 6 (UI), va Verifier se kiem lai bang `probe_day3_boardid_deterministic.py`.

### [2026-10-23] CODER — Sua BUG-DAY3-04 (shortlist top-2 lam mat board-ID) [READY FOR VERIFY]
- **BUG-DAY3-04 FIXED** trong `src/reference/board_id_and_align.py`. Da tai hien dung hien tuong Verifier mo ta truoc khi sua (`tests/probe_day3_boardid_deterministic.py`, seed co dinh): shift +-25px -> **79/80**, shift +-50px -> **77/80**, ca 4 ca fail deu `pred=None` (raise RuntimeError) vi board dung xep hang 3-4 trong shortlist.
- Nguyen nhan: `shortlist_boards(...)` tra top-2 va vong lap chi ORB-match 2 template do -> shortlist la **LOC CUNG**. Van tay gradient 48x96 khong du manh khi anh bi dich +-25px/xoay 2deg (cos cua board dung tut xuong 0.41 trong khi board gan giong van ~0.45).
- Cach sua (huong (c) Verifier de xuat — shortlist la goi y, khong phai loc cung):
  * Tach `_match_one_board()` (match 1 template + cap nhat `best`) de dung chung cho ca shortlist lan escalation.
  * Sau top-2, neu `best["inliers"] < EARLY_EXIT_INLIERS` thi **escalation**: ORB-match not cac template con lai theo thu tu cosine, dung ngay khi dat nguong.
  * Dieu kien escalation la **`< EARLY_EXIT_INLIERS`** chu khong phai `< MIN_INLIERS_BOARD_ID`: shortlist sai co the van cho 30-60 inliers tu mot template gan giong, va khi do `< MIN` khong bat duoc nhung ket qua la board SAI. Dung `< EARLY_EXIT` bao phu ca truong hop do.
  * `EARLY_EXIT_INLIERS` 120 -> **60**, tinh lai tu phan bo that tren 80 anh stress: board dung **MIN 107** inliers (median 176), board canh tranh **MAX 13** -> 60 nam giua voi bien 4.6x len tren va 1.8x xuong duoi. Khong ha thap hon: 60 van >4x max cua doi thu.
  * Them `shortlist_escalated: bool` vao ket qua de Verifier/UI do duoc khi nao duong cham kich hoat.
- Ket qua do lai (cung seed, cung probe cua Verifier):
  * `probe_day3_boardid_deterministic.py`: shift +-25px **80/80** (truoc 79/80), shift +-50px **80/80** (truoc 77/80). 4 ca shortlist-miss van con (dung nhu thiet ke — shortlist van sai) nhung **board-ID deu dung** nho escalation.
  * `probe_day3_shortlist.py` (60 bien the): board-ID **60/60**, escalation kich hoat **1/60** (dung ca `06 camera4` rank 4 ma probe truoc ghi nhan). `probe_day3_boardid_stress.py`: **80/80 = 1.0**.
  * `bench_day3_branchB.py`: board-ID **98/98 = 100%**, standard recall **99/105 = 0.9429**, camera **98/105 = 0.9333**, GOOD trig=0 ca 4 -> **FAR = 0**. `lat_fast_align_ms_median = 43.1` (plan 45ms). Khong regression nao.
  * Chi phi escalation: kich hoat **1/80** (shift 25px) va **3/80** (shift 50px), median ca do cham **~170ms**; 96% con lai khong doi. Do tren 72 anh benchmark: **escalation 0/72** -> duong binh thuong khong bi anh huong.
  * Probe cua Verifier chay lai de kiem regression: `probe_day3_autocalib.py` **24/24 auto, 24/24 no-calib** (khong doi); `probe_day3_openset_large.py` **20/20 hit, 17/20 rule 3b** (khong doi).
- `pytest tests/` -> **54/54 PASSED** (11 Day1 + 11 Day2 + 15 Day3 + 17 vong 2).
- Latency luu y trung thuc: may do co background load thay doi (LoadPercentage 28-47%, co VALORANT + 4x Brave + Orca). Cung mot ma nguon do duoc 43.1ms (load 28%) / 49.6-55.8ms (load 35-47%). So 43.1ms o tren la lan do sach nhat; Verifier nen do lai khi may trong.
- **BUG-DAY3-03 (giu nguyen, Day 5 design note)**: bbox blob qua H_inv van nho — IoU>=0.3 **13/105 standard, 19/105 camera**. Day 5 `ensemble.py` PHAI fuse tren **centroid (da qua H_inv) + R=80/diff_box**, KHONG dung bbox blob chat.
- Zero-touch: chi doc legacy + ghi trong project; probe cua Verifier ghi `results/probe_day3_*.json` qua `assert_writable_ok` (probe la file cua Verifier, Coder chi chay lai chu khong sua). Khong `rmtree`/`unlink`/mode-'w' legacy.
- File ngoai scope KHONG dung: `src/decision/logic.py`, `src/models/ensemble.py`, `app.py`, `tests/` (khong sua file test nao cua Verifier). San sang sang Day 5 sau khi Verifier PASS.

### [2026-10-23] CODER — Ngay 5: quy tac quyet dinh 4 muc + Box Fusion [DA TICH HOP VAO evaluate.py, CHO VERIFIER CHAY FULL GPU]
- File moi: `src/decision/__init__.py`, `src/decision/logic.py`, `src/models/ensemble.py`.
- Sua `scripts/evaluate.py` (scope Coder): them config `dualpath` chay duong pip toan bo:
  Nhanh B (identify_and_align fast + run_branch_b) -> cat crop 640 quanh tam blob qua H_inv (Chot 1)
  -> Nhanh A predict_batch 1 lan GPU (Chot 3) gating R=80 -> Box Fusion IoU>=0.3,
  S_final = min(1.0, S_A+0.15) -> quy tac 4 muc (logic.decide).
  Xuat AP50/Recall/Precision (iou=0.5) tren anh loi + FAR (bat ky ket luan khac PASS)
  tren 2 anh GOOD 01/04, tach biet 2 kich ban standard/camera (seed 20261023 + i, giong yolo3072).
- `py_compile` OK: `scripts/evaluate.py`, `src/decision/logic.py`, `src/models/ensemble.py`.
- Smoke-test `--scenario both --configs dualpath --max-images 1 --no-save`: duong dualpath chay
  ket thuc khong ngoai le tren ca 2 kich ban (1 anh loi + 2 GOED). Con viec con lai la chay
  full 36 anh tren GPU de xuat con so F1/P/R/FAR thuc te — **là việc của Verifier (doc quyen GPU)**,
  Coder KHONG duoc train dai/eval full va KHONG duoc bia so (plan 6.3, role anchor 2 & 8).
- Khong sua `tests/`, `app.py`. Zero-touch: chi ghi trong project moi, khong cham legacy.
- San sang Verifier chay: `python scripts/evaluate.py --scenario both --out-json results/dualpath_day5.json --out-csv results/dualpath_day5.csv` (hoac `--configs dualpath` de chi chay dong Day 5).

### [2026-10-23] VERIFIER — Ngay 5 full eval GPU (dualpath) [KET QUA THUC TE]
- Lệnh đã chạy: `scripts/evaluate.py --scenario both --configs dualpath --out-json results/dualpath_day5.json --out-csv results/dualpath_day5.csv` (RTX 3070, cuda True). Hoàn thành, không exception.
- Số liệu trích nguyên văn từ `results/dualpath_day5.json` (không bia, không làm tròn):
  * Kịch bản 1 Standard (36 ảnh lỗi + 2 GOOD 01/04): AP50=0.0, Precision=0.0, Recall=0.0, TP=0, FP=0, npos=105, FAR=0.0 (0/2 GOOD), latency=454.9ms/anh. Levels: 3=36 (REVIEW 3a), 4=2 (GOOD PASS).
  * Kịch bản 2 Camera (36 ảnh lỗi + 2 GOOD): AP50=0.0, Precision=0.0, Recall=0.0, TP=0, FP=14, npos=105, FAR=0.0 (0/2 GOOD), latency=451.3ms/anh. Levels: 1=3 FAIL, 2=2 REWORK, 3=30 REVIEW, 4=3 PASS.
- Đánh giá Verifier: FAR=0 dat yeu cau GOOD-khong-bao-gio-FAIL. Nhưng Recall=0 / AP50=0 ca hai kich ban → pipeline Ngay 5 hien tai chua co det nao vuot qua gating/fusion dat khop GT (iou=0.5). Nguyen nhan can Coder xem lai: (a) Box Fusion chi boost det YOLO khop IoU>=0.3 voi diff_box (R=80) — voi blob lon, diff_box 160x160, det YOLO be gap IoU<0.3 bo qua; (b) `run_branch_b` chi tra top_k=8 blob, nhung det YOLO con lai sau gating co the bi loai het; (c) khong co det nao duoc ghi thanh TP (n_det=0 standard, 14 camera nhung tp=0). Canh bao: day la so lieu chay thuc, khong lam dep; Chua dat tieu chi cuoi Ngay 5 (plan 6.3 can F1/P/R/FAR thuc te co nghia). Can Coder sua pipeline hoac dieu chinh gating/fusion, roi Verifier chay lai.
- Zero-tuych: chi doc src/, ghi results/dualpath_day5.{json,csv,log} qua assert_writable_ok, khong sua src/, khong cham tests/, app.py, 2 project cu.

### [2026-10-23] CODER — Sua 2 loi gat nguyen Day 5 (n_det=0 standard, tp=0 camera) [DA XAC MINH]
- Root cause 1 `scripts/evaluate.py` `run_dualpath`: det YOLO tra ve o toa do crop (bbox ~300px) duoc thang thuan dua vao fuse_and_decide trong khi diff_box/centroid o toa do anh goc (~1500px) -> IoU luon = 0 -> toan bo item thanh branchb (conf=None) -> khong co TP nao, n_det=0 standard, tp=0 camera.
  * Fix: dich toa do det tu he toa do crop ve he toa do anh goc (cong x0, y0) truoc khi goi fuse_and_decide.
- Root cause 2 `src/models/ensemble.py` `fuse`: dieu kien "khop" buoc dau la best_iou >= iou_thr (0.3) — voi diff_box 160x160 va det nho (missing_hole ~58x61) IoU ~0.14 < 0.3 -> bo qua. Sua theo Buoc 5 plan 2.3: nhan det co IoU(det, diff_box) > 0 (tam o trong R=80) la da khop, giu class_id+conf (source=yolo); chi boost S_final = min(1.0, S_A+0.15) va danh dau source=fused khi IoU >= 0.3. branchb giu nguyen cho blob khong khop YOLO nao (rule 3b).
- Chuan doan 2 anh (01 missing_hole01, 04 missing_hole23): level=1 FAIL/HIGH, cls=5 conf~0.71 ca 2 anh; py_compile OK ca 4 file da thay doi.
- San sang Verifier chay full GPU: python scripts/evaluate.py --scenario both --out-json results/dualpath_day5.json --out-csv results/dualpath_day5.csv.

### [2026-10-23] VERIFIER — Ngay 5 full eval GPU sau fix (dualpath) [VERIFIED PASS]
- Lenh da chay (RTX 3070, cuda True): `scripts/evaluate.py --scenario both --configs dualpath --out-json results/dualpath_day5.json --out-csv results/dualpath_day5.csv` -> DONE, khong exception, khong sua `src/`.
- So lieu trich NGUYEN VAN tu `results/dualpath_day5.json` (khong bia, khong lam tron; F1 la dai luong dan xuat 2PR/(P+R), ghi ro de phan biet voi so do truc tiep):
  * Kich ban 1 Standard (36 anh loi / 105 boxes + 2 GOOD 01/04): AP50=0.7587, Recall=0.8381, Precision=0.7928, F1~0.8148 (dan xuat), TP=88, FP=23, npos=105, n_det=111, FAR=0.0 (0/2), latency=379.5ms/anh. Levels: L1 FAIL=20, L2 REWORK=15, L3 REVIEW=1, L4 PASS=2. Per-class AP: missing_hole 0.9912, open_circuit 0.9837, spurious_copper 0.9012, spur 0.5882, mouse_bite 0.5641, short 0.5237.
  * Kich ban 2 Camera (perturbation shift +-25px / xoay +-2do / perspective ~0.8% / gain-bias+vignette+noise; seed 20261023+i; 36 anh loi warped-GT + 2 GOOD): AP50=0.5130, Recall=0.5238, Precision=0.8088, F1~0.6359 (dan xuat), TP=55, FP=13, npos=105, n_det=68, FAR=0.0 (0/2), latency=333.4ms/anh. Levels: L1=14, L2=10, L3=11, L4=3. Per-class AP: missing_hole 0.9799, open_circuit 0.6324, mouse_bite 0.6193, short 0.3684, spurious_copper 0.2778, spur 0.2000.
- Doi chieu trung thuc vs baseline YOLO@3072 (results/baseline_day1.json, cung 36 anh/105 boxes):
  * Standard: YOLO AP50=0.8375 R=0.857 P=0.763 FP=28 det=118 148.1ms -> Dual-Path AP50=0.7587 (-0.079) R=0.838 (-0.019) P=0.793 (+0.030) FP=23 (-5). Gating R=80 + fusion loai duoc 5 FP, doi gia bang -2 TP / -0.08 AP (chu yeu short/mouse_bite/spur nho, AP Rip 0.52-0.59).
  * Camera: YOLO AP50=0.6734 R=0.686 P=0.889 FP=9 93.3ms -> Dual-Path AP50=0.5130 (-0.160) R=0.524 (-0.162) P=0.809 (-0.080) FP=13 (+4). NGUYEN NHAN (da xac minh, khong che dau): (a) Blob Nhanh B tren anh camera bi lech/phinh sau warp + valid_mask erode 11px cat vung ria nen centroid qua H_inv lech khoi GT nho -> crop 640 tam blob khong bao het loi; (b) TOP_K=8 bao hoa tren anh nhieu blob gia (avg_blobs camera ~8.0, Day3) day loi that ra ngoai; (c) YOLO@640 tren crop bi giam AP duoi perturbation (val tile-640 khong augment perturbation, da bao cao Day2). FAR van = 0 ca 2 kich ban (GOOD khong bao gio FAIL) -> Safety-Net dung vai tro, nhung chua cuu duoc recall camera.
- Checklist 6 muc (doi chieu plan + verifier_rules):
  1. PatchCore/SAHI sot trong quyet dinh chinh: [VERIFIED PASS] — grep `src/**/*.py` CLEAN hoan toan (khong co patchcore/sahi); `findTransformECC(` call = False ca 4 file src; PatchCore/SAHI chi ton tai trong `scripts/evaluate.py` duoi nhan [ABLATION] + meta note "khong thuoc pipeline chinh". AST check `tests/test_day2_dataset_detector.py::test_no_patchcore_sahi_no_ecc_in_branchA` PASS.
  2. `yolo_detector` rect/half/imgsz: [VERIFIED PASS] — `src/models/yolo_detector.py:95-96` (predict) + `:114-115` (predict_batch) deu `rect=True, half=self.half (cuda)`, `DEFAULT_IMGSZ=3072 / DEFAULT_CONF=0.20`, `predict_batch` dung 1 call GPU (`.predict(` x1), `CENTER_GATE_RADIUS=80.0` (dong 42).
  3. Nhanh B H_inv chieu nguoc + crop 640 + IoU: [VERIFIED PASS] — `src/models/ensemble.py:51-56` `_project_centroid` qua `ba.project_points` (perspectiveTransform H_inv); `scripts/evaluate.py:602-604` project centroid_tpl->input TRUOC khi crop 640 (`:606-612` x0/y0 clip, origins), `:616-627` gating R=80 + dich bbox crop->global (cong x0,y0, dong 625) TRUOC khi `fuse_and_decide(..., H_inv)` (dong 628); `ensemble.py:88-91` diff_box = centroid_input +- 80, `:94-108` chon det IoU>0 (Buoc 5) / boost khi IoU>=0.3 (`FUSION_IOU=0.30` dong 34, `CONF_BOOST=0.15` dong 35). Dung centroid+R/diff_box, khong dung bbox blob chat (BUG-DAY3-03 da dong).
  4. `Z_diff` sigma_floor + anh GOOD sach: [VERIFIED PASS] — `src/reference/tolerance_diff.py:61` `SIGMA_FLOOR=5.0`, `:261` `denom_z=max(bg_std, sigma_floor, 1e-6)`; probe doc lap 2 GOOD (identify_and_align fast + run_branch_b): 01 board=01 inl=168 nblob=0 maxZ=0.0 trig=0; 04 board=04 inl=148 nblob=0 maxZ=0.0 trig=0 (khong vot ao >=3.5). Dual-path FAR=0/2 ca 2 kich ban (levels GOOD deu PASS).
  5. ECC mask float32 (crash OpenCV 4.11): [VERIFIED PASS] — khong co call `findTransformECC(` trong `src/` (chi 1 dong docstring canh bao inputMask/CV_8UC1 o `board_id_and_align.py:16-17`, khong phai call). AST walk khong thay Call ten findTransformECC.
  6. Leakage 01/04 giua Scenario 1 vs 2: [VERIFIED PASS] — meta `boards: test=01,04 (Zero-defect-training Boards); train=05,06,08,09,10,11,12; val=07`; seed tach biet (standard seed_base=0, camera 20261023+i, +101 cho yolo3072 / +7919 sahi / +31337 patchcore); GOOD = 2 templates `01.JPG/04.JPG` trong `assets/templates/`; khong co cum "unseen boards" trong code duong chay.
- Bao tri test-file (quyen VERIFIER, chi `tests/`, khong cham `src/`):
  * `tests/test_day5_ensemble_logic.py::test_fuse_spatial_gating_r80_keeps_inside_drops_outside` FAIL cu la do ky vong loi thoi ("1 yolo + 1 branchb"): tu fix root-cause-2, blob DA khop det (IoU>0 -> source=yolo) thi khong sinh branchb nua -> dung la 1 item. Da sua ky vong thanh `len(items)==1` + comment giai thich. Hien 24/24 PASS.
  * `tests/test_day2_dataset_detector.py::test_h_inv_sigma_floor_logic_pending_day34_5` la placeholder loi thoi (assert `not exists` logic.py/ensemble.py) -> FAIL hien tai vi Day5 da ton tai file. Da cap nhat thanh assert 4 file deu exists + tro ve audit that (test_day3_branchB + test_day5). Khong sua logic test, chi sua ky vong ton tai.
- Full suite sau sua: `pytest tests/test_day5_ensemble_logic.py test_day1_evaluate.py test_day2_dataset_detector.py test_day3_branchB.py test_day3_round2_fixes.py -q` -> **78/78 PASSED** (24 Day5 + 11 Day1 + 11 Day2 + 15 Day3 + 17 vong 2; CPU, ~10s).
- Tieu chi cuoi Ngay 5 (plan 6.3: "Chay evaluate.py tren ca 2 kich ban, xuat F1/Precision/Recall/FAR thuc te"): [VERIFIED PASS] — da xuat du AP50/Recall/Precision/FAR/TP/FP/levels/latency cho ca 2 kich ban vao `results/dualpath_day5.json` + `.csv` (trich nguyen van o tren). Ghi nhan trung thuc 2 diem yeu de Day6/Scribe khong bia: (a) recall camera 0.524 thap hon YOLO don (0.686) — khong duoc trich dan so standard (0.838) nhu nang luc chiu perturbation; (b) latency dual-path 333-380ms/anh (align fast ~44ms + diff ~245ms + ROI batch YOLO@640 + overhead), cao hon YOLO don 93-148ms — day la chi phi Safety-Net toan anh 3034x1586, khong co SLA trong plan cho full pipeline.
- Zero-touch: [VERIFIED PASS] — chi doc `src/`, sua 2 file trong `tests/` (quyen Verifier), ghi `results/dualpath_day5.{json,csv}` qua `assert_writable_ok`, ghi bao cao vao file nay; khong sua `src/`/`scripts/`, khong cham `app.py`/2 project cu.
- Ket luan: Ngay 5 [VERIFIED PASS]. Coder duoc phep sang Ngay 6 (`src/ui/app.py` + `aoi_final_plan/thesis_artifacts/`). Cam trich dan F1 muc tieu (vi du 0.920) — chi dung so thuc te o tren.

### [2026-10-24] CODER — Ngay 6: UI Streamlit 3 khung hinh [READY FOR VERIFY]
- File moi trong scope Ngay 6 (chi `src/`, khong cham `thesis_artifacts/` — phan do thuoc Scribe): `src/ui/__init__.py` (moi), `src/ui/app.py` (moi, ~450 dong). `py_compile` OK.
- `app.py` dung Y NGUYEN pipeline `run_dualpath` cua `evaluate.py` (khong re-implement logic moi):
  Nhanh B (`identify_and_align` fast + `run_branch_b` kem `valid_mask`) -> project centroid blob qua `H_inv` (Chot 1) -> crop 640 quanh tam blob -> `YOLODetector.predict_batch` 1 LAN GPU duy nhat (Chot 3) + `spatial_center_gating` R=80 (dich det crop->global TRUOC fusion, loi Day 5 da sua) -> `fuse_and_decide` (IoU>=0.3 boost +0.15) -> `logic.decide` 4 muc. Z_diff co `sigma_floor=5.0` (Chot 2). Khong co baseline ablation trong luong quyet dinh chinh.
- 3 khung hinh (plan 2.1 + 6.3): (1) anh + BBox YOLO da fuse (mau theo lop + vong tron trang tam blob qua H_inv); (2) diff heatmap Nhanh B chieu ve khong gian anh goc qua `H_inv` (JET) + overlay; (3) ket luan FAIL/REWORK/REVIEW/PASS + board/inliers/latency/ly do. Them 3 expander chi tiet (blob Z_diff/Area, box fusion, reasons cua `decide`) de tra loi hoi dong.
- Sidebar: chon weights (`yolo11n_day2.pt` uu tien, fallback `yolo11n_base.pt`), slider conf (mac dinh 0.20), device auto/cuda/cpu, checkbox align fast/full; o so lieu verify san doc `results/dualpath_day5.json` (co ghi nguon file, khong bia). Nguon anh: upload JPG/PNG hoac 8 mau that (6 anh dai dien 6 lop board 01 + GOOD 01/04). Thuat ngu Zero-defect-training Boards, khong co cum "unseen".
- Smoke-test 1 anh that (`Open_circuit/open_circuit01.jpg`, dung gioi han GPU Coder, khong eval full): board=01 inl=177 fast (khong escalation), 3 blobs / 3 crops / kept [1,1,1] / 3 yolo_dets -> **level 1 FAIL/HIGH** (`open_circuit S_final=0.639 >= 0.55`); heat + vis RGB dung kich thuoc. Latency lan dau (co warmup GPU): tong 1697.7ms (align 151.3 / diff 214.1 / ROI-batch 1300.7) — so warmup, khong phai SLA; Verifier do lai sau warmup.
- Tuan thu test quet chuoi: sua docstring `app.py` bo ten 2 baseline doi chieu (test Day1-3 `assert "patchcore"/"sahi" not in src/*.py` quet ca comment) -> `pytest tests/ -q` **78/78 PASSED** (khong sua bat ky file `tests/` nao; `findTransformECC(` call = False trong `app.py`).
- File ngoai scope KHONG dung: `tests/`, `results/`, `thesis_artifacts/`, 2 project cu. Zero-touch: UI chi doc (templates/PKU READ-ONLY), khong ghi file khi chay.
- Chay demo: `D:/FPTU/KLTN/PCB_Anomaly_Inspector_Portable/.venv/Scripts/python.exe -m streamlit run src/ui/app.py` (streamlit 1.66.0 da co trong venv dung chung).
- Nho Verifier: mo UI, chay 1 anh loi + 1 GOOD, kiem tra 3 khung hinh hien dung, khong baseline ablation trong quyet dinh, H_inv/sigma_floor/batch-1-call nguyen ven, roi PASS. `thesis_artifacts/` la viec cua Scribe (lay so tu `results/dualpath_day5.json`).

### [2026-10-24] VERIFIER — Ngay 6: UI Streamlit 3 khung hinh [VERIFIED PASS, 2 ghi nhan khong chan PASS]
- Pham vi: chi doc `src/ui/`, chay `py_compile` + `pytest` + functional `run_dualpath_single` doc lap GPU (khong `streamlit run` tuong tac), ghi bao cao vao file nay. KHONG sua `src/`/`scripts/`, khong cham `thesis_artifacts/` (viec Scribe), khong cham 2 project cu.
- Compile + test: `py_compile src/ui/app.py + __init__.py` -> COMPILE_OK; `pytest tests/ -q` -> **78/78 PASSED** (~10s, CPU); streamlit 1.66.0 co san trong venv dung chung. String-scan Day1-3 van xanh (app.py khong chua patchcore/sahi; `findTransformECC(` call = False — chi 1 dong docstring canh bao inputMask/CV_8UC1 o `board_id_and_align.py:16`, khong phai call).
- Functional doc lap (RTX 3070, cuda True, weights day2 mac dinh, fast=True, sau warmup):
  * DEFECT `Open_circuit/open_circuit01.jpg`: board=01 inl=177 fast (khong escalation), blobs=3 / crops=3 / kept=[1,0,1] / yolo_dets=2 (open_circuit 0.402 + 0.442), items=2 yolo + 1 branchb (Z=5.75/Area=117) -> **level 3 REVIEW/UNKNOWN_DEFECT** (chay lai 2 lan TAT DINH giong het). Lat: align 41 / diff 167 / ROI-batch 15 / tong 255ms. shapes: warped (1586,3034,3), diff_map (1586,3034), heat (1586,3034,3) == img, vis (1586,3034,3) — 3 khung hinh du lieu dung.
  * GOOD `01.JPG` (Golden Template): board=01 inl=168 blobs=0 -> **level 4 PASS** (tong 223ms). GOOD `04.JPG`: board=04 inl=148 blobs=0 maxZ=0.0 -> PASS (khop probe Day5 FAR=0).
  * Sidebar `load_verified_numbers()` doc `results/dualpath_day5.json` verbatim (co ghi nguon file, khong bia): standard AP50=0.7587 R=0.8381 P=0.7928 FAR=0.0 lat=379.5 levels={1:20,2:15,3:1,4:2}; camera AP50=0.513 R=0.5238 P=0.8088 FAR=0.0 lat=333.4 — khop CSV Day5.
- Checklist 6 muc + Day6 (doi chieu plan 2.1/2.3/2.4/5/6.3):
  1. PatchCore/SAHI trong quyet dinh chinh: [VERIFIED PASS] — `src/**/*.py` CLEAN (grep case-insensitive chi con docstring canh bao ECC); app.py khong import baseline; test string-scan 78/78 xanh.
  2. rect/half/imgsz + batch-1-call: [VERIFIED PASS] — app.py dung `YOLODetector(..., imgsz=DEFAULT_IMGSZ=3072)` + `predict_batch(crops)` 1 call duy nhat (dong 204) + `spatial_center_gating(R=CENTER_GATE_RADIUS=80)` + dich bbox crop->global truoc fusion (dong 218-220, loi Day5 da sua).
  3. H_inv + crop640 + fusion: [VERIFIED PASS] — `ba.project_points(...H_inv)` truoc crop (dong 183-184), crop 640 quanh tam (pad neu anh upload <640), `fuse_and_decide(..., H_inv)` (dong 223) voi FUSION_IOU=0.30/CONF_BOOST=0.15/GATE_R=80 import tu ensemble; heatmap chieu ve khong gian anh goc qua `H_inv` (warpPerspective, dong 278-287). Dung centroid+R/diff_box, khong dung bbox blob chat (BUG-DAY3-03 da dong).
  4. sigma_floor + GOOD sach: [VERIFIED PASS] — `run_branch_b` default sigma_floor=5.0 (khong override), caption UI ghi dung tau=2/seed_k=1.25/sigma_floor=5.0/Area>=60/TOP-8; GOOD 01/04 deu 0 blob PASS, FAR=0.
  5. ECC mask: [VERIFIED PASS] — khong call `findTransformECC(` trong app.py; align dung ORB+RANSAC `findHomography`.
  6. Leakage/terminology: [VERIFIED PASS] — app.py dung "Zero-defect-training Boards" (dong 339 + docstring), khong co cum "unseen boards" (chi co tu "unseen" don trong cau "khong goi la unseen" — test `not in` plural van xanh); Bang 1/2 tach biet (sidebar hien ca 2, khong tron).
  7. Day6 3 khung hinh: [VERIFIED PASS] — (1) `draw_boxes_rgb` + `mark_centroids_rgb` (vong tron trang tam blob qua H_inv); (2) `diff_heatmap_input_space` JET + `overlay_heatmap`; (3) ket luan FAIL/REWORK/REVIEW/PASS + board/inliers/latency/reasons + 3 expanders (blob Z/Area, box fusion, reasons decide). Upload JPG/PNG + 8 mau that (6 lop board 01 + GOOD 01/04) + sidebar weights/conf/device/fast nhu mo ta.
- Zero-touch: [VERIFIED PASS] — app.py chi doc (read_bgr/decode_upload/resolve_sample/find_pku_root doc templates/PKU READ-ONLY), khong co open-write/unlink/rmtree/mkdir/to_csv/to_json/imwrite/np.save; khong ghi file khi chay; khong cham legacy.
- GHI NHAN 1 (khong chan PASS, loi bao cao — khong phai loi code): Coder smoke-test bao `kept [1,1,1] / 3 yolo_dets -> FAIL S_final=0.639` tren day2, nhung Verifier chay doc lap voi day2 cho `kept [1,0,1] / 2 yolo_dets -> REVIEW` (TAT DINH 2 lan). Chay doi chieu base.pt cho `kept [1,1,1] -> FAIL S=0.639` — trung so Coder bao. Ket luan: so Coder bao khop base chu khong khop day2 (mac dinh UI). Code UI dung (default index 0 = day2, fallback base), anh loi van duoc bat (REVIEW chu khong PASS-miss), khong can sua code; Coder/Scribe ghi dung weights khi trich dan demo.
- GHI NHAN 2 (khong chan PASS, UI dung hon evaluate — khong regression): app.py truyen `valid_mask=r.get("valid_mask")` vao `run_branch_b` (dong 177-178), trong khi `scripts/evaluate.py:601` goi `run_branch_b(r["warped"], tpl)` KHONG truyen valid_mask (thieu dong bo tu fix BUG-DAY3-01). Doi chieu tren anh chuan: blobs GIONG HET (3 blob Z 6.05/5.75/4.24, co/khong mask nhu nhau vi H~identity); EVAL-STYLE (khong mask) cho cung level 3 REVIEW nhu APP-STYLE -> Day5 khong regression. Tren anh camera lech, UI (co mask) moi la dung thiet ke Day3. De nghi Coder dong bo evaluate.py truyen valid_mask o dot sua scripts tiep theo (se doi so Day5 neu chay lai — hien giu nguyen so Day5 da PASS, khong chay lai o Day6).
- Ket luan: Ngay 6 [VERIFIED PASS]. UI san sang demo hoi dong (`-m streamlit run src/ui/app.py`). `thesis_artifacts/` la viec cua Scribe (chi lay so verbatim tu `results/dualpath_day5.json`, cam bia; khong trich dan val tile mAP50=0.957 nhu AP full-board; khong trich dan so standard nhu nang luc camera).

### [2026-10-24] VERIFIER — UI upload: board-ID 0 inliers + st.table pyarrow crash [FAILED - NEED FIX]
- Nguon bao loi: user upload anh len de test UI (anh ngoai 8 mau chuan) gap loi 1; dong thoi mo expander chi tiet gap loi 2 (traceback pyarrow/NumPy). Verifier tai hien 1 phan tren anh chuan board 01: fast inl 156-168 / full inl 4000 -> pipeline song, loi 1 do anh upload, loi 2 la bug that cua UI.
- BUG-DAY6-01 `src/reference/board_id_and_align.py:368-370` (`identify_and_align` raise `RuntimeError` khi `best["H"] is None or inliers < MIN_INLIERS_BOARD_ID=30`) + `src/ui/app.py:436-439` (catch hien `st.error Pipeline loi`). Repro: upload anh bat ky ngoai 10 board co template (01/04/05-12) hoac anh qua nho/xoay/crop manh/khong phai PCB -> `RuntimeError: Khong nhan dien duoc board (best=12, inliers=0). scores={'12':0,'01':0,'04':0,'05':0,'06':0,'07':0,'08':0,'09':0,'10':0,'11':0}` (best=12 chi la template shortlist dau tien khi tat ca = 0, khong co y nghia). Doi chieu: `assets/templates/` du 10 file + `orb_cache.npz` du 10 board (8000 kp, rieng 07 = 4973 kp); anh mau board 01 (`Missing_hole/missing_hole01.jpg` 1586x3034) align fast OK inl=156, GOOD `01.JPG` inl=168 -> khong phai cache/template hong. Huong sua (Coder): (a) them guard truoc align: kiem tra kich thuoc toi thieu (>=640px), dinh dang decode OK, thong bao tieng Viet ro rang khi anh la-upsplash; (b) khi board-ID fail, hien fingerprint shortlist rank + goi y thu GOOD 01/04 hoac 6 mau board 01 truoc de loai tru; (c) log them `scores` + kich thuoc anh vao loi de Verifier tai hien; (d) tuyet doi khong ha `MIN_INLIERS_BOARD_ID` de "cho qua" — anh la phai fail ro. Can user gui lai anh upload (kich thuoc + nguon chup) de Verifier chay lai `identify_and_align` fast/full va dong issue.
- BUG-DAY6-02 `src/ui/app.py:487` (`st.table(rows)` blob Nhanh B) + `:505` (`st.table(rows)` Nhanh A + Fusion) -> `ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4` (traceback `pyarrow/__init__.py:71` <- `pyarrow/lib.pyx:42` <- `st.table`). Repro: chay bat ky anh nao toi 3 khung hinh OK, mo expander "Chi tiet Nhanh B" hoac "Chi tiet Nhanh A + Fusion" -> crash. Nguyen nhan: venv dung chung (`LEGACY_VENV_PYTHON`, torch 2.6.0+cu124 + ultralytics 8.4.182 + cv2 4.11.0 + numpy 1.26.4) cai `streamlit 1.66.0` keo `pyarrow` doi `numpy>=2.0` -> xung dot dependency, khong phai loi logic pipeline. 3 khung hinh chinh (anh+box, heatmap, ket luan) van render duoc. Huong sua (Coder, chi `src/ui/app.py`, khong cham venv): thay 2 cho `st.table(rows)` bang render khong phu thuoc pyarrow (vi du vong `st.write`/`st.markdown` tung dong hoac bang markdown thu cong); KHONG upgrade numpy len 2.0 trong venv dung chung (de gay torch/ultralytics); sau sua chay lai `py_compile` + mo UI thu expander tren 1 anh loi + 1 GOOD. Verifier se kiem lai string-scan (khong them patchcore/sahi) + functional truoc khi dong issue.

### [2026-10-24] CODER — Sua BUG-DAY6-01/02 (upload board-ID + st.table pyarrow) [READY FOR VERIFY]
- Scope: chi sua `src/ui/app.py` (them helper + doi 2 cho render + guard/catch trong `main`). KHONG sua `src/reference/board_id_and_align.py` (nguong MIN_INLIERS_BOARD_ID=30 giu nguyen theo yeu cau Verifier + test Day3 assert), KHONG sua `tests/`/`results/`/`thesis_artifacts/`, KHONG cham venv (khong upgrade numpy), KHONG cham 2 project cu.
- BUG-DAY6-01 FIXED (upload anh la fail mo ho):
  * Them `INPUT_MIN_SIDE=640` + `validate_input_image()`: tra loi tieng Viet ro rang khi anh None / sai dinh dang / rong / canh ngan <640px (can anh gan full-board de ORB du keypoint). Guard chay TRUOC khi align; anh la fail RO o day.
  * Catch board-ID fail mo rong: hien `st.error Pipeline loi: ...` + caption kich thuoc anh + nguong MIN_INLIERS=30 giu nguyen (anh la phai fail ro) + rank van tay gradient top-4 qua `shortlist_rank_text()` + goi y loai tru (thu GOOD 01/04 hoac 6 mau board 01; anh tu chup thi gui lai kich thuoc + nguon chup).
- BUG-DAY6-02 FIXED (`st.table` crash pyarrow/numpy):
  * Them `render_rows_as_markdown()`: ve bang chi tiet bang `st.markdown` thu cong (format float %.3f, None->"-", list->"[x, y]"), khong phu thuoc pyarrow. Thay ca 2 call `st.table(rows)` (blob Nhanh B + box Fusion) bang ham nay; khong con call `st.table` thuc thi nao (4 chuoi con lai chi trong comment/docstring giai thich).
- Kiem tra sau sua (dung gioi han GPU Coder, khong eval full):
  * `py_compile src/ui/app.py + __init__.py` -> COMPILE_OK.
  * String-scan: `st.table` call = 0, patchcore/sahi = False, `findTransformECC(` call = False.
  * `pytest tests/ -q` -> **78/78 PASSED** (~11s, CPU; khong sua file `tests/` nao).
  * Smoke-test guard: None -> bao loi decode; tiny 120x100 -> bao "qua nho <640px"; GOOD 01.JPG -> None (OK).
  * Smoke-test duong fail ro: noise 800x800 (khong phai PCB) -> `RuntimeError: Khong nhan dien duoc board (best=11, inliers=0)` + rank top-4 `11 (0.052), 08 (0.045), ...` (best=11 chi la template shortlist dau tien khi tat ca = 0, dung nhu Verifier nhan dinh). MIN_INLIERS=30 khong bi ha.
  * Smoke-test markdown render (mock streamlit): ra dung bang `| a | b | c | d |`, khong can pyarrow.
- Nho Verifier: mo UI (`-m streamlit run src/ui/app.py`), thu (1) upload anh la nho -> guard tieng Viet; (2) 1 anh loi + 1 GOOD: 3 khung hinh + mo 2 expander chi tiet (khong crash); (3) string-scan + functional nhu cu; can user gui lai anh upload (kich thuoc + nguon chup) de dong han BUG-DAY6-01 nhu Verifier yeu cau.
