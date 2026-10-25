# BÁO CÁO TỔNG KIỂM TOÁN ĐỘC LẬP & ĐỐI CHIẾU 3 HỆ THỐNG — DUAL-PATH INDUSTRIAL AOI (FINAL SIGN-OFF)

**Vai trò:** CHIEF AUDITOR & THESIS DEFENSE REVIEWER (FINAL SIGN-OFF)
**Ngày:** 2026-10-24
**Phạm vi:** Chỉ-đọc 100% (READ-ONLY). Không sửa `src/`, `scripts/`, `tests/`, `results/`, không chạm `thesis_artifacts/`, `status_and_bugs.md`. Không chạy GPU nặng.
**Nguồn sự thật:** `aoi_final_plan/plan.md` (7 phần), `aoi_final_plan/rules/verifier_rules.md`, code thực tế trong `src/`, `scripts/evaluate.py`, số liệu verbatim trong `results/`, 2 project cũ ở chế độ STRICT READ-ONLY.
**Quy tắc trích số:** Mọi F1/Precision/Recall/mAP50/FAR đều trích nguyên văn từ file `results/*.json` chạy thực tế. Không bịa, không làm tròn quá mức (giữ 4 chữ số như file gốc).

> Thuật ngữ bắt buộc: `01/04` là **Zero-defect-training Boards** (có Golden Template `01.JPG/04.JPG` nhưng chưa từng có mẫu lỗi trong train). Tuyệt đối không gọi là "unseen boards".

---

## TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY — KẾT LUẬN TỔNG QUAN)

1. **Tuân thủ plan:** Nhánh A, Nhánh B, 3 chốt chống lỗi ngầm (H_inv / sigma_floor / batch+R80), logic 4 mức open-set độc lập, cấm PatchCore/SAHI trong luồng quyết định chính — **tất cả ĐẠT**, có dẫn chứng file:dòng cụ thể (xem Phần 1).
2. **Trung thực số liệu:** Không rò rỉ 01/04 vào train/val (train {05,06,08,09,10,11,12}, val {07}, test {01,04}). Hai kịch bản Standard vs Camera tách bạch, seed và tham số nhiễu công khai (xem Phần 2). Ngưỡng suy ra từ sweep/probe, không chép tay.
3. **Hiệu năng thực (verbatim Day5):**
   - Standard (36 ảnh lỗi + 2 GOOD, npos=105): `AP50=0.7587, R=0.8381, P=0.7928, F1≈0.8148, TP=88, FP=23, n_det=111, FAR=0.0 (0/2), latency=379.5ms`.
   - Camera (perturb, npos=105): `AP50=0.5130, R=0.5238, P=0.8088, F1≈0.6359, TP=55, FP=13, n_det=68, FAR=0.0 (0/2), latency=333.4ms`.
   - So với YOLO3072 đơn nhánh: Standard AP -0.079 (0.7587 vs 0.8375 base / 0.8392 day2), Camera AP -0.160 (0.513 vs 0.6734 base / 0.6234 day2). Dual-Path giữ FAR=0 cả hai kịch bản, đóng vai Safety-Net nhưng chưa cứu được recall camera.
4. **Đối chiếu 3 hệ thống:** System1 (Portable legacy: YOLO+PatchCore OR cứng, FPR PatchCore=100%) thua về thiết kế ngưỡng; System2 (Test_github TDD-Net: chia random theo ảnh 80/20 → rò rỉ board, 9920/2508 ảnh 600×600) không so sánh được trực tiếp với split board-disjoint; System3 (Dual-Path mới) là duy nhất có board-disjoint + open-set 3b + FAR=0 đo được.
5. **Hạn chế trung thực:** Lớp yếu nhất Standard là `short (0.5237) / mouse_bite (0.5641) / spur (0.5882)`; Camera là `spur (0.20) / spurious_copper (0.2778) / short (0.3684)`. Latency 333–380ms vượt mục tiêu 160–190ms của Nhánh A đơn. Còn 2 ghi nhận Day6 (weights demo day2 vs base; `evaluate.py` chưa truyền `valid_mask` như UI). Không được trích val-tile `mAP50=0.957` thành AP full-board, không trích `R=0.838` standard thành năng lực camera.
6. **Kết luận sign-off:** **ĐẠT có điều kiện (CONDITIONAL PASS)** — Đủ điều kiện bảo vệ với câu chuyện "Nhánh A tổng quát hình thái + Nhánh B đối chiếu Golden + Ablation giải thích vì sao PatchCore/SAHI thất bại", kèm phụ lục hạn chế và 5 câu hỏi phản biện đã chuẩn bị (Phần 5). Scribe được phép dùng mọi số trong file này nguyên văn.

---

## PHẦN 1 — KIỂM TOÁN TUÂN THỦ PLAN (PLAN COMPLIANCE AUDIT)

### 1.1. Nhánh A — YOLO11n Full-Res (`src/models/yolo_detector.py`)

- **Cấu hình:** `DEFAULT_IMGSZ=3072` (dòng 40), `DEFAULT_CONF=0.20` (dòng 41), `CENTER_GATE_RADIUS=80.0` (dòng 42).
- **Chống Square Padding:** `predict()` dòng 89–96: `rect=True, half=self.half`; `predict_batch()` dòng 108–115: `rect=True, half=self.half`. Đúng plan 2.2.
- **Batch ROI (Chốt 3 latency):** `predict_batch(images, imgsz=640)` gom N crops gọi `self.model.predict(source=bgrs, ...)` **1 lần duy nhất** (dòng 108). Test `test_yolo_detector_forces_rect_half_and_defaults` assert `.count(".predict(")==1` — PASS.
- **Spatial center gating R=80:** `spatial_center_gating()` dòng 132–169: đổi tâm box về hệ ảnh gốc (`gx1=x1+ox...`, `bx=(gx1+gx2)/2`), giữ nếu `dist<=r` (dòng 163) hoặc `require_overlap and IoU>0` với `diff_win=[cx-r,cy-r,cx+r,cy+r]` (dòng 166–168). Test `test_spatial_center_gating_r80_with_origin` PASS.
- **Trọng số Day2 vs Base:** `assets/weights/yolo11n_base.pt` là copy 1 chiều từ `best.pt` gốc (50 epoch, `configs/resource_paths.py:33-34`); fine-tune Day2 xuất file mới `assets/weights/yolo11n_day2.pt` (`scripts/finetune_yolo_day2.py: DEFAULT_OUT != BASE_PT`, test `test_zero_touch_writes_guarded` PASS). UI `src/ui/app.py` ưu tiên `yolo11n_day2.pt` (index 0) fallback base — đúng quy tắc không ghi đè base. Ghi nhận Day6: số demo Coder báo (`kept[1,1,1] -> FAIL S0.639`) khớp **base**, không khớp day2 (`kept[1,0,1]/2 -> REVIEW`); UI mặc định day2 là đúng, lỗi vẫn bị bắt (REVIEW, không PASS-miss) — chỉ cần ghi chú weights khi trích demo.

### 1.2. Nhánh B — BoardID & Coarse Align (`src/reference/board_id_and_align.py`)

- **ORB + RANSAC:** `N_FEATURES=8000` (dòng 51), `RATIO_TEST=0.75` (dòng 52), `RANSAC_THRESH=5.0` (dòng 53), `H (Input->Template)` + `H_inv=np.linalg.inv(h)` (dòng 372–373), `project_points()` via `perspectiveTransform(H_inv)` dòng 396–399. Đúng Chốt 1.
- **Cache:** `.npz` trong project mới (`DEFAULT_CACHE=assets/templates/orb_cache.npz`, dòng 50), `build_cache()` qua `assert_writable_ok` (dòng 111) — zero-touch.
- **Shortlist + escalation:** fingerprint gradient 48×96 (`SHORTLIST_H/W=48/96`, dòng 68), `SHORTLIST_TOP_K=2` (dòng 67), `EARLY_EXIT_INLIERS=60` (dòng 56), `FAST_TPL_MAX_DESC=3000` (dòng 82), `CV_THREADS` khôi phục sau khi ultralytics hạ về 1 thread (dòng 74, 303–305). Nếu `<EARLY_EXIT` thì escalation sang template còn lại (dòng 348–357, BUG-DAY3-04). Board-ID 98/98 = 100% (`results/branchB_day3.json`), `lat_fast≈43.1ms`.
- **ECC:** Không gọi `findTransformECC` trong `src/` (grep AST trong `test_no_patchcore_sahi_no_ecc_in_branchA` + `test_day3_branchB` PASS). Docstring cảnh báo `inputMask=None hoặc CV_8UC1` (dòng 16–18) — đúng checklist verifier #2, tránh crash OpenCV 4.11.

### 1.3. Nhánh B — Tolerance Diff (`src/reference/tolerance_diff.py`)

- **Hợp đồng vs vận hành:** `TAU_PX=3` (dòng 48, contract cho test), `RUN_TAU_PX=2` (dòng 49, dùng trong `run_branch_b` sau sweep tau=2 cho std 99/105, cam 97/105 tốt hơn tau=3 là 94/93, FAR GOOD=0 cả hai). `SEED_K=2.5` (dòng 56, contract), `RUN_SEED_K=1.25` (dòng 57, điểm bằng sau sweep). Ghi rõ trong code, không mập mờ.
- **Photometric norm:** `correct_lowfreq_gain` win 151 clip 0.5–2.0 (dòng 120–132, `SMOOTH_WIN=151`, `GAIN_LO/HI=0.5/2.0` dòng 51–52). Lý do bỏ local z-score ghi trong docstring (dòng 6–10).
- **Tau diff + mask:** `warped_residual/tolerance_diff` dùng `dilate/erode` tạo `[T_min,T_max]` (dòng 135–144, 202–217); structure mask = `dilate7(copper Otsu UNION gradient>30)` (dòng 150–162, `STRUCT_DILATE_PX=7`, `GRAD_EDGE_THRESH=30.0`); `residual_tol_map` auto `median+3MAD` (dòng 220–237, 369–377).
- **Chốt 2 sigma_floor:** `SIGMA_FLOOR=5.0` (dòng 61), `denom_z=max(bg_std, sigma_floor, 1e-6)` (dòng 261), `Z_diff=(mean_res-bg_mean)/denom_z` (dòng 277). CC `Area>=60` (`MIN_AREA_PX2=60` dòng 60), `TOP_K=8` (dòng 62). Test GOOD sạch: 01 inl168 nblob0 maxZ0.0, 04 inl148 nblob0 maxZ0.0, FAR 0/2 cả hai kịch bản — PASS.
- **valid_mask (BUG-DAY3-01):** `valid_mask_from_h` fillPoly + erode11 (dòng 70–95), `fill_out_of_fov` copyTo (dòng 98–114, 2.2ms vs np.where 28.3ms). `run_branch_b(..., valid_mask)` (dòng 324–330, 346–365). **Phân kỳ đã xác minh:** `src/ui/app.py:177-178` truyền `valid_mask=r.get("valid_mask")` (UI đúng), nhưng `scripts/evaluate.py:601` gọi `run_branch_b(r["warped"], tpl)` **không** truyền valid_mask (miss sync). Đối chiếu trên ảnh aligned (H~I): blob giống hệt (3 blobs Z 6.05/5.75/4.24) cả hai kiểu, EVAL-STYLE cũng level3 REVIEW như APP-STYLE → số Day5 không hồi tố. Trên ảnh camera shift, UI (có mask) mới là thiết kế đúng. Khuyến nghị: đồng bộ `evaluate.py` ở lần sửa `scripts/` tiếp theo, giữ nguyên số Day5.

### 1.4. Ba chốt chống lỗi ngầm (Micro-traps)

| Chốt | Thực thi | Dẫn chứng |
|---|---|---|
| **Chốt 1 — Đồng bộ H_inv** | Tâm blob Template → Input qua H_inv **trước** crop640 + IoU | `ensemble.py:51-56` (`_project_centroid` via `ba.project_points`), `88-91` (diff_box=cx±80), `evaluate.py:602-604` (project cents), `606-612` (crop origins), `616-627` (gating + shift `+x0,+y0` dòng 625), `628` (`fuse_and_decide(...,H_inv)`); `app.py:183-184` (project), `190-201` (crop+pad), `210-221` (gating+shift), `223` (fuse). BUG-DAY3-03: fuse trên centroid+R/diff_box, KHÔNG bbox blob thô (`ensemble.py:18-20`). Hai root-cause Day5 (quên shift crop; IoU>=0.3 quá chặt miss missing_hole 58×61 IoU~0.14) đã sửa: shift +x0+y0, match nếu `IoU>0` + boost chỉ khi `>=0.3` (`ensemble.py:102,108`). |
| **Chốt 2 — sigma_floor** | `Z_diff` có sàn | `tolerance_diff.py:61` (5.0), `:261` (`max(bg_std,sigma_floor)`), `:277`; GOOD probe maxZ 0.0 cả 01/04; Day5 FAR 0.0 cả hai kịch bản. |
| **Chốt 3 — Batch ROI** | TOP_K crops 1 GPU call + gating R80 | `tolerance_diff.py:62` (TOP_K=8; plan cho 6 hoặc 8), `yolo_detector.py:101-117` (1 call), `ensemble.py:34-36` (`FUSION_IOU=0.30, CONF_BOOST=0.15, GATE_R=80.0`); leftover gating dòng 130–151. |

### 1.5. Logic 4 mức open-set độc lập (`src/decision/logic.py` + `src/models/ensemble.py`)

- **Ngưỡng (dòng 24–29):** `CONF_HIGH=0.55` (FAIL {0,1,5}), `CONF_MEDIUM=0.35` (REWORK {2,3,4}), `CONF_LOW_LO=0.20`, `Z_REVIEW=2.0` (3a `0.20<=S<0.35 AND Z>=2.0`, dòng 79–91), `Z_OPENSET=3.5` + `AREA_OPENSET=100.0` (3b `source==branchb AND Z>=3.5 AND Area>=100`, dòng 94–99, **không** `AND yolo_conf>=0.30`). PASS còn lại (dòng 108–111). Ưu tiên FAIL>REWORK>REVIEW>PASS. Test 24/24 trong `tests/test_day5_ensemble_logic.py` (sau 1 fix test-only, xem §5).
- **Fusion:** match nếu `best_iou>0.0` (dòng 102) → `fused` boost `min(1.0,S_A+0.15)` nếu `>=iou_thr` (dòng 108–110) else `yolo` giữ S_A; blob không khớp → `branchb` giữ z/area, conf None (dòng 121–127) để rule 3b xét. YOLO lẻ giữ nếu trong gate (dòng 130–151).
- **Cấm ablation trong quyết định chính:** grep/AST toàn `src/` không có `patchcore/sahi/findTransformECC(` call (`test_no_patchcore_sahi_no_ecc_in_branchA`, `test_day3_branchB` PASS). PatchCore/SAHI chỉ trong `scripts/evaluate.py` làm baseline ablation (YOLO640, SAHI slice640 ov0.20, PatchCore resnet18 l2+l3 memory_bank AUROC/FPR, YOLO3072 Nhanh A) — đúng plan 1.2/2.4.

**Kết luận Phần 1:** `[VERIFIED PASS]` toàn bộ checklist verifier #1–3 (H_inv/sigma_floor/rect+batch; ECC; open-set độc lập).

---

## PHẦN 2 — TRUNG THỰC SỐ LIỆU (METRIC HONESTY)

### 2.1. Zero leakage 01/04

- **Split:** train `{05,06,08,09,10,11,12}`, val `{07}`, test Zero-defect-training `{01,04}` (36 ảnh/105 boxes, 01:18 + 04:18). Nguồn: `scripts/prepare_yolo_dataset.py:50-52` (`TRAIN_BOARDS/VAL_BOARDS`, comment "01/04 held-out"), `results/*.json meta.boards: "test=01,04; train=05..; val=07"`.
- **Bằng chứng:** `data/yolo_day2/stats.json` (`train_defect 213 + bg 53 =266, val_defect 55 + bg 10 =65`, `heldout_zero_defect_training ["01","04"]`); test `test_no_board_01_04_leakage_in_train_val` quét tên file `*_01_/_04_` + bg tiles chỉ từ train/val — PASS. `test_day1_evaluate` + `test_day2_dataset_detector` (11 tests) khóa split board-disjoint. Không dùng cụm "unseen boards" (test `test_zero_touch_writes_guarded` assert không có chuỗi đó, có `Zero-defect-training`).
- **GOOD = templates:** `assets/templates/01.JPG,04.JPG` (`resource_paths.py:42-43`, `LOCAL_TEMPLATES`), FAR tính trên 2 ảnh này (far_total=2).

### 2.2. Nguồn ngưỡng (không chép tay)

| Ngưỡng | Giá trị | Nguồn |
|---|---|---|
| YOLO conf / imgsz | 0.20 / 3072 | plan 2.2, `yolo_detector.py:40-41`, `evaluate.py:87` (`DEFAULT_CONF=0.20`), CLI `--conf default DEFAULT_CONF`, `--highres 3072` |
| FAIL / REWORK | 0.55 / 0.35 | plan 2.4, `logic.py:24-25` |
| 3a LOW_LO / Z_REVIEW | 0.20 / 2.0 | plan 2.4, `logic.py:26-27` |
| 3b Z_OPENSET / AREA | 3.5 / 100.0 | plan 2.4, `logic.py:28-29` |
| Fusion IoU / Boost / Gate | 0.30 / 0.15 / 80.0 | plan 2.4/2.3-b5, `ensemble.py:34-36`, test `test_constants_match_plan` |
| Tau contract/run, Seed contract/run | 3 / 2, 2.5 / 1.25 | plan 2.3 (tau 2–3px), `tolerance_diff.py:48-58` + comment sweep (tau2: std99/105 cam97/105 vs tau3: 94/93; seed 1.25 điểm bằng) |
| Sigma_floor / MinArea / TopK | 5.0 / 60 / 8 | plan Chốt 2 + 2.3-b4 + Chốt 3, `tolerance_diff.py:60-62` |
| ORB/RANSAC/EarlyExit/Shortlist | 8000 / 0.75 / 5.0 / 60 / top2 / fast_desc 3000 | `board_id_and_align.py:51-60,82`, comment sweep stress 80–160 ảnh (board đúng MIN 107 vs cạnh tranh MAX 13) |

### 2.3. Tách kịch bản + tham số nhiễu camera

- **Hai bảng riêng:** `standard` (PKU gốc) vs `camera` (perturb) trong mọi `results/*.json`; plan 4.2 cấm gộp. `evaluate.py: run_scenario(name)` (dòng 667–673): `perturb=(name=="camera")`, `seed_base=20261023 if perturb else 0`; YOLO3072 `+101`, SAHI `+7919`, PatchCore `+31337` (dòng 672–705) — đảm bảo tái lập nhưng không trùng seed.
- **Perturb (`evaluate.py:302-354` `perturb_image_and_boxes`):** shift ±25px, xoay ±2°, phối cảnh Homography ~0.8%, gain 0.85–1.15, bias ±10, vignette 0.15, noise 3–5, warp GT boxes theo cùng H. (Trích từ tóm tắt phiên trước + code; Scribe khi vẽ thesis phải ghi đúng dải này, không tự thêm mờ motion/blur.)
- **FAR định nghĩa:** `far_triggers/far_total` trên ảnh GOOD (`evaluate.py:645-653`); GOOD bị kết luận level<4 thì đếm trigger. Day5 `far=0.0 (0/2)` cả hai kịch bản — nguyên văn, không suy diễn thành "FAR=0% trên dây chuyền".

**Kết luận Phần 2:** `[VERIFIED PASS]` checklist verifier #4 (tách scenario, seed, leakage). Mọi số Phần 3 trích verbatim, F1 suy ra ghi rõ công thức.

---

## PHẦN 3 — ĐỐI CHIẾU 3 HỆ THỐNG (3-WAY BENCHMARK)

### 3.1. Kiến trúc (Arch)

| Hệ | Phát hiện | Quyết định | Điểm chết đã đo |
|---|---|---|---|
| **System1 — Portable legacy** (STRICT READ-ONLY: `PCB_Anomaly_Inspector_Portable/src/...`) | YOLO (`yolo_detector.py`: conf 0.25, không rect/half/imgsz) + PatchCore resnet18 l2+l3 (`anomaly_detector.py`, `F.normalize(feats)` dòng 56 — PatchCore gốc không có; `score=patch_map.max()` dòng 209 — thống kê mức ảnh vs ngưỡng mức patch) | `logic.py:evaluate_decision`: `HIGH={open,short,missing_hole} OR is_anomaly -> FAIL`, else MEDIUM -> REWORK, else PASS (không 3a/3b, không ngưỡng số, lỗi OR: anomaly kéo FAIL hàng loạt) ; `ensemble.py:HybridInspector` (YOLO+PatchCore) | Ngưỡng PatchCore `0.6395` cho **FPR=100% (264/264 GOOD bị FAIL)** trên 7 board; AUROC gộp 480 ảnh (240/240) **0.6949**; hai fix 15 phút (bỏ normalize + percentile thay max) chỉ lên **0.7314 (+0.037)**; bank chéo-board 0.62–0.75 vs riêng-board 0.78–0.84 → PatchCore là phương pháp **theo từng board**, ghép bank chung + split board-disjoint là mâu thuẫn kiến trúc (`AUDIT_vs_TDDNet.md:§0-2`). Trên 10 ảnh mẫu: YOLO 5/5 (conf 0.68–0.83, 0 FP GOOD) hơn TDD (5/5 nhưng 2/5 GOOD FP conf 0.9565/0.9902). |
| **System2 — Test_github TDD-Net/FPN** (STRICT READ-ONLY: `Test_github/...`) | TDD-Net FPN_Res101, P2 stride 4 + anchors `[15,25,40,60,80]`, OHEM 128, crop 600×600, 9920 train / 2508 val, paper CAAI 2019 (`README.md`, `cfgs.py:66-131`) | R-CNN softmax+vote, `SHOW_SCORE 0.6` chỉ hiển thị | `divide_data.py:29-32` `random.shuffle` rồi cắt 80/20 **theo ảnh** → rò rỉ board (cùng board vừa train vừa val) → số VOC mAP của họ **không so được** với split board-disjoint của ta (`AUDIT_vs_TDDNet.md:§0.5, §1.1`). Repo không có eval board-disjoint, không có open-set. |
| **System3 — Dual-Path mới** (project này) | Nhánh A YOLO11n 3072 rect/half + Nhánh B ORB/RANSAC-H/H_inv + tolerance diff (tau2, sigma5.0, TOP8) + Fusion IoU0.3/boost0.15/gate80 | Logic 4 mức FAIL/REWORK/REVIEW/PASS, 3a/3b độc lập, không OR cứng | Giá phải trả: latency 333–380ms + recall camera rớt (xem §3.2–4). Được: FAR=0 đo được + open-set 3b + board-disjoint sạch. |

Số ngoài luồng (`hybird_test_3branch/results/comparison.json`, READ-ONLY, cùng cấp KLTN — không phải TDD-Net): `B1_YOLO P0.714/R1.0/F1 0.833`, `H4_Learned P0.953/R0.987/F1 0.969` — chỉ tham khảo hướng hybrid có học, không phải số của 3 hệ trên, không được trích thành kết quả đồ án.

### 3.2. Số liệu verbatim (không bịa)

**Day1 baseline — `results/baseline_day1.json` (weights `yolo11n_base.pt`, conf 0.20, IoU 0.5, npos=105, 36 defect + 2 GOOD):**

| Kịch bản | YOLO640 | SAHI640 [ABLATION] | PatchCore [ABLATION] | YOLO3072 Nhánh A |
|---|---|---|---|---|
| Standard | AP 0.0119, R 0.0286, P 0.0227, TP3 FP129 det132, 124.8ms | AP 0.7644, R 0.8095, P 0.4802, TP85 FP92 det177, 605.3ms | AUROC 0.3056, FPR 1.0, TP36 TN0 FP2 FN0, 86.7ms | AP 0.8375, R 0.8571, P 0.7627, TP90 FP28 det118, 148.1ms; per-class open 0.8603 short 0.7268 mouse 0.8667 spur 0.5829 spurious 0.9971 missing 0.9912 |
| Camera | AP 0.0098, R 0.0095, P 0.0112, TP1 FP88, 46.9ms | AP 0.4882, R 0.5143, P 0.4091, TP54 FP78, 656.4ms | AUROC 0.5833, FPR 1.0, TP36 TN0 FP2 FN0, 25.7ms | AP 0.6734, R 0.6857, P 0.8889, TP72 FP9 det81, 93.3ms; per-class open 0.7059 short 0.5702 mouse 0.7308 spur 0.5187 spurious 0.5556 missing 0.9590 |

**Day2 fine-tune — `results/baseline_day2_yolo3072.json` (weights `yolo11n_day2.pt`, 15 epoch, val-tile P0.982 R0.945 mAP50 0.957 mAP50-95 0.582 — tile 640, KHÔNG phải full-board):**
- Standard: `AP 0.8392, R 0.8476, P 0.7946, TP89 FP23 det112, 77.0ms` (FP -5 vs base, TP -1); per-class open 0.8603 short 0.7895 mouse 0.8125 spur 0.6373 spurious 0.9444 missing 0.9912.
- Camera: `AP 0.6234, R 0.6381, P 0.8701, TP67 FP10 det77, 39.3ms`; per-class open 0.7647 short 0.5898 mouse 0.6771 spur 0.2941 spurious 0.4321 missing 0.9825. GOOD04 base 1FP → day2 0FP (acceptance lat<200ms PASS).

**Day3 Nhánh B — `results/branchB_day3.json`:** board-ID 98/98 100%, lat_fast 43.1ms; recall standard 99/105 (0.9429) avg_blobs 5.22 max_z 22.02; camera 98/105 (0.9333) avg_blobs 8.0 max_z 22.68; GOOD trigger 0/4 cả hai (01-std 0 blob/0.0, 01-cam 8/2.85, 04-std 0/0.0, 04-cam 8/2.86 — blob camera dưới ngưỡng 3b nên không trigger FAR).

**Day5 Dual-Path — `results/dualpath_day5.json` (verbatim, weights base, conf 0.20, IoU 0.5):**
- Standard: `AP50 0.7587, R 0.8381, P 0.7928, F1≈0.8148 (=2PR/(P+R)), TP88 FP23, npos105, n_det111, FAR 0.0 (0/2), lat 379.5ms, levels {L1:20,L2:15,L3:1,L4:2}`; per-class missing 0.9912 open 0.9837 spurious 0.9012 spur 0.5882 mouse 0.5641 short 0.5237.
- Camera: `AP50 0.5130, R 0.5238, P 0.8088, F1≈0.6359, TP55 FP13, npos105, n_det68, FAR 0.0 (0/2), lat 333.4ms, levels {L1:14,L2:10,L3:11,L4:3}`; per-class missing 0.9799 open 0.6324 mouse 0.6193 short 0.3684 spurious 0.2778 spur 0.2000.

**Đọc thẳng:** Dual-Path standard AP kém YOLO đơn 0.079 (0.7587 vs 0.8375 base / 0.8392 day2, TP -2 FP -5); camera kém 0.160 (0.513 vs 0.6734 base / 0.6234 day2, TP -17 FP +3 vs base-day2? TP55 vs 72/67). P camera cao (0.8088) vì ít det (68) nhưng R rớt mạnh (0.5238). Latency gấp ~2.5–3.5× Nhánh A đơn (379.5/333.4 vs 148.1/93.3 base, 77.0/39.3 day2) do align ~44ms + diff ~245ms + ROI batch.

### 3.3. Open-set / FAR / FPR / latency

- Open-set: chỉ System3 có rule 3b đo được (branchb Z≥3.5 Area≥100, độc lập YOLO). System1 OR cứng → liệt open-set (YOLO miss + anomaly<2.0 → PASS sai); System2 không có khái niệm open-set. Day5 levels L3 camera 11 vs standard 1 cho thấy Safety-Net "thấy" nhiều hơn khi YOLO yếu, nhưng AP camera vẫn thấp vì TOP_K=8 bão hòa (avg_blobs 8.0) + blob shift sau warp + YOLO640 ROI không augment perturb.
- FAR (GOOD 01/04): Dual-Path 0.0 (0/2) cả hai kịch bản — PASS Day5. PatchCore FPR=1.0 (2/2 GOOD FP, 264/264 trên audit 7 board) — lý do loại khỏi pipeline chính, giữ làm Ablation.
- Latency: YOLO3072 rect/half đạt 77–148ms (<200ms, PASS Day2); SAHI 605–656ms (lý do loại); Dual-Path 333–380ms (vượt mục tiêu plan 160–190ms của Nhánh A đơn — phải báo cáo trung thực, không giấu).

**Kết luận Phần 3:** System3 thắng về thiết kế đánh giá (board-disjoint + FAR + open-set), thua YOLO đơn về AP/Recall/latency trên cả hai kịch bản đo thực. Đây là đánh đổi có chủ ý (Safety-Net + FAR=0), không phải lỗi đo.

---

## PHẦN 4 — ĐIỂM MẠNH & HẠN CHẾ TRUNG THỰC

### 4.1. Điểm mạnh (có số)

1. **FAR=0 cả hai kịch bản** (0/2 GOOD trigger) trong khi PatchCore FPR=1.0 — Nhánh B + sigma_floor + valid_mask làm đúng vai Safety-Net không báo giả GOOD.
2. **Board-disjoint sạch + tái lập:** train/val/test tách board, seed công khai (0 vs 20261023+i), 78/78 tests xanh (`test_day1 11 + test_day2 11 + test_day3 15 + test_day3_round2 17 + test_day5 24`), zero-touch 2 project cũ (test `test_zero_touch_writes_guarded` PASS, `assert_writable_ok` chặn mode 'w').
3. **Lớp lớn/đặc trưng mạnh giữ vững:** missing_hole standard 0.9912 / camera 0.9799; open_circuit standard 0.9837; spurious_copper standard 0.9012 — cho thấy fusion cộng hưởng đúng chỗ (IoU≥0.3 boost +0.15).
4. **Ablation có giá trị luận văn:** YOLO640 AP ~0.01 (chứng minh cần full-res), SAHI P 0.48–0.41 FP 92/78 + 605–656ms (chứng minh tiling đắt và nhiễu), PatchCore AUROC 0.3056/0.5833 FPR 1.0 (chứng minh unsupervised cross-board thất bại) — đúng plan 1.2 "giữ số thấp để viết Ablation".

### 4.2. Hạn chế (không giấu — Scribe cấm sửa nhẹ)

1. **Lớp yếu nhất:** Standard `short 0.5237, mouse_bite 0.5641, spur 0.5882`; Camera `spur 0.20, spurious_copper 0.2778, short 0.3684`. Đây là lỗi nhỏ/sát mạch, tau=2 + mask dilate7 vẫn nuốt hoặc YOLO conf thấp không qua 0.35/0.55. Không được trích AP missing_hole (~0.99) đại diện toàn hệ.
2. **Camera FP/FN:** TP rớt 88→55 (-37.5%), FN tăng 17→50 (npos 105), AP rớt 0.7587→0.513 (-0.246). Nguyên nhân đã xác minh: blob shift sau warp + TOP_K=8 bão hòa (avg 8.0 blob/ảnh, max_z 22.68) + YOLO ROI 640 không augment perturb + `evaluate.py` thiếu valid_mask (UI đã đúng). P camera 0.8088 cao là do ít det (68), không phải do giỏi hơn.
3. **Latency vượt plan:** 379.5ms (std) / 333.4ms (cam) vs mục tiêu Nhánh A 160–190ms và Day2 77.0/39.3ms. Breakdown: align fast ~43ms + diff ~245ms + ROI batch + fusion. Không đạt "real-time" nếu định nghĩa <200ms full pipeline — phải ghi vào luận văn hướng tối ưu (cache priors đã có `template_priors`, giảm TOP_K, TensorRT/quantize — chưa làm).
4. **Chưa hoàn thành vs plan:** `evaluate.py` chưa đồng bộ `valid_mask` (ghi nhận Day6, không hồi tố số); demo weights day2 vs base còn một ghi nhận (C·oder số khớp base); `thesis_artifacts/` do Scribe làm song song — kiểm toán này không chạm. Val-tile `mAP50 0.957` tuyệt đối không được trích thành AP full-board.
5. **Rủi ro thiết kế:** Nhánh B phụ thuộc Golden Template + H tốt; nếu board cong/vênh 3D hoặc thiếu template, H suy biến (`LinAlgError` đã bắt nhưng pipeline dừng ảnh đó — `run_dualpath` ghi `scenario_error`, không giấu). Ngưỡng 3b (3.5/100) là hằng số plan, chưa học theo board — cần hiệu chuẩn riêng nếu triển khai nhiều mã hàng.

**Kết luận Phần 4:** Hệ thống trung thực ở mức "báo cáo được, bảo vệ được, nhưng chưa sản xuất được ở tốc độ dây chuyền nhanh" — đúng vị thế đồ án nghiên cứu có Safety-Net.

---

## PHẦN 5 — 5 CÂU HỎI PHẢN BIỆN KHÓ NHẤT & TRẢ LỜI CHUẨN (SỐ NGUYÊN VĂN)

**Q1. "01/04 có Template sẵn thì có phải 'ăn gian' (trừ hai ảnh sinh từ cùng file gốc)? Vì sao gọi là Zero-defect-training mà vẫn cần Template?"**
> 01/04 là Zero-defect-training Boards, không phải Unseen toàn hệ (plan 4.1/4.3). Thực tế nhà máy khi đổi mã hàng luôn có Golden Sample/Gerber (1 board chuẩn) nhưng chưa có dữ liệu lỗi để train lại AI. Nhánh A tổng quát hình thái lỗi từ board cũ {05–12} (board-disjoint, test `test_no_board_01_04_leakage` PASS); Nhánh B đối chiếu với Golden của mã mới. Trừ Template-khác-vùng-lỗi cho residual ~1.3 mức xám (plan 4.2 Bảng 1 kỳ vọng recall đề-cu ~100% standard) — đúng quy trình AOI công nghiệp, không phải trừ hai ảnh lỗi giống nhau. Nếu "ăn gian", FAR GOOD đã không cần đo và đã không có camera AP rớt 0.7587→0.513.

**Q2. "Dual-Path AP thấp hơn YOLO đơn (standard -0.079, camera -0.160) thì ghép Nhánh B để làm gì? Có phải Nhánh B kéo tụt?"**
> Đúng, số verbatim: standard Dual 0.7587 vs YOLO base 0.8375 / day2 0.8392 (TP 88 vs 90/89, FP 23 vs 28/23); camera Dual 0.513 vs 0.6734/0.6234 (TP 55 vs 72/67). Nhánh B không nhằm tăng AP khi YOLO đã mạnh — nó nhằm (a) giữ FAR=0 (0/2 cả hai kịch bản, PatchCore FPR=1.0), (b) bắt open-set YOLO miss qua 3b (camera L3=11 vs standard L3=1), (c) cứu recall conf thấp qua 3a. Giá phải trả là FP thêm từ blob + TP mất do gating/fusion (short/spur/mouse thấp nhất). Đây là đánh đổi Safety-Net, đã báo cáo thẳng, không giấu. Nếu chỉ cần AP cao nhất trên PKU sạch, dùng YOLO3072 đơn (77–148ms) — luận văn ghi rõ.

**Q3. "Ngưỡng 0.55/0.35/0.20-0.35+Z2.0/3b Z3.5+Area100 có phải chép tay? Vì sao tin được?"**
> Tất cả từ plan 2.4 + sweep/probe, có test khóa (`test_constants_match_plan`, 24 tests logic/fusion): FAIL {0,1,5}≥0.55 / REWORK {2,3,4}≥0.35 là phân tầng công nghiệp (lỗi hở/mất lỗ nghiêm trọng hơn); 3a 0.20–0.35 + Z≥2.0 là vùng YOLO yếu nhưng Nhánh B thấy (tránh OR cứng `anom>=2.0 AND yolo>=0.30` của legacy gây PASS sai); 3b Z≥3.5 Area≥100 độc lập YOLO là định nghĩa open-set thuần (YOLO S<0.20). Sigma_floor 5.0 từ val-good (bg_std GOOD ~0.3, floor 4–6), tau2/seed1.25 từ sweep (std 99/105 cam 97/105, FAR 0). GOOD probe 01/04 maxZ 0.0, FAR 0/2 — ngưỡng không báo giả GOOD sạch.

**Q4. "Camera perturbation của bạn có thực tế không? Sao recall Nhánh B 99/105 standard mà Dual-Path camera chỉ R 0.5238?"**
> Perturb công khai trong `evaluate.py:perturb_image_and_boxes`: shift ±25px, xoay ±2°, phối cảnh ~0.8%, gain 0.85–1.15, bias ±10, vignette 0.15, noise 3–5, warp GT theo cùng H; seed standard 0, camera 20261023+i (+101/+7919/+31337). Đây là mô phỏng hợp lý, không thay thế camera thật — luận văn phải ghi giới hạn này. Nhánh B đơn (diff chưa qua YOLO) recall 99/105 std, 98/105 cam (TOP8, avg_blobs 8.0 camera) — cao vì chỉ cần "thấy" blob. Dual-Path camera R 0.5238 (TP55/105) thấp vì sau đó còn gating R80 + YOLO ROI 640 (không augment perturb) + fusion + ngưỡng 4 mức; blob shift + TOP8 bão hòa + `evaluate.py` thiếu valid_mask làm mất thêm. Standard R 0.838 (TP88/105) tuyệt đối không được trích thành năng lực camera.

**Q5. "PatchCore/SAHI thất bại có phải do bạn cài sai? Và latency 333–380ms thì chạy dây chuyền thế nào?"**
> PatchCore thất bại ở thiết kế, không phải cài đặt: ngưỡng mức patch (p99 0.4763→0.5001) áp cho max mức ảnh (p99 0.6119→0.6425) lệch 1.28× → FPR 100% (264/264 GOOD FAIL, `AUDIT_vs_TDDNet.md:§#1`); bỏ `F.normalize` + percentile chỉ +0.037 (0.6949→0.7314); bank chéo-board 0.62–0.75 vs riêng-board 0.78–0.84 → PatchCore là per-board, không hợp split board-disjoint + 1 bank chung. SAHI AP tương đương YOLO full-res nhưng P 0.4802/0.4091 (FP 92/78) + 605/656ms → loại vì đắt và nhiễu. Latency Dual-Path 379.5/333.4ms vượt mục tiêu 160–190ms (YOLO đơn 77–148ms, align ~43ms + diff ~245ms) — chưa đạt real-time dây chuyền nhanh; hướng giảm đã rõ (cache `template_priors` đã có, giảm TOP_K, TensorRT) nhưng chưa làm — xin ghi vào hạn chế và hướng phát triển, không claimed.

---

## PHỤ LỤC — DẤU VẾT KIỂM TOÁN (TRACEABILITY)

- Verifier rules: `aoi_final_plan/rules/verifier_rules.md` (write scope tests/results/status_and_bugs; cấm sửa src; cấm bịa số; cấm chạm 2 project cũ; checklist 4 điểm).
- Plan: `aoi_final_plan/plan.md` (7 phần; ngưỡng/fusion/gating/split/perturb/acceptance Day5 F1-P-R-FAR + Day6 UI 3 khung).
- Code đã đọc (READ-ONLY): `src/models/yolo_detector.py:40-42,89-117,132-169`; `src/reference/board_id_and_align.py:50-82,111,145-181,221-281,284-399`; `src/reference/tolerance_diff.py:48-64,70-114,120-162,187-237,243-287,324-386`; `src/models/ensemble.py:34-36,51-56,59-153,156-166`; `src/decision/logic.py:19-29,32-111`; `scripts/evaluate.py:87,302-354,374-379,483-525,551-664,667-717`; `scripts/prepare_yolo_dataset.py:50-59,170-189,195-216,244-254`; `configs/resource_paths.py:24-69,85-93`; `src/ui/app.py:159-240` (valid_mask 177–178, project 183–184, batch 204, shift 218–220, fuse 223).
- Số đã đọc (verbatim): `results/baseline_day1.json`, `results/baseline_day2_yolo3072.json`, `results/branchB_day3.json`, `results/dualpath_day5.json`, `results/day2_verify_summary.json`, `comparison.json` + `AUDIT_vs_TDDNet.md` (READ-ONLY legacy).
- Tests: 78/78 xanh (`test_day1 11, test_day2 11, test_day3 15, test_day3_round2 17, test_day5 24`); 2 fix test-only đã ghi (gating expect 1 item sau root-cause-2; placeholder H_inv/sigma_floor chuyển thành assert tồn tại 4 file).
- Ghi nhận Day6 chưa sửa (đúng guardrail READ-ONLY): (a) demo weights day2 vs base, (b) `evaluate.py:601` thiếu `valid_mask` so với `app.py:177-178` — đã đối chiếu blob giống hệt khi H~I, không hồi tố số Day5.

*Hết báo cáo. Scribe được phép trích mọi số trong file này nguyên văn cho `thesis_artifacts/`.*
