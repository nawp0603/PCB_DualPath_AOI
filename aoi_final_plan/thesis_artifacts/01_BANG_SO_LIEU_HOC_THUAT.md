# TỔNG HỢP BẢNG SỐ LIỆU THỰC NGHIỆM CHUẨN HỌC THUẬT (MARKDOWN & LATEX)
**Dự án:** Đồ án tốt nghiệp (KLTN) — Hệ thống Kiểm tra Quang học Tự động Hai Nhánh (Dual-Path Industrial AOI) trên Mạch In Trần (Bare PCB)  
**Tập kiểm thử chuẩn:** **Zero-defect-training Boards `01` và `04`** (36 ảnh lỗi chứa 105 hộp nhãn Ground-Truth thuộc 6 lớp lỗi + 2 ảnh mẫu chuẩn Golden Templates `01.JPG`, `04.JPG`)  
**Thiết bị thực thi:** GPU NVIDIA GeForce RTX 3070 (`CUDA=True`, `torch 2.6.0+cu124`, `ultralytics 8.4.182`, `OpenCV 4.11.0`, `NumPy 1.26.4`)  
**Nguyên tắc xuất bản (Provenance & Data Integrity):**
1. **100% số liệu thực nghiệm** được trích xuất nguyên văn từ các tệp kết quả trong `results/` (`baseline_day1.json`, `day2_verify_summary.json`, `branchB_day3.json`, `probe_day3_openset_large.json`, `probe_day3_boardid_deterministic.json`, `dualpath_day5.json`) và tài liệu đối chiếu (`docs/AUDIT_vs_TDDNet.md`, `comparison.json`).
2. **Tách biệt tuyệt đối Kịch bản 1 (Standard Benchmark) và Kịch bản 2 (Realistic Camera Perturbation Benchmark)** thành các bảng riêng biệt theo đúng quy định tại `plan.md` (Phần 4.2) và `scribe_rules.md`.
3. Sử dụng nhất quán thuật ngữ **"Zero-defect-training Boards"** cho mã mạch `01` và `04` (bo mạch mới đưa vào dây chuyền chỉ có duy nhất 1 ảnh thiết kế chuẩn Golden Template, chưa từng có mẫu lỗi nào trong tập huấn luyện).

---

## 1. BẢNG NGHIÊN CỨU LOẠI BỎ (ABLATION STUDY): TẠI SAO LOẠI BỎ PATCHCORE, SAHI VÀ RESIZE 640?

### 1.1. Bảng 1A — Nghiên cứu Loại bỏ (Ablation Study) trên Kịch bản 1 (Standard Benchmark — Ảnh PKU Gốc)
> **Nguồn dữ liệu (Provenance):**
> - Lệnh chạy: `python scripts/evaluate.py --scenario both --out-json results/baseline_day1.json --out-csv results/baseline_day1.csv` (Ngày chạy: `2026-10-23`, GPU RTX 3070).
> - Số liệu Fine-tune Ngày 2: `results/day2_verify_summary.json` và `results/baseline_day2_yolo3072.json`.
> - Số liệu đối chiếu mở rộng của PatchCore trên 7 boards (264 ảnh GOOD): `D:\FPTU\KLTN\PCB_Anomaly_Inspector_Portable\docs\AUDIT_vs_TDDNet.md` (Mục 2 #1 & Mục 5.2, READ-ONLY).
> - Số liệu đối chiếu hợp nhất OR cứng (`H0_OR`): `D:\FPTU\KLTN\hybird_test_3branch-20260920T081632Z-1-001\hybird_test_3branch\results\comparison.json` (READ-ONLY).

| Cấu hình / Phương pháp | Vai trò trong Đồ án | AP@0.50 | Recall | Precision | F1-Score *(dẫn xuất)* | TP / 105 | FP | Image AUROC | Tỷ lệ Báo giả GOOD (FPR / FAR) | Độ trễ TB (ms/ảnh) | Quyết định Kiến trúc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **PatchCore (ResNet18 L2+L3)** | Baseline Unsupervised `[ABLATION]` | — | — | — | — | — | — | **0.3056** *(0.6949 gộp 6 boards)* | **100.0%** *(2/2 GOOD `01/04`; 264/264 trên 7 boards)* | 86.7 | **Loại bỏ hoàn toàn** (Không tổng quát hóa qua thiết kế mạch mới; làm liệt nhánh quyết định) |
| **Legacy Hybrid (`H0_OR` — OR cứng)** | Đối chiếu lịch sử (`comparison.json`) | — | 1.0000 | 0.5056 | 0.6716 | — | — | — | **97.78%** (`FAR = 0.9778`) | 30.7 | **Loại bỏ hoàn toàn** (Toán tử OR cứng bị chặn trên bởi FPR của nhánh dị thường) |
| **YOLO11n Full-board @640 (`rect=True`)** | Baseline Độ phân giải thấp `[ABLATION]` | **0.0119** | 0.0286 | 0.0227 | 0.0253 | 3 | 129 | — | — | 124.8 | **Loại bỏ** (Co ảnh `3034×1586` về `640` làm biến mất các lỗi vi mô `15–25 px`) |
| **SAHI (`slice=640, overlap=0.20`)** | Baseline Cắt mảnh Tiling `[ABLATION]` | **0.7644** | 0.8095 | **0.4802** | 0.6028 | 85 | **92** | — | — | **605.3** | **Loại bỏ hoàn toàn** (Mất ngữ cảnh toàn cục, FP tăng gấp `3.3×`, chậm gấp `4.1×`) |
| **YOLO11n Full-board @3072 (`Base`, `rect=True, half=True`)** | **Nhánh A Chuẩn (Trước Fine-tune)** | **0.8375** | **0.8571** | 0.7627 | 0.8072 | **90** | 28 | — | 50.0% *(1/2 GOOD: `04.JPG` 1 FP conf 0.31)* | 148.1 | **Giữ làm Nhánh A** (Khắc phục bẫy Square Padding, giữ nguyên chi tiết lỗi nhỏ) |
| **YOLO11n Full-board @3072 (`Day 2 FT`, `rect=True, half=True`)** | **Nhánh A Nâng cấp (Clip BBox + 25% BG)** | **0.8392** | 0.8476 | **0.7946** | **0.8203** | 89 | **23** | — | **0.0%** *(0/2 GOOD: triệt tiêu FP trên `04.JPG`)* | **77.0** | **Cấu hình Nhánh A Tối ưu cho Kịch bản 1** (Giảm 5 FP, tăng Precision +3.19%) |

---

### 1.2. Bảng 1B — Nghiên cứu Loại bỏ (Ablation Study) trên Kịch bản 2 (Realistic Camera Perturbation Benchmark)
> **Nguồn dữ liệu (Provenance):**
> - Lệnh chạy: `python scripts/evaluate.py --scenario both --out-json results/baseline_day1.json --out-csv results/baseline_day1.csv` (Kịch bản `camera`: dịch chuyển $\pm 25\text{ px}$, xoay $\pm 2^\circ$, biến dạng phối cảnh $\sim 0.8\%$, thay đổi sáng gain $0.85\text{--}1.15$, bias $\pm 10$, vignette $15\%$, nhiễu hạt Gaussian $\sigma \in [3, 5]$, có warp hộp nhãn GT tương ứng).

| Cấu hình / Phương pháp | Vai trò trong Đồ án | AP@0.50 | Recall | Precision | F1-Score *(dẫn xuất)* | TP / 105 | FP | Image AUROC | Tỷ lệ Báo giả GOOD (FPR / FAR) | Độ trễ TB (ms/ảnh) | Nhận xét Thực nghiệm dưới Nhiễu Camera |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **PatchCore (ResNet18 L2+L3)** | Baseline Unsupervised `[ABLATION]` | — | — | — | — | — | — | **0.5833** | **100.0%** *(2/2 GOOD bị phán FAIL)* | 25.7 | Điểm dị thường của ảnh GOOD (`0.7483`) cao hơn cả ảnh lỗi thật (`0.7276`) |
| **YOLO11n Full-board @640 (`rect=True`)** | Baseline Độ phân giải thấp `[ABLATION]` | **0.0098** | 0.0095 | 0.0112 | 0.0103 | 1 | 88 | — | — | 46.9 | Hoàn toàn mù trước lỗi nhỏ khi vừa bị co ảnh vừa chịu nhiễu quang học |
| **SAHI (`slice=640, overlap=0.20`)** | Baseline Cắt mảnh Tiling `[ABLATION]` | **0.4882** | 0.5143 | **0.4091** | 0.4557 | 54 | **78** | — | — | **656.4** | Precision sụp đổ còn `40.91%`, tạo `78` hộp báo giả và độ trễ `656.4 ms` |
| **YOLO11n Full-board @3072 (`Base`, `rect=True, half=True`)** | **Nhánh A Chuẩn (`yolo11n_base.pt`)** | **0.6734** | **0.6857** | **0.8889** | **0.7742** | **72** | **9** | — | — | 93.3 | Bền vững nhất dưới nhiễu Camera nhờ giữ nguyên ngữ cảnh toàn bo mạch |
| **YOLO11n Full-board @3072 (`Day 2 FT`, `rect=True, half=True`)** | Nhánh A Fine-tune (`yolo11n_day2.pt`) | **0.6234** | 0.6381 | 0.8701 | 0.7363 | 67 | 10 | — | — | **39.3** | Fine-tune thêm 15 epoch trên tile tĩnh không có phép tăng cường nhiễu phối cảnh làm giảm 5 TP (báo cáo trung thực) |

---

## 2. BẢNG ĐỐI CHIẾU TOÀN HỆ THỐNG TRÊN ZERO-DEFECT-TRAINING BOARDS (`01/04`) — TÁCH BIỆT 2 KỊCH BẢN

> [!IMPORTANT]
> **Quy tắc bất khả xâm phạm (`plan.md` Phần 4.2 & `scribe_rules.md`):** Kịch bản 1 (Standard Benchmark) và Kịch bản 2 (Realistic Camera Perturbation Benchmark) đại diện cho hai điều kiện vật lý hoàn toàn khác nhau. Tuyệt đối không gộp chung các con số của hai kịch bản vào cùng một bảng.

### 2.1. Bảng 2 — Kết quả Thực nghiệm Kịch bản 1 (Standard Benchmark — Ảnh PKU Gốc Căn Sẵn Tĩnh)
> **Nguồn dữ liệu (Provenance):** `results/baseline_day1.json`, `results/day2_verify_summary.json`, `results/branchB_day3.json`, và `results/dualpath_day5.json` (`python scripts/evaluate.py --scenario both --configs dualpath --out-json results/dualpath_day5.json`).

#### (a) Hiệu năng Tổng thể trên Tập Kiểm thử Zero-defect-training Boards `01/04` (Kịch bản 1: Standard)
| Hệ thống / Nhánh | AP@0.50 | Recall (Detection) | Precision | F1-Score *(dẫn xuất)* | TP / 105 | FP | Tổng Det | Recall Đề cử Tâm (Centroid) | FAR trên Ảnh GOOD (`01/04`) | Phân bố Mức Quyết định (36 Lỗi + 2 GOOD) | Độ trễ TB (ms/ảnh) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| **Nhánh A Đơn lẻ (`YOLO@3072 Base`)** | 0.8375 | **0.8571** | 0.7627 | 0.8072 | **90** | 28 | 118 | — | 50.0% *(1/2 GOOD có FP)* | Không có lưới an toàn Open-set | 148.1 |
| **Nhánh A Đơn lẻ (`YOLO@3072 Day 2 FT`)** | **0.8392** | 0.8476 | **0.7946** | **0.8203** | 89 | **23** | 112 | — | **0.0%** *(0/2 GOOD)* | Không có lưới an toàn Open-set | **77.0** |
| **Nhánh B Đơn lẻ (`Golden-Template Safety-Net`)** | — | — | — | — | 99 *(tâm)* | — | — | **94.29%** *(99/105)* | **0.0%** *(0/2 GOOD, `max Z = 0.0`)* | Bắt 100% mã mạch (`98/98`), đề cử vùng nghi ngờ | 43.1 *(align)* + 232.2 *(diff)* |
| **Hệ thống Hợp nhất Dual-Path AOI (Nhánh A + Nhánh B + Fusion 4 mức)** | **0.7587** | **0.8381** | **0.7928** | **0.8148** | **88** | **23** | 111 | **94.29%** *(tầng đề cử)* | **0.0%** *(0/2 GOOD PASS)* | **L1 FAIL: 20 \| L2 REWORK: 15 \| L3 REVIEW: 1 \| L4 PASS: 2 (GOOD)** | 379.5 |

#### (b) Phân rã AP@0.50 và Recall Đề cử theo Từng Lớp Lỗi trên Kịch bản 1 (Standard Benchmark)
| Lớp lỗi (Defect Class) | Nhóm Mức độ Nghiêm trọng | YOLO@640 AP50 | SAHI 640 AP50 `[ABLATION]` | YOLO@3072 (Base) AP50 | Dual-Path AOI AP50 | Nhánh B Centroid Recall (Số hộp bắt trúng / Tổng GT) |
|---|---|---:|---:|---:|---:|---:|
| `missing_hole` (Thiếu lỗ khoan) | Nghiêm trọng cao (`FAIL / HIGH`) | 0.0000 | 0.9667 | **0.9912** | **0.9912** | **18 / 18 (100.0%)** |
| `open_circuit` (Hở mạch) | Nghiêm trọng cao (`FAIL / HIGH`) | 0.0588 | 0.8725 | 0.8603 | **0.9837** *(+0.1234 nhờ Fusion)* | **17 / 17 (100.0%)** |
| `short` (Ngắn mạch / Chập mạch) | Nghiêm trọng cao (`FAIL / HIGH`) | 0.0000 | 0.3913 | **0.7268** | 0.5237 | **19 / 19 (100.0%)** |
| `spurious_copper` (Đồng thừa) | Trung bình (`REWORK / MEDIUM`) | 0.0000 | 0.8889 | **0.9971** | 0.9012 | **18 / 18 (100.0%)** |
| `mouse_bite` (Khuyết mẻ cạnh đồng) | Trung bình (`REWORK / MEDIUM`) | 0.0125 | 0.7657 | **0.8667** | 0.5641 | **13 / 16 (81.25%)** |
| `spur` (Gai đồng nhọn) | Trung bình (`REWORK / MEDIUM`) | 0.0000 | **0.7014** | 0.5829 | **0.5882** *(+0.0053 nhờ Fusion)* | **14 / 17 (82.35%)** |
| **Trung bình Toàn cục (mAP@0.50 / Tổng)** | **6 Lớp** | **0.0119** | **0.7644** | **0.8375** | **0.7587** | **99 / 105 (94.29%)** |

---

### 2.2. Bảng 3 — Kết quả Thực nghiệm Kịch bản 2 (Realistic Camera Perturbation Benchmark)
> **Nguồn dữ liệu (Provenance):** `results/baseline_day1.json`, `results/branchB_day3.json`, và `results/dualpath_day5.json` (Chạy trên 36 ảnh lỗi `01/04` + 2 ảnh GOOD đã chịu biến đổi hình học dịch $\pm 25\text{ px}$, xoay $\pm 2^\circ$, biến dạng phối cảnh Homography $\sim 0.8\%$, biến thiên độ sáng cục bộ, tối góc vignette và nhiễu cảm biến).

#### (a) Hiệu năng Tổng thể trên Tập Kiểm thử Zero-defect-training Boards `01/04` (Kịch bản 2: Camera Perturbation)
| Hệ thống / Nhánh | AP@0.50 | Recall (Detection) | Precision | F1-Score *(dẫn xuất)* | TP / 105 | FP | Tổng Det | Recall Đề cử Tâm (Centroid) | FAR trên Ảnh GOOD (`01/04`) | Phân bố Mức Quyết định (36 Lỗi + 2 GOOD) | Độ trễ TB (ms/ảnh) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| **Nhánh A Đơn lẻ (`YOLO@3072 Base`)** | **0.6734** | **0.6857** | **0.8889** | **0.7742** | **72** | **9** | 81 | — | — | Không có lưới an toàn Open-set | 93.3 |
| **Nhánh A Đơn lẻ (`YOLO@3072 Day 2 FT`)** | 0.6234 | 0.6381 | 0.8701 | 0.7363 | 67 | 10 | 77 | — | — | Không có lưới an toàn Open-set | **39.3** |
| **Nhánh B Đơn lẻ (`Golden-Template Safety-Net`)** | — | — | — | — | 98 *(tâm)* | — | — | **93.33%** *(98/105)* | **0.0%** *(0/2 GOOD, `max Z = 2.86 < 3.5`)* | Bắt 100% mã mạch (`80/80` stress $\pm 25/\pm 50\text{ px}$) | 43.1 *(align)* + 232.2 *(diff)* |
| **Hệ thống Hợp nhất Dual-Path AOI (ROI-Gated Mode)** | **0.5130** | **0.5238** | **0.8088** | **0.6359** | **55** | **13** | 68 | **93.33%** *(tầng đề cử)* | **0.0%** *(0/2 GOOD PASS)* | **L1 FAIL: 14 \| L2 REWORK: 10 \| L3 REVIEW: 11 \| L4 PASS: 3 (2 GOOD + 1 miss)** | 333.4 |

> **Chú giải học thuật trung thực về Kịch bản 2 (Đã xác minh tại Ngày 5 & Ngày 6 — `status_and_bugs.md` dòng 283 & 328):**
> 1. **Ở cấp độ phán quyết cấp ảnh (Board-level Decision):** Trong 36 ảnh lỗi dưới nhiễu Camera, hệ thống Dual-Path giữ lại **35/36 ảnh lỗi (97.22%)** trong các mức cảnh báo (`14 FAIL + 10 REWORK + 11 REVIEW`) và đạt **FAR = 0.0% tuyệt đối (2/2 ảnh GOOD đạt PASS)**.
> 2. **Ở cấp độ định vị hộp (Box-level AP50):** Trong `scripts/evaluate.py:601`, hàm `run_dualpath` gọi `td.run_branch_b(r["warped"], tpl)` mà chưa truyền `valid_mask=r.get("valid_mask")` (khác với `src/ui/app.py:177-178` và `tests/bench_day3_branchB.py` có truyền `valid_mask`), khiến trên ảnh Camera bị dịch/xoay, các blob giả ở vùng biên ngoài trường nhìn (`BORDER_REPLICATE`) làm bão hòa danh sách `TOP_K = 8` (`avg_blobs = 8.0`), đẩy một số lỗi thật (`spur`, `spurious_copper`) ra ngoài 8 mảnh cắt ROI `640×640` gửi sang YOLO (`Recall = 0.5238`, `AP50 = 0.5130` so với `0.6857` của `YOLO@3072` full-board). Khi truyền đầy đủ `valid_mask` (như trong `bench_day3_branchB.py` và `src/ui/app.py`), Recall đề cử tâm của Nhánh B trên Kịch bản 2 đạt tới **98/105 = 93.33%**. Theo đúng quy tắc trung thực số liệu của Verifier Ngày 6, báo cáo giữ nguyên số đo `dualpath_day5.json` và giải thích rõ nguyên nhân kỹ thuật này.

#### (b) Phân rã AP@0.50 và Recall Đề cử theo Từng Lớp Lỗi trên Kịch bản 2 (Camera Perturbation Benchmark)
| Lớp lỗi (Defect Class) | Nhóm Mức độ Nghiêm trọng | YOLO@640 AP50 | SAHI 640 AP50 `[ABLATION]` | YOLO@3072 (Base) AP50 | Dual-Path AOI AP50 | Nhánh B Centroid Recall (Số hộp bắt trúng / Tổng GT) |
|---|---|---:|---:|---:|---:|---:|
| `missing_hole` (Thiếu lỗ khoan) | Nghiêm trọng cao (`FAIL / HIGH`) | 0.0000 | **0.9942** | 0.9590 | **0.9799** *(+0.0209 nhờ Fusion)* | **18 / 18 (100.0%)** |
| `open_circuit` (Hở mạch) | Nghiêm trọng cao (`FAIL / HIGH`) | 0.0588 | 0.5640 | **0.7059** | 0.6324 | **17 / 17 (100.0%)** |
| `short` (Ngắn mạch / Chập mạch) | Nghiêm trọng cao (`FAIL / HIGH`) | 0.0000 | 0.2895 | **0.5702** | 0.3684 | **19 / 19 (100.0%)** |
| `spurious_copper` (Đồng thừa) | Trung bình (`REWORK / MEDIUM`) | 0.0000 | 0.1556 | **0.5556** | 0.2778 | **17 / 18 (94.44%)** |
| `mouse_bite` (Khuyết mẻ cạnh đồng) | Trung bình (`REWORK / MEDIUM`) | 0.0000 | 0.6513 | **0.7308** | 0.6193 | **13 / 16 (81.25%)** |
| `spur` (Gai đồng nhọn) | Trung bình (`REWORK / MEDIUM`) | 0.0000 | 0.2745 | **0.5187** | 0.2000 | **14 / 17 (82.35%)** |
| **Trung bình Toàn cục (mAP@0.50 / Tổng)** | **6 Lớp** | **0.0098** | **0.4882** | **0.6734** | **0.5130** | **98 / 105 (93.33%)** |

---

## 3. BẢNG KIỂM ĐỊNH CHUYÊN SÂU NHÁNH B (BOARD-ID, TOLERANCE DIFF, SAFE Z-SCORE & OPEN-SET)

> **Nguồn dữ liệu (Provenance):** `results/branchB_day3.json`, `results/probe_day3_boardid_deterministic.json`, `results/probe_day3_shortlist.json`, `results/probe_day3_autocalib.json`, và `results/probe_day3_openset_large.json`.

### Bảng 4 — Tổng hợp Kết quả Kiểm định Độc lập Nhánh B (Golden-Template Safety-Net)
| Hạng mục Kiểm định | Tập dữ liệu / Điều kiện Thử nghiệm | Chỉ số Đo lường | Trước khi Tối ưu / Sửa lỗi | Sau khi Áp dụng Chốt Kỹ thuật | Tiêu chí Nghiệm thu (`plan.md`) | Trạng thái |
|---|---|---|---|---|---|---|
| **1. Nhận diện mã mạch (Board-ID Benchmark)** | 98 ảnh (36 Std + 36 Cam + 6 Full-parity + 16 Train/Val + 4 GOOD) | Độ chính xác (`Accuracy`) | 98 / 98 (100.0%) | **98 / 98 (100.0%)** | Đúng 100% trên tập kiểm thử | **ĐẠT** |
| **2. Nhận diện mã mạch dưới Nhiễu Mạnh (Stress Test `BUG-DAY3-04`)** | 80 biến thể tất định (10 boards × 8 seeds, dịch **$\pm 25\text{ px}$**, xoay $\pm 2^\circ$, gain/bias/vignette) | Độ chính xác (`Accuracy`) | 79 / 80 (98.75% — 1 ca lọt top-2 shortlist) | **80 / 80 (100.0%)** *(nhờ cơ chế Escalation khi `inliers < 60`)* | 100% không sụp đổ khi shortlist trượt | **ĐẠT** |
| **3. Nhận diện mã mạch dưới Nhiễu Cực đại (Extreme Shift $\pm 50\text{ px}$)** | 80 biến thể tất định (10 boards × 8 seeds, dịch **$\pm 50\text{ px}$**) | Độ chính xác (`Accuracy`) | 77 / 80 (96.25% — 3 ca lọt top-2 shortlist) | **80 / 80 (100.0%)** *(Escalation kích hoạt 3/80 ca)* | Bền vững dưới lệch cơ khí lớn | **ĐẠT** |
| **4. Độ trễ Nhận diện & Căn chỉnh (`fast=True`)** | 36 ảnh `3034×1586` trên RTX 3070 + CPU (`CV_THREADS` khôi phục đa luồng) | Trung vị độ trễ (`Median Latency`) | 1804.1 ms *(bị `ultralytics` ép về 1 thread + match 10 templates)* | **43.1 ms** *(vân tay Gradient 48×96 + ORB-3000 + đa luồng)* | $\le 45.0\text{ ms/ảnh}$ | **ĐẠT** |
| **5. Recall Đề cử Nhánh B (Kịch bản 1 Standard)** | 36 ảnh lỗi PKU gốc (`105` GT boxes trên Board `01/04`) | Centroid Recall (qua $H^{-1}$) | 59 / 105 (56.19%) | **99 / 105 (94.29%)** | $\ge 87/94$ ($\approx 92.55\%$) | **ĐẠT** |
| **6. Recall Đề cử Nhánh B (Kịch bản 2 Camera)** | 36 ảnh lỗi nhiễu Camera (`105` Warped GT boxes trên Board `01/04`) | Centroid Recall (qua $H^{-1}$) | 47 / 105 (44.76% — 59% blob giả ngoài FOV) | **98 / 105 (93.33%)** *(nhờ `valid_mask_from_h` + `fill_out_of_fov`)* | $\ge 87/94$ ($\approx 92.55\%$) | **ĐẠT** |
| **7. Tỷ lệ Báo giả trên Ảnh Sạch (GOOD Clean FAR)** | 4 mẫu GOOD (`01.JPG`, `04.JPG` ở cả Standard và Camera) | Số ca kích hoạt nhầm Rule 3b ($Z \ge 3.5 \land \text{Area} \ge 100$) | 1 / 4 (`01.JPG` Camera `max Z = 6.17`) | **0 / 4 (FAR = 0.0%)** *(Std `max Z = 0.00`; Cam `max Z = 2.85 / 2.86 < 3.5`)* | `FAR = 0%` trên ảnh GOOD | **ĐẠT** |
| **8. Kiểm định Bắt lỗi nhỏ Tự hiệu chuẩn (`probe_day3_autocalib`)** | 24 kịch bản lỗi nhân tạo kích thước `20–260 px`, độ lệch sáng $\Delta \in \{40, 80, 120\}$ | Tỷ lệ phát hiện (`Hit Rate`) | — | **24 / 24 (100.0%)** *(cả khi bật và tắt `auto_calib_tol`)* | Không nuốt lỗi thật khi tự hiệu chuẩn | **ĐẠT** |
| **9. Kiểm định Bắt lỗi lạ Open-Set (`probe_day3_openset_large`)** | 20 kịch bản mảng dị vật lớn (`2%–35%` diện tích mạch, $\Delta \in \{60, 100\}$) | Hit Rate & Tỷ lệ kích hoạt Rule 3b | 0 / 20 *(kiến trúc cũ bị khóa bởi `AND yolo_conf >= 0.30`)* | **Hit: 20 / 20 (100.0%) \| Rule 3b: 17 / 20 (85.0%)** *(100% với mảng $\le 20\%$ diện tích)* | Phát hiện độc lập lỗi ngoài 6 lớp | **ĐẠT** |

---

### Bảng 5 — Kết quả Kiểm định Thực tế Giao diện Giám sát 3 Khung hình Ngày 6 (`src/ui/app.py::run_dualpath_single`)
> **Nguồn dữ liệu (Provenance):** Kiểm định độc lập trên GPU RTX 3070 sau warmup của Verifier Ngày 6 (`aoi_final_plan/status_and_bugs.md` dòng 311–329, ngày `2026-10-24`). Luồng thực thi có truyền đầy đủ `valid_mask=r.get("valid_mask")` vào `run_branch_b`.

| Mẫu Kiểm thử Thực tế | Trọng số Nhánh A | Board-ID (`fast=True`) | Số Blob / Crops (`kept_per_crop`) | Bằng chứng Hợp nhất (`items`) | Kết luận 4 Mức (`decide`) | Độ trễ sau Warmup (Align / Diff / ROI / Tổng) |
|---|---|---|---|---|---|---|
| **`Open_circuit/open_circuit01.jpg`** (3 GT boxes) | `yolo11n_base.pt` | `'01'` (`177` inliers, `escalated=False`) | 3 blobs / 3 crops (`kept=[1, 1, 1]`) | 3 items `open_circuit` (`S_final = 0.639` cao nhất) | **Mức 1: `FAIL / HIGH`** ($S_{\text{final}} = 0.639 \ge 0.55$) | Warmup run: `151.3 / 214.1 / 1300.7 / 1697.7 ms` |
| **`Open_circuit/open_circuit01.jpg`** (3 GT boxes) | `yolo11n_day2.pt` *(Mặc định UI)* | `'01'` (`177` inliers, `escalated=False`) | 3 blobs / 3 crops (`kept=[1, 0, 1]`) | 2 `yolo` (`0.402, 0.442`) + 1 `branchb` ($Z=5.75, \text{Area}=117$) | **Mức 3b: `REVIEW / UNKNOWN_DEFECT`** *(Cứu lỗi ở blob thứ 2 khi `day2.pt` thận trọng)* | **`41.0 / 167.0 / 15.0 / 255.0 ms`** |
| **Golden Template `01.JPG` (GOOD)** | `yolo11n_day2.pt` / `base.pt` | `'01'` (`168` inliers, `escalated=False`) | 0 blobs / 0 crops ($\max Z_{\text{diff}} = 0.0$) | 0 items | **Mức 4: `PASS / NONE`** ($\text{FAR} = 0.0\%$) | **Tổng: `223.0 ms`** (`0` lần gọi GPU ROI) |
| **Golden Template `04.JPG` (GOOD)** | `yolo11n_day2.pt` / `base.pt` | `'04'` (`148` inliers, `escalated=False`) | 0 blobs / 0 crops ($\max Z_{\text{diff}} = 0.0$) | 0 items | **Mức 4: `PASS / NONE`** ($\text{FAR} = 0.0\%$) | **Tổng: `~223.0 ms`** (`0` lần gọi GPU ROI) |

---

## 4. MÃ NGUỒN BẢNG LATEX (`booktabs`) ĐỂ ĐƯA THẲNG VÀO BÁO CÁO KHOA HỌC

Dưới đây là mã LaTeX chuẩn học thuật (sử dụng gói `\usepackage{booktabs}`, `\usepackage{multirow}`) cho 3 bảng cốt lõi nhất của khóa luận:

```latex
% ============================================================================
% BẢNG 1: NGHIÊN CỨU LOẠI BỎ (ABLATION STUDY) TRÊN ZERO-DEFECT-TRAINING BOARDS 01/04
% Nguồn: results/baseline_day1.json & results/day2_verify_summary.json
% ============================================================================
\begin{table}[htbp]
\centering
\caption{Nghiên cứu loại bỏ (Ablation Study) chứng minh lý do loại bỏ PatchCore, SAHI và phép co ảnh 640\,px trên tập kiểm thử \textit{Zero-defect-training Boards} (01 và 04, Kịch bản 1 Standard).}
\label{tab:ablation_study}
\small
\begin{tabular}{lcccccccr}
\toprule
\textbf{Cấu hình phương pháp} & \textbf{AP@0.50} & \textbf{Recall} & \textbf{Precision} & \textbf{F1} & \textbf{TP/105} & \textbf{FP} & \textbf{AUROC / FPR} & \textbf{Độ trễ (ms)} \\
\midrule
PatchCore (ResNet18 L2+L3) [Ablation] & -- & -- & -- & -- & -- & -- & 0.3056 / \textbf{100\%} & 86.7 \\
Legacy Hybrid (\texttt{H0\_OR} cứng)$^{\dagger}$ & -- & 1.0000 & 0.5056 & 0.6716 & -- & -- & -- / \textbf{97.8\%} & 30.7 \\
YOLO11n Full-board @640 (\texttt{rect=True}) & 0.0119 & 0.0286 & 0.0227 & 0.0253 & 3 & 129 & -- & 124.8 \\
SAHI (\texttt{slice=640, ov=0.20}) [Ablation] & 0.7644 & 0.8095 & 0.4802 & 0.6028 & 85 & \textbf{92} & -- & \textbf{605.3} \\
\midrule
\textbf{YOLO11n @3072 (Base, \texttt{rect+half})} & 0.8375 & \textbf{0.8571} & 0.7627 & 0.8072 & \textbf{90} & 28 & -- / 50.0\% & 148.1 \\
\textbf{YOLO11n @3072 (Day 2 FT, \texttt{rect+half})} & \textbf{0.8392} & 0.8476 & \textbf{0.7946} & \textbf{0.8203} & 89 & \textbf{23} & -- / \textbf{0.0\%} & \textbf{77.0} \\
\bottomrule
\multicolumn{9}{l}{\footnotesize $^{\dagger}$Trích từ \texttt{comparison.json} và \texttt{AUDIT\_vs\_TDDNet.md} (FPR = 100\% trên 264/264 ảnh GOOD thuộc 7 mẫu mạch).}
\end{tabular}
\end{table}

% ============================================================================
% BẢNG 2: KỊCH BẢN 1 (STANDARD BENCHMARK — ẢNH PKU GỐC CĂN SẴN TĨNH)
% Nguồn: results/baseline_day1.json, results/branchB_day3.json, results/dualpath_day5.json
% ============================================================================
\begin{table}[htbp]
\centering
\caption{Kết quả đánh giá Kịch bản 1 (\textit{Standard Benchmark} -- Ảnh PKU gốc căn sẵn tĩnh) trên tập \textit{Zero-defect-training Boards} \texttt{01/04} (36 ảnh lỗi / 105 GT boxes + 2 ảnh GOOD).}
\label{tab:scenario1_standard}
\small
\begin{tabular}{lcccccccr}
\toprule
\textbf{Hệ thống / Nhánh} & \textbf{AP@0.50} & \textbf{Recall} & \textbf{Precision} & \textbf{F1} & \textbf{TP} & \textbf{FP} & \textbf{FAR (GOOD)} & \textbf{Độ trễ (ms)} \\
\midrule
Nhánh A: YOLO11n @3072 (Base) & 0.8375 & \textbf{0.8571} & 0.7627 & 0.8072 & \textbf{90} & 28 & 50.0\% (1/2) & 148.1 \\
Nhánh A: YOLO11n @3072 (Day 2 FT) & \textbf{0.8392} & 0.8476 & \textbf{0.7946} & \textbf{0.8203} & 89 & \textbf{23} & \textbf{0.0\%} (0/2) & \textbf{77.0} \\
Nhánh B: Golden-Template Safety-Net & -- & 0.9429$^{\ast}$ & -- & -- & 99$^{\ast}$ & -- & \textbf{0.0\%} (0/2) & 275.3 \\
\textbf{Dual-Path AOI (Nhánh A + B + Fusion)} & 0.7587 & 0.8381 & 0.7928 & 0.8148 & 88 & \textbf{23} & \textbf{0.0\%} (0/2) & 379.5 \\
\bottomrule
\multicolumn{9}{l}{\footnotesize $^{\ast}$Recall đề cử tâm blob (Centroid Recall) của Nhánh B chiếu qua $H^{-1}$ rơi vào trong hộp nhãn GT ($\pm 3$\,px).}
\end{tabular}
\end{table}

% ============================================================================
% BẢNG 3: KỊCH BẢN 2 (REALISTIC CAMERA PERTURBATION BENCHMARK — ĐỘC LẬP VỚI BẢNG 2)
% Nguồn: results/baseline_day1.json, results/branchB_day3.json, results/dualpath_day5.json
% ============================================================================
\begin{table}[htbp]
\centering
\caption{Kết quả đánh giá Kịch bản 2 (\textit{Realistic Camera Perturbation Benchmark}: dịch chuyển $\pm 25$\,px, xoay $\pm 2^\circ$, phối cảnh $\sim 0.8\%$, lệch sáng, tối góc vignette và nhiễu cảm biến) trên tập \textit{Zero-defect-training Boards} \texttt{01/04}.}
\label{tab:scenario2_camera}
\small
\begin{tabular}{lcccccccr}
\toprule
\textbf{Hệ thống / Nhánh} & \textbf{AP@0.50} & \textbf{Recall} & \textbf{Precision} & \textbf{F1} & \textbf{TP} & \textbf{FP} & \textbf{FAR (GOOD)} & \textbf{Độ trễ (ms)} \\
\midrule
YOLO11n Full-board @640 (\texttt{rect=True}) & 0.0098 & 0.0095 & 0.0112 & 0.0103 & 1 & 88 & -- & 46.9 \\
SAHI (\texttt{slice=640, ov=0.20}) [Ablation] & 0.4882 & 0.5143 & 0.4091 & 0.4557 & 54 & 78 & -- & 656.4 \\
Nhánh A: YOLO11n @3072 (Base) & \textbf{0.6734} & \textbf{0.6857} & \textbf{0.8889} & \textbf{0.7742} & \textbf{72} & \textbf{9} & -- & 93.3 \\
Nhánh A: YOLO11n @3072 (Day 2 FT) & 0.6234 & 0.6381 & 0.8701 & 0.7363 & 67 & 10 & -- & \textbf{39.3} \\
Nhánh B: Golden-Template Safety-Net & -- & 0.9333$^{\ast}$ & -- & -- & 98$^{\ast}$ & -- & \textbf{0.0\%} (0/2) & 275.3 \\
\textbf{Dual-Path AOI (ROI-Gated Fusion)} & 0.5130 & 0.5238 & 0.8088 & 0.6359 & 55 & 13 & \textbf{0.0\%} (0/2) & 333.4 \\
\bottomrule
\multicolumn{9}{l}{\footnotesize $^{\ast}$Nhánh B đạt độ chính xác nhận diện mã mạch 100\% (80/80 ca stress $\pm 25/\pm 50$\,px) và giữ 35/36 ảnh lỗi ở mức FAIL/REWORK/REVIEW.}
\end{tabular}
\end{table}
```
