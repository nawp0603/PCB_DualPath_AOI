# KE HOACH TRIEN KHAI HE THONG DUAL-PATH INDUSTRIAL AOI
## Phat hien loi mach PCB tran (6 lop loi + Open-set safety-net)

**Du an:** Do an tot nghiep (KLTN) — PCB Anomaly Inspector Portable
**Tac gia:** Principal Computer Vision Architect
**Ngay:** 2026-10-23
**Thu muc thuc thi doc lap:** `D:\FPTU\KLTN\PCB_DualPath_AOI\` (khong cham 2 project cu)
**Tai lieu goc:** Ban ke hoach chi tiet cua KLTN-AOI-Model (7 phan ky thuat).

---

## PHAN 1: TONG QUAN CHIEN LUOC & LY DO LOAI BO HUONG CU

### 1.1. Muc tieu
Phat hien va phan loai loi tren mach PCB tran voi 6 lop loi chuan (`open_circuit`, `short`, `missing_hole`, `mouse_bite`, `spur`, `spurious_copper`) va kha nang bat loi la Open-set (vi du: vet o hoa chat, xuoc phip lon, di vat roi vao mach, chay mach). Dat do chinh xac cao nhat tren ca du lieu chuan (PKU goc) va du lieu mo phong nhieu Camera thuc te (dich chuyen, xoay, phoi canh, thay doi sang, vignette, nhieu cam bien).

### 1.2. Quyet dinh kien truc chot
| Quyet dinh | Chi tiet |
|---|---|
| **Kien truc chinh** | **Dual-Path Industrial AOI**: Nhanh A (YOLO11n Full-Resolution) + Nhanh B (Golden-Template Safety-Net dua tren ORB/SIFT Homography & Tolerance-Aware Difference) + Hop nhat quyet dinh 4 muc. |
| **Loai bo hoan toan PatchCore** | Khong chay trong pipeline thuc te. Ly do: Cross-board AUROC chi dat 0.3056-0.69, FPR = 100% (264/264 anh GOOD bi gan FAIL), khong tong quat hoa qua cac thiet ke mach khac nhau. |
| **Loai bo hoan toan SAHI** | Khong chay trong pipeline thuc te. Ly do: AP50 tuong duong YOLO full-res nhung Precision tut tu 0.865 xuong 0.489 (FP tang tu 14 len 95) va lam cham toc do suy luan. |
| **Tan dung cho Bao cao Do an** | Giu nguyen so lieu thuc nghiem thap cua PatchCore (AUROC 0.3056 board-disjoint, FPR 100%) va SAHI (P 0.489, FP 95) de dua vao muc *"Ablation Study: Tai sao SOTA Unsupervised (PatchCore) va Tiling (SAHI) that bai tren mach PCB nguyen tam?"*. |
| **Quy tac trung thuc so lieu** | Khong ghi bat ky con so F1/Precision muc tieu nao (vi du F1 = 0.920) thanh ket qua da do khi chua chay `evaluate.py`. |

---

## PHAN 2: KIEN TRUC CHI TIET HE THONG DUAL-PATH INDUSTRIAL AOI

### 2.1. So do khoi tong quan

```
ANH DAU VAO (Full-board ~1586x3034 hoac Camera 4K, RGB uint8)
|
+-> [NHANH A] Semantic Defect Detector — YOLO11n Full-Res
|     +- Suy luan toan khung hinh: imgsz=3072 (hoac 3584), conf=0.20
|     +- Bat buoc: rect=True, half=True (tranh Square Padding)
|     +- Dau ra: B_A = { (toa do, class 1..6, conf S_A) }
|
+-> [NHANH B] Golden-Template Safety-Net — Coarse-to-Fine Registration
|     +- Buoc 1: Board ID & Coarse Align (ORB/SIFT + RANSAC Homography H + H_inv)
|     +- Buoc 2: Photometric Normalization (Local Gain-Bias / CLAHE)
|     +- Buoc 3: Tolerance-Aware Diff (tau=2-3 px, Copper Mask, Residual Tolerance Map)
|     +- Buoc 4: Connected Components (Area >= 60 px2, tinh Z_diff)
|     +- Buoc 5: ROI Verification & Spatial Center Gating (crop 640x640, R=80 px)
|        +- Dau ra: B_B = { (toa do, nhan YOLO S_B, hoac khong nhan nhung Z_diff cao) }
|
+-> [MODULE HOP NHAT] Spatial & Decision Fusion — 4 muc
      +- Box Fusion: IoU(B_A, B_B) >= 0.3 -> S_final = min(1.0, S_A + 0.15)
      +- Quyet dinh 4 muc: FAIL / REWORK / REVIEW / PASS
```

### 2.2. Nhanh A — Semantic Defect Detector (YOLO11n Full-Res)

**Cau hinh suy luan:**
- Model: YOLO11n (`assets/weights/yolo11n_base.pt` — ban copy tu `best.pt` goc 50 epoch / 6 lop; fine-tune them 15-20 epoch neu co thoi gian).
- Suy luan toan khung hinh: `imgsz=3072` (hoac `3584`), `conf=0.20`.
- **Khac phuc bay Square Padding:** Bat buoc truyen `rect=True, half=True` khi goi `model.predict()` de giu nguyen ty le khung hinh chu nhat (~1:2 cua bo mach PKU). Neu khong, anh bi pad thanh vuong 3584x3584 (>47% dien tich dem vo nghia), khoi Spatial Attention (`C2PSA`) tinh ma tran chu y tren toan bo feature map P5, latency vot len ~361 ms. Voi `rect=True`, latency giam xuong ~160-190 ms.
- Dau ra: tap hop du doan `B_A` voi toa do, lop (1..6), do tu tin `S_A`.

**Khac phuc loi Data Pipeline khi train/fine-tune (Ngay 2):**
- Sua `clip_bbox_to_tile` tai `scripts/prepare_yolo_dataset.py`: cat (clip) toa do hop ve bien `[0, tile_size]` thay vi tra ve `None` (ban cu lam mat nhan loi o ria tile → false negatives co y).
- Them 25% tile nen sach (hard-negative background tiles) vao tap train va fine-tune nhe 15-20 epoch de triet tieu hien tuong bao gia (False Positives) tren crop 640x640.

### 2.3. Nhanh B — Golden-Template Safety-Net (Coarse-to-Fine Registration & Diff)

**Buoc 1 — Board ID & Coarse Alignment:**
- Tien trich xuat dac trung ORB/SIFT cua 10 anh Golden Template (`assets/templates/01.JPG`, `04.JPG`, `05.JPG` … `12.JPG`) va luu cache vao file `.npz`.
- Khi co anh dau vao, khop dac trung va dung `cv2.findHomography(..., cv2.RANSAC)` de:
  - Chon ma board co so luong *inliers* cao nhat (nhan dien ma board).
  - Thu duoc ma tran bien doi `H` (Input → Template) va ma tran nghich dao `H_inv` (Template → Input).
- **Luu y quan trong:** Khong dung `cv2.findTransformECC` (ECC) lam nhanh chinh vi de ket cuc tri dia phuong tren luoi pad lap lai va khong xu ly duoc phoi canh 8 bac tu do. Neu dung ECC de refine nhe, bat buoc truyen `inputMask=None` hoac mask co kieu `CV_8UC1` de tranh loi crash cua OpenCV 4.11.0 (`_mask.type() == CV_8UC1`).

**Buoc 2 — Photometric Normalization:**
- Chuan hoa do sang va do tuong phan cuc bo (Local Gain-Bias / CLAHE) de chong lech sang va hieu ung toi goc (vignette).

**Buoc 3 — Tolerance-Aware Diff:**
- So sanh cuong do sang tung pixel anh chup voi vung lan can tau = 2-3 px cua Template.
- Dung phep `cv2.dilate` va `cv2.erode` de tao khoang bao `[T_min, T_max]` voi toc do cao (~3 ms, khong can vong lap Python).
- Thuc hien tren vung mat na dong (*Copper Mask*), ket hop tru ban do dung sai (*Residual Tolerance Map*) hieu chuan tu cac khung good.

**Buoc 4 — Connected Components & Safe Z-Score:**
- Tim cac vung lien thong co dien tich `Area >= 60 px2`.
- Tinh diem `Z_diff` cho moi blob (xem chot ky thuat 2: bat buoc `sigma_floor`).

**Buoc 5 — ROI Verification & Spatial Center Gating:**
- Chieu tam vung loi ve he toa do anh goc (xem Chot ky thuat 1).
- Cat manh `640x640` quanh tam vung diff (c_x, c_y) dua vao YOLO11n.
- Chi chap nhan box cua YOLO neu giao cat (`IoU > 0`) hoac co tam nam trong ban kinh `R = 80 px` quanh (c_x, c_y). Loai bo hoan toan cac box bao gia o ria manh crop.

### 2.4. Module Hop Nhat Khong Gian & Quyet Dinh 4 Muc

**Box Fusion (hop nhat khong gian):**
- Tren he toa do anh dau vao: neu hop du doan B_A (tu Nhanh A) va B_B (tu Nhanh B) trung vi tri (`IoU >= 0.3`), cong huong do tu tin:
  `S_final = min(1.0, S_A + 0.15)`

**Quy tac phan quyet 4 muc (loai bo hoan toan OR cung va sua loi liet Open-set):**

| Muc | Dieu kien | Ket luan |
|---|---|---|
| **1. FAIL (HIGH)** | Co loi thuoc `{open_circuit, short, missing_hole}` voi `S_final >= 0.55` | FAIL / HIGH |
| **2. REWORK (MEDIUM)** | Co loi thuoc `{mouse_bite, spur, spurious_copper}` voi `S_final >= 0.35` | REWORK / MEDIUM |
| **3a. REVIEW (Cuu Recall 6 lop)** | Co du doan 6 lop voi do tu tin thap (`0.20 <= S_final < 0.35`) NHUNG trung vung Nhanh B co `Z_diff >= 2.0` | Nang len REWORK / MEDIUM hoac REVIEW |
| **3b. REVIEW (Open-set Pure)** | YOLO hoan toan khong bat duoc (`S_final < 0.20`), NHUNG Nhanh B phat hien vung bien dang lon nam tren duong dong/pad cua Template (`Z_diff >= 3.5` VA `Area >= 100 px2`) | REVIEW / UNKNOWN_DEFECT |
| **4. PASS (GOOD)** | Cac truong hop con lai | PASS / NONE |

**Sua loi mau thuan logic (Open-set bi liet):** Khong dung `anom_z >= 2.0 AND yolo_conf_max >= 0.30` (neu YOLO bo sot, he thong van chon PASS). Tach thanh hai truong hop doc lap nhu bang tren.

---

## PHAN 3: 3 CHOT CHAN KY THUAT CHONG LOI NGAM (MICRO-TRAPS SAFEGUARDS)

### Chot 1: Dong bo He quy chieu Toa do (Coordinate Alignment)
Phep tru anh thuc hien tren khong gian Template (sau khi warp bang `H`). Khi tim duoc tam blob (c_x_tpl, c_y_tpl) trong khong gian Template, bat buoc dung ma tran nghich dao `H_inv` (`cv2.perspectiveTransform`) chieu nguoc toa do ve khong gian Anh dau vao goc **truoc khi** crop `640x640` va **truoc khi** tinh IoU voi Nhanh A. Neu khong chieu nguoc, box tu Nhanh B se khong trung vi tri voi box tu Nhanh A, Box Fusion se that bai.

### Chot 2: Chong "Ao giac Z-Score" tren anh sach (Self-Normalization Trap)
Khi tinh `Z_diff` cho tung blob, bat buoc ap dung san do lech chuan toi thieu `sigma_floor` (hieu chuan tu tap val good, khoang `4.0-6.0` muc xam):

```
Z_diff = (blob_intensity - bg_mean) / max(bg_std, sigma_floor)
```

Neu khong, anh sach co `bg_std` qua nho (~0.3) lam hat bui 2 muc xam bi phong dai len `Z_diff > 5.0`, gay bao gia lien tuc.

### Chot 3: Suy luan Batch cho ROI Crops (Latency Guardrail)
- Gioi han toi da `TOP_K = 6` hoac `8` blobs co `Z_diff` lon nhat tren moi anh.
- Gom toan bo cac manh crop `640x640` vao 1 danh sach va goi `model.predict(batch_crops, imgsz=640, half=True)` trong **1 lan goi GPU duy nhat**.
- Tuyet doi khong dung vong lap `for` goi `predict` tung manh (loi latency cascade).

---

## PHAN 4: THIET KE THUC NGHIEM & CHIEN THUAT BAO VE DO AN

### 4.1. Dinh nghia lai tap Test (01 va 04)
Khong goi 01/04 la *"Unseen Boards cho toan he thong"*. Dinh nghia chinh xac la **"Zero-defect-training Boards"** — bo mach moi dua vao day chuyen chi co duy nhat 1 anh thiet ke chuan Golden Template, chua tung co mau loi nao trong tap huan luyen.

- Nhanh A (YOLO): tong quat hoa hinh thai loi tu cac ma mach cu (`05-12`).
- Nhanh B (Template): doi chieu voi Golden Template cua ma mach moi (`01/04`).

Day la cau chuyen cong nghiep thuc te: khi day chuyen doi ma mach moi, ky su luon co ngay ban ve Gerber hoac 1 bo mach mau chuan dau tien, nhung chua co du lieu loi de train lai AI.

### 4.2. Tach bach 2 Kich ban Danh gia trong `evaluate.py` (Khong tron lan so lieu)

| Kich ban | Mo ta | Ky vong |
|---|---|---|
| **Bang 1 (Standard Benchmark)** | Danh gia tren anh PKU goc (can san tinh). | Recall de cu dat 100% (vi anh loi va template khac nhau chi o vung loi, sai so nen ~1.3 muc xam). |
| **Bang 2 (Realistic Camera Perturbation Benchmark)** | Danh gia tren anh `01/04` va anh GOOD da ap bien doi gia lap camera (dich chuyen ±25 px, xoay ±2 do, bien dang phoi canh Homography, thay doi sang cuc bo/vignette, nhieu hat). | Recall de cu Nhanh B o muc thuc te ~87-89/94, FAR thap nho Residual Tolerance Map. |

**Quy tac:** Khong duoc ghep tron con so tu Bang 1 va Bang 2 vao cung mot bang ket qua.

### 4.3. Cau tra loi cho cau hoi "Sat thu" cua Hoi dong
> *"Neu goi 01 va 04 la Unseen Boards, tai sao he thong lai co san anh Template 01.JPG va 04.JPG? Phep tru o Nhanh B co dang 'an gian' vi tru hai buc anh ky thuat so sinh ra tu cung mot file goc khong?"*

**Cau tra loi chuan:** 01/04 la *Zero-defect-training Boards*, khong phai *Unseen Boards*. Trong nha may thuc te, khi thay ma mach moi, ky su luon co Golden Sample (1 bo mach chuan) nhung chua co du lieu loi. Nhanh B dung Golden Template nay lam chuan doi chieu — do chinh la quy trinh AOI cong nghiep. Khong phai "an gian", ma la thiet ke co muc dich.

---

## PHAN 5: LO TRINH THUC THI 6 NGAY & DANH SACH FILE DU KIEN

| Ngay | Cong viec chinh | File tao moi / cap nhat | Dau vao | Dau ra | Tieu chi nghiem thu (Acceptance Criteria) |
|---|---|---|---|---|---|
| **Ngay 1** | Xay dung `evaluate.py` & do 4 dong Baseline/Ablation | `scripts/evaluate.py` (moi) | Du lieu PKU, split board-disjoint (train 05-12, val 07, test 01/04) | Bang ket qua: YOLO@640, SAHI 640, PatchCore, YOLO@3072 rect=True | Co 4 dong so lieu co ban, du de viet muc Ablation Study |
| **Ngay 2** | Sua du lieu train, fine-tune YOLO, cap nhat inference | `scripts/prepare_yolo_dataset.py` (moi)<br>`src/models/yolo_detector.py` (moi)<br>`assets/weights/yolo11n_day2.pt` (fine-tune, khong ghi de `yolo11n_base.pt`) | `assets/weights/yolo11n_base.pt`, PKU train | `prepare_yolo_dataset.py` clip bbox + 25% background tiles<br>`yolo_detector.py` imgsz=3072/3584, rect=True, half=True | FP tren anh sach giam; latency YOLO@3072 rect=True ≈ 160-190 ms |
| **Ngay 3-4** | Xay dung module Nhanh B | `src/reference/board_id_and_align.py` (moi)<br>`src/reference/tolerance_diff.py` (moi) | 10 anh Golden Template (`assets/templates/`) | ORB/SIFT descriptor cache (.npz), Homography H + H_inv, Tolerance-Aware Diff, Z_diff voi sigma_floor | Nhan dien ma board dung 100% tren 90 anh unseen (45 ms/anh)<br>Recall Nhanh B tren camera perturbation ≈ 87-89/94 |
| **Ngay 5** | Xay dung quy tac quyet dinh & hop nhat | `src/decision/logic.py` (moi)<br>`src/models/ensemble.py` (moi) | Dau ra Nhanh A + Nhanh B | Quy tac 4 muc, Box Fusion (IoU ≥ 0.3, S_final = min(1.0, S_A + 0.15)), Spatial Center Gating (R = 80 px) | Chay `evaluate.py` tren ca 2 kich ban, xuat F1/Precision/Recall/FAR thuc te |
| **Ngay 6** | Hoan thien UI & bao cao | `src/ui/app.py` (moi)<br>`aoi_final_plan/thesis_artifacts/` (bao cao) | Toan bo pipeline | Giao dien Streamlit 3 khung hinh: (1) anh + BBox YOLO, (2) diff heatmap, (3) ket luan 4 muc | Demo san sang bao ve truoc hoi dong |

---

## PHAN 6: QUY TRINH TRIEN KHAI AN TOAN (SANDBOX-FIRST WORKFLOW)

### 6.1. Nguyen tac to chuc thu muc (project moi)
```
D:/FPTU/KLTN/PCB_DualPath_AOI/
+-- aoi_final_plan/           <- Ke hoach nay (chi doc khi code)
|   +-- plan.md
|   +-- status_and_bugs.md
|   +-- rules/               <- coder / verifier / scribe rules
|   +-- thesis_artifacts/    <- Bang + figure bao cao (Scribe Agent)
+-- assets/
|   +-- weights/yolo11n_base.pt   <- Copy 1 lan tu best.pt goc (khong ghi de)
|   +-- templates/01.JPG ... 12.JPG (10 files, copy 1 lan)
+-- configs/resource_paths.py <- Khai bao duong dan READ-ONLY + ensure_pku_dataset()
+-- data/                    <- PKU_Market_PCB (Kaggle fallback) + yolo_day2 (tile moi)
+-- src/
|   +-- models/yolo_detector.py   <- Nhanh A (Ngay 2)
|   +-- reference/                <- Nhanh B (Ngay 3-4)
|   +-- decision/logic.py         <- Quy tac 4 muc (Ngay 5)
|   +-- ui/app.py                 <- Streamlit demo (Ngay 6)
+-- scripts/
|   +-- evaluate.py               <- Ngay 1 (2 kich ban tach biet)
|   +-- prepare_yolo_dataset.py   <- Ngay 2 (clip bbox + 25% bg tiles)
+-- tests/                   <- Test cua Verifier
+-- results/                 <- CSV/JSON output cua Verifier
```

### 6.2. Quy tac zero-touch (khong lam hong project cu)
- **Khong doc-ghi lan lon:** Moi duong dan ngoai project moi la STRICT READ-ONLY (xem `configs/resource_paths.py` + dieu khoan trong rules).
- **Khong xoa** file cu trong project moi sau khi da tao; khong ghi de `assets/weights/yolo11n_base.pt` (fine-tune luu file moi `yolo11n_day2.pt`).
- Moi file du lieu moi (dataset tile Ngay 2) xuat ra thu muc moi rieng (`data/yolo_day2/`), TUYET DOI KHONG ghi de vao thu muc dataset cua project cu.
- Moi module moi duoc phat trien va kiem thu **co lap truoc khi ghep vao nhanh chinh**.
- Neu bat ky ngay nao khong hoan thanh, dung lai va cho chi thi — khong day code chua kiem thu vao nhanh chinh.

### 6.3. Checkpoint giua cac ngay
- Cuoi Ngay 1: `scripts/evaluate.py` chay duoc, xuat 4 dong baseline.
- Cuoi Ngay 2: YOLO@3072 rect=True chay duoc, latency < 200 ms, FP giam so voi baseline.
- Cuoi Ngay 4: Nhanh B nhan dien ma board dung 100%, recall camera perturbation >= 87/94.
- Cuoi Ngay 5: `evaluate.py` xuat F1/Precision/Recall/FAR thuc te cho ca 2 kich ban.
- Cuoi Ngay 6: UI Streamlit demo 3 khung hinh, bao cao da co bang so lieu thuc.

---

## PHAN 7: BAN DO TAI NGUYEN THUC TE (VERIFIED RESOURCE MAP — quet 2026-10-23)

Tat ca duong dan ngoai project moi la **STRICT READ-ONLY** (chi doc de doi chieu / copy 1 chieu vao project moi). Cam ghi/xoa/sua.

### 7.1. Bang Danh muc Duong dan Thuc te

| # | Nhom tai nguyen | Duong dan tuyet doi | Trang thai |
|---|---|---|---|
| 1a | Anh goc PKU-Market-PCB (`images/` 6 lop x 15 anh) | `D:\FPTU\KLTN\PCB_Anomaly_Inspector_Portable\data\raw\pku\images\` | CO SAN (READ-ONLY) |
| 1b | Nhan VOC (`Annotations/` 6 lop) | `D:\FPTU\KLTN\PCB_Anomaly_Inspector_Portable\data\raw\pku\Annotations\` | CO SAN (READ-ONLY) |
| 1c | 10 anh Golden Template `PCB_USED` (`01,04,05-12.JPG`; khong co `02,03.JPG` trong bo du lieu goc) | `D:\FPTU\KLTN\PCB_Anomaly_Inspector_Portable\data\raw\pku\PCB_USED\` | CO SAN (READ-ONLY) → da copy vao `assets/templates/` |
| 1d | Pool unified 10k (`manifest.csv` + grids/soldef) | `D:\FPTU\KLTN\pcb_unified_10k\` | CO SAN (READ-ONLY) |
| 1e | Vi tri download PKU day du (Kaggle fallback trong `ensure_pku_dataset()`) | `D:\FPTU\KLTN\PCB_DualPath_AOI\data\PKU_Market_PCB\` | TU DONG TAI (`akhatova/pcb-defects`) khi thieu |
| 2a | Trong so YOLO11n tot nhat (50 epoch, 6 lop, 5.5 MB) | `D:\FPTU\KLTN\PCB_Anomaly_Inspector_Portable\models\yolo\best.pt` | CO SAN (READ-ONLY) → da copy thanh `assets/weights/yolo11n_base.pt` |
| 2b | Checkpoint PatchCore (baseline doi chieu) | `D:\FPTU\KLTN\PCB_Anomaly_Inspector_Portable\models\patchcore\model.ckpt` (+ `models/onnx/patchcore.onnx`) | CO SAN (READ-ONLY) |
| 3a | Tai lieu doi chieu | `D:\FPTU\KLTN\PCB_Anomaly_Inspector_Portable\docs\AUDIT_vs_TDDNet.md` | CO SAN (READ-ONLY) |
| 3b | Ket qua doi chieu (nam NGOAI project goc, cung cap KLTN) | `D:\FPTU\KLTN\hybird_test_3branch-20260920T081632Z-1-001\hybird_test_3branch\results\comparison.json` | CO SAN (READ-ONLY) |
| 3c | Script kiem nghiem dang ky anh (`temp/check20*`, `temp/check21*`) | — | [CAN USER CUNG CAP DUONG DAN] (quet khong thay trong project goc) |
| 4a | Project cu 1 (pipeline goc) | `D:\FPTU\KLTN\PCB_Anomaly_Inspector_Portable\` | CO SAN (READ-ONLY) |
| 4b | Project cu 2 — repo TDD-Net portable | `D:\FPTU\KLTN\Test_github\Tiny-Defect-Detection-for-PCB\` | CO SAN (READ-ONLY) |
| 4c | Project cu 2 — repo FPN ResNet101 (ckpt) | `D:\FPTU\KLTN\Test_github\FPN_Res101_0117_OHEM\` (+ 2 file `.zip` chua giai nen) | CO SAN (READ-ONLY) |
| 5 | Moi truong Python dung chung | `D:\FPTU\KLTN\PCB_Anomaly_Inspector_Portable\.venv\Scripts\python.exe` | CO SAN — torch 2.6.0+cu124 (CUDA True), ultralytics 8.4.182, cv2 4.11.0, numpy 1.26.4 |

### 7.2. Quy tac "Copy vs. Read-Only Reference" khi khoi tao Project moi

- **CHI COPY 1 LAN sang project moi (`assets/`):**
  - File trong so `best.pt` → doi ten thanh `assets/weights/yolo11n_base.pt` (de khi fine-tune o Ngay 2 khong ghi de mat file goc).
  - 10 anh Template `01.JPG ... 12.JPG` → `assets/templates/`.
- **TRO DUONG DAN READ-ONLY qua `configs/resource_paths.py`:**
  - Toan bo tap du lieu anh lon (PKU `images/`, `Annotations/`, `pcb_unified_10k/`), thu muc 2 project cu, va cac file ket qua doi chieu (`comparison.json`, `AUDIT_vs_TDDNet.md`).
- **KHI TAO DU LIEU MOI O NGAY 2** (sua `clip_bbox_to_tile` + them 25% tile nen): Bat buoc xuat du lieu da cat ra thu muc moi rieng cua project moi (`data/yolo_day2/`), TUYET DOI KHONG ghi de vao thu muc dataset dang dung cua project cu (cam qua `assert_writable_ok()` trong `resource_paths.py`).

---

*Ket thuc ke hoach. Moi chi tiet trien khai code se duoc thuc hien trong cac file moi tai `D:\FPTU\KLTN\PCB_DualPath_AOI\` theo lo trinh 6 ngay tren, khong cham 2 project cu.*
