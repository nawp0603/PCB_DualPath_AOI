# DANH MỤC TÀI LIỆU & HÌNH ẢNH BÁO CÁO KHOA HỌC (`aoi_final_plan/thesis_artifacts/`)

**Dự án:** Đồ án tốt nghiệp (KLTN) — Hệ thống Kiểm tra Quang học Tự động Hai Nhánh (*Dual-Path Industrial AOI*) trên Mạch In Trần (*Bare PCB*)  
**Tác nhân thực hiện:** `THESIS SCRIBE & VISUALIZATION AGENT (READ-ONLY CODE)`  
**Ngày hoàn thành:** `2026-10-24`  

---

## 1. Danh mục Tài liệu Báo cáo Học thuật (`.md` & LaTeX)

| # | Tên File | Nội dung & Mục đích Sử dụng |
|---:|---|---|
| 1 | [`01_BANG_SO_LIEU_HOC_THUAT.md`](./01_BANG_SO_LIEU_HOC_THUAT.md) | Toàn bộ các **Bảng so sánh chuẩn học thuật (Markdown & mã LaTeX `booktabs`)** trích xuất chính xác 100% từ `results/`:<br>• **Bảng 1A & 1B:** Nghiên cứu loại bỏ (*Ablation Study*) chứng minh lý do loại bỏ PatchCore ($\text{AUROC}=0.3056, \text{FPR}=100\%$), SAHI ($\text{FP}=92, \text{Precision}=0.4802, 605.3\text{ ms}$) và `YOLO@640` ($\text{AP}_{50}=0.0119$).<br>• **Bảng 2:** Kết quả **Kịch bản 1 (Standard Benchmark)** trên tập **Zero-defect-training Boards `01/04`** (tách biệt hoàn toàn với Kịch bản 2).<br>• **Bảng 3:** Kết quả **Kịch bản 2 (Realistic Camera Perturbation Benchmark)** trên tập **Zero-defect-training Boards `01/04`**. <br>• **Bảng 4:** Kiểm định chuyên sâu Nhánh B (Board-ID $100\%$, Stress $\pm 25/\pm 50\text{ px}$ đạt $80/80$, Centroid Recall $94.29\%/93.33\%$, $\text{FAR}=0\%$, Open-set $20/20$ Hit Rate). |
| 2 | [`02_PHUONG_PHAP_TOAN_HOC_VA_KIEN_TRUC.md`](./02_PHUONG_PHAP_TOAN_HOC_VA_KIEN_TRUC.md) | Văn bản **Chương 3 — Cơ sở Phương pháp luận & Mô hình Toán học** bằng tiếng Việt chuẩn học thuật để đưa thẳng vào báo cáo `.docx`:<br>• Định nghĩa **Zero-defect-training Boards (`01/04`)** & Giao thức **Board-Disjoint**.<br>• Toán học của bẫy đệm vuông (*Square Padding*) trên khối chú ý bậc hai `C2PSA` và suy luận chữ nhật `rect=True, half=True`.<br>• Mô hình 5 bước của Nhánh B: Vân tay Gradient $48\times 96$ + Escalation ORB-3000 + Homography $H, H^{-1}$ + Mặt nạ ngoài FOV (`valid_mask_from_h`) + Chuẩn hóa quang trắc $151\times 151\text{ px}$ + Khoảng bao dung sai hình thái $[T_{\min}, T_{\max}]$ ($\tau=2\text{ px}$) + Tự hiệu chuẩn Median–MAD + **Safe Z-Score ($\sigma_{\text{floor}}=5.0$)**.<br>• Đồng bộ hệ quy chiếu qua $H^{-1}$ (Chốt 1), Batch ROI (Chốt 3), Cổng không gian $R=80\text{ px}$ và Hợp nhất 4 mức (tách độc lập Rule 3a / 3b). |
| 3 | [`03_BIEN_LUAN_KET_QUA_VA_SLIDE_BAO_VE.md`](./03_BIEN_LUAN_KET_QUA_VA_SLIDE_BAO_VE.md) | Văn bản **Chương 4 — Biện luận Kết quả Thực nghiệm** + **Đề cương chi tiết 12 Slide Bảo vệ** + **Bộ câu trả lời chuẩn cho 4 câu hỏi "Sát thủ" của Hội đồng** (Câu hỏi về *Zero-defect-training Boards `01/04`*, sự đánh đổi Box-level AP vs. Board-level Safety dưới nhiễu Camera, lý do không dùng ECC, và phân tích độ trễ thực thi). |

---

## 2. Danh mục Hình ảnh Minh họa Độ phân giải Cao (300 DPI) trong `figures/`

| # | Hình ảnh (300 DPI PNG) | Kích thước | Nội dung Trực quan hóa | Script Sinh ảnh التương ứng |
|---:|---|---:|---|---|
| 1 | [`figures/fig1_homography_alignment.png`](./figures/fig1_homography_alignment.png) | `2400×1420` | So sánh trước/sau đăng ký ảnh **ORB + RANSAC Homography ($H, H^{-1}$)** và khử nhiễu biên ngoài trường nhìn (`valid_mask_from_h` + `fill_out_of_fov`) trên mẫu lỗi thuộc **Zero-defect-training Board `01`** dưới nhiễu Camera. | [`gen_fig1_homography.py`](./gen_fig1_homography.py) |
| 2 | [`figures/fig2_tolerance_diff_heatmap.png`](./figures/fig2_tolerance_diff_heatmap.png) | `2400×1420` | Đối chiếu phép trừ tuyệt đối ngây thơ $|W_{\text{corr}} - T|$ với **Tolerance-Aware Diff Heatmap ($\tau=2\text{ px}$)**, Mặt nạ cấu trúc hợp nhất ($\mathcal{M}_{\text{union}}$) và **Chốt chặn Safe Z-Score ($\sigma_{\text{floor}}=5.0$)** trên ảnh lỗi & ảnh chuẩn GOOD ($\text{FAR}=0\%$). | [`gen_fig2_tolerance_heatmap.py`](./gen_fig2_tolerance_heatmap.py) |
| 3 | [`figures/fig2b_three_panel_aoi_ui.png`](./figures/fig2b_three_panel_aoi_ui.png) | `2400×1180` | Giao diện giám sát công nghiệp **3 khung hình** (Khung 1: Nhánh A YOLO11n `@3072 rect=True`; Khung 2: Bản đồ nhiệt Tolerance Diff Nhánh B; Khung 3: Phán quyết hợp nhất 4 mức). | [`gen_fig2_tolerance_heatmap.py`](./gen_fig2_tolerance_heatmap.py) |
| 4 | [`figures/fig3_box_fusion_recall_rescue.png`](./figures/fig3_box_fusion_recall_rescue.png) | `2400×1420` | Minh họa **Chốt 1 ($H^{-1}$)**, **Cổng lọc không gian $R=80\text{ px}$** (giải quyết `BUG-DAY3-03`), **Cộng hưởng độ tự tin Box Fusion ($+0.15$)** và **Quy tắc 3a cứu Recall** khi YOLO có độ tự tin thấp ($[0.20, 0.35)$) kèm ma trận phân bố 4 mức. | [`gen_fig3_box_fusion_recall.py`](./gen_fig3_box_fusion_recall.py) |
| 5 | [`figures/fig4_openset_safety_net.png`](./figures/fig4_openset_safety_net.png) | `2400×1420` | Minh họa khả năng **Bắt lỗi lạ Open-set thuần túy (Rule 3b)** trên **Zero-defect-training Board `04`** khi YOLO bỏ sót hoàn toàn ($0\text{ box}$), phân tích lỗi logic cũ (`AND yolo_conf >= 0.30`) và bảng kiểm định $20$ ca từ `probe_day3_openset_large.json`. | [`gen_fig4_openset_defect.py`](./gen_fig4_openset_defect.py) |
| 6 | [`figures/fig5_ablation_and_scenarios_chart.png`](./figures/fig5_ablation_and_scenarios_chart.png) | `2400×1460` | Biểu đồ học thuật tổng hợp **Ablation Study** (PatchCore, SAHI, `YOLO@640`, `YOLO@3072`) và đối chiếu **Kịch bản 1 (Standard)** vs. **Kịch bản 2 (Camera Perturbation)** tách biệt hoàn toàn, kèm phân rã 6 lớp lỗi. | [`gen_fig5_ablation_charts.py`](./gen_fig5_ablation_charts.py) |

---

## 3. Hướng dẫn Tái tạo Toàn bộ Hình ảnh (Reproducibility)

Để chạy lại toàn bộ 5 script sinh ảnh (chỉ đọc dữ liệu từ `results/` và `src/`, ghi duy nhất vào `aoi_final_plan/thesis_artifacts/figures/`):

```powershell
& "D:\FPTU\KLTN\PCB_Anomaly_Inspector_Portable\.venv\Scripts\python.exe" "d:\FPTU\KLTN\PCB_DualPath_AOI\aoi_final_plan\thesis_artifacts\generate_all_artifacts.py"
```
