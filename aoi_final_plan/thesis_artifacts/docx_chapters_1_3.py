# -*- coding: utf-8 -*-
"""
Chapters 0 (Cover Page & Executive Summary), 1, 2, and 3 for the Full Thesis Documentation (.docx).
100% real empirical metrics from results/*.json and FINAL_AUDIT_AND_COMPARISON.md.
"""
from pathlib import Path
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx_style_utils import (
    FONT_MAIN, NAVY_RGB, SLATE_RGB, BODY_RGB,
    add_h1, add_h2, add_h3, add_p, add_bullet, add_callout, add_styled_table, add_figure,
)


def build_cover_and_executive_summary(doc: Document, fig_dir: Path) -> None:
    # Cover Page Title Block
    p_uni = doc.add_paragraph()
    p_uni.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_uni.paragraph_format.space_before = Pt(18)
    r_uni = p_uni.add_run("TRƯỜNG ĐẠI HỌC FPT — KHOA CÔNG NGHỆ THÔNG TIN\nBÁO CÁO KHÓA LUẬN TỐT NGHIỆP & TÀI LIỆU KỸ THUẬT TOÀN DIỆN")
    r_uni.font.name = FONT_MAIN
    r_uni.font.size = Pt(12.5)
    r_uni.font.bold = True
    r_uni.font.color.rgb = SLATE_RGB

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(14)
    p_title.paragraph_format.space_after = Pt(10)
    r_title = p_title.add_run(
        "HỆ THỐNG KIỂM TRA NGOẠI QUAN CÔNG NGHIỆP HAI NHÁNH\n"
        "(DUAL-PATH INDUSTRIAL AOI)\n"
        "PHÁT HIỆN VÀ PHÂN LOẠI LỖI TRÊN BẢNG MẠCH PCB TRẦN"
    )
    r_title.font.name = FONT_MAIN
    r_title.font.size = Pt(18.0)
    r_title.font.bold = True
    r_title.font.color.rgb = NAVY_RGB

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(14)
    r_sub = p_sub.add_run(
        "Kết hợp Nhận diện Ngữ nghĩa Toàn khung hình (YOLO11n Full-Res @3072 rect/half) và\n"
        "Lưới An Toàn Đối chiếu Mẫu chuẩn Golden-Template (Coarse-to-Fine Homography & Tolerance-Aware Diff)\n"
        "Đánh giá Độc lập Mã Mạch (Board-Disjoint) trên Zero-Defect-Training Boards (01 & 04)"
    )
    r_sub.font.name = FONT_MAIN
    r_sub.font.size = Pt(11.5)
    r_sub.font.italic = True
    r_sub.font.color.rgb = BODY_RGB

    add_callout(
        doc,
        "THÔNG TIN KIỂM TOÁN & CAM KẾT TRUNG THỰC SỐ LIỆU (DATA INTEGRITY SIGN-OFF)",
        [
            "• Thiết bị kiểm định: NVIDIA GeForce RTX 3070 (CUDA=True, PyTorch 2.6.0+cu124, Ultralytics 8.4.182, OpenCV 4.11.0).",
            "• Tập dữ liệu: PKU-Market-PCB (10 mã mạch: 01, 04, 05, 06, 07, 08, 09, 10, 11, 12; độ phân giải ~3034×1586 px).",
            "• Giao thức chia tách Board-Disjoint: Train = {05,06,08,09,10,11,12}; Val = {07}; Test = {01,04} (Zero-defect-training Boards).",
            "• Nguồn số liệu: 100% trích nguyên văn (verbatim) từ results/baseline_day1.json, results/day2_verify_summary.json, results/branchB_day3.json, results/dualpath_day5.json và FINAL_AUDIT_AND_COMPARISON.md.",
            "• Kiểm thử phần mềm tự động: 78/78 bài kiểm thử pytest đạt PASSED (100%).",
        ],
    )

    add_h1(doc, "TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)")
    add_p(
        doc,
        "Trong dây chuyền sản xuất bảng mạch in trần (Bare Printed Circuit Board — Bare PCB), hệ thống kiểm tra quang học tự động (Automated Optical Inspection — AOI) phải giải quyết đồng thời hai thách thức đối nghịch: (1) phát hiện chính xác các khuyết tật hình thái vi mô có kích thước chỉ từ 15×15 px đến 80×80 px (như pinhole, spur, mouse_bite, open_circuit, short, spurious_copper) nằm lọt thỏm trên bức ảnh toàn bo mạch độ phân giải rất lớn (~3034×1586 px); và (2) vận hành tin cậy ngay khi nhà máy chuyển đổi sang gia công một mã bo mạch hoàn toàn mới — bài toán Zero-defect-training Boards (minh họa bởi mã mạch 01 và 04), nơi kỹ sư chỉ có duy nhất một bản thiết kế chuẩn (Golden Template) nhưng chưa có bất kỳ mẫu lỗi nào của mã mạch đó để huấn luyện lại mô hình học sâu.",
    )
    add_p(
        doc,
        "Tài liệu này trình bày toàn diện quá trình nghiên cứu, kiểm toán phương pháp luận trên 2 hệ thống tiền nhiệm (Project gốc PCB_Anomaly_Inspector_Portable và các repo đối chiếu từ GitHub như TDD-Net), chỉ ra các lỗ hổng kỹ thuật cốt lõi (rò rỉ dữ liệu do chia ngẫu nhiên theo mảnh cắt, sự sụp đổ của PatchCore với FPR = 100% khi đánh giá chéo mã mạch, hiện tượng bùng nổ 92 False Positives của SAHI, và sự tê liệt của cổng hợp nhất OR cứng), từ đó đề xuất kiến trúc Hệ thống Kiểm tra Ngoại quan Công nghiệp Hai Nhánh (Dual-Path Industrial AOI).",
    )

    add_bullet(
        doc,
        "Nhánh A — Semantic Defect Detector (YOLO11n Full-Res @3072, rect=True, half=True): ",
        "Giải quyết triệt để 'Bẫy đệm vuông' (Square Padding Trap) làm lãng phí 46.88% diện tích tính toán trên khối chú ý không gian bậc hai C2PSA, kết hợp sửa lỗi cắt hộp biên (clip_bbox_to_tile cứu 35 hộp nhãn) và bổ sung 25.1% mảnh nền âm tính khó (Hard-Negative Background Tiles). Kết quả trên Kịch bản 1 (Standard): AP@0.50 đạt 0.8375 (Base) và 0.8392 (Day 2 FT), Precision tăng từ 0.7627 lên 0.7946, giảm 5 FP, triệt tiêu hoàn toàn báo giả trên ảnh GOOD 04.JPG (FAR = 0.0%), với độ trễ chỉ 77.0 ms/ảnh.",
    )
    add_bullet(
        doc,
        "Nhánh B — Golden-Template Safety-Net (ORB/RANSAC Homography + Tolerance-Aware Diff): ",
        "Sử dụng vân tay Gradient 48×96 bất biến ánh sáng kết hợp ORB-3000, RANSAC Homography (H, H^-1) và cơ chế leo thang (Escalation khi inliers < 60), đạt độ chính xác nhận diện mã mạch 98/98 = 100.0% (Benchmark) và 80/80 = 100.0% dưới nhiễu dịch chuyển ±25 px và ±50 px với độ trễ trung vị 43.1 ms. Phép trừ dung sai hình thái học (tau = 2 px) kết hợp mặt nạ trường nhìn hợp lệ (valid_mask_from_h khử 59% blob giả ngoài FOV) và tự hiệu chuẩn Median-MAD giúp đạt Centroid Recall 99/105 = 94.29% (Kịch bản 1) và 98/105 = 93.33% (Kịch bản 2).",
    )
    add_bullet(
        doc,
        "Ba chốt chặn kỹ thuật & Hợp nhất 4 mức (FAIL / REWORK / REVIEW 3a-3b / PASS): ",
        "Đảm bảo đồng bộ tọa độ qua H^-1 (Chốt 1), sàn độ lệch chuẩn nền sigma_floor = 5.0 chống bẫy tự chuẩn hóa trên ảnh sạch (Chốt 2), suy luận lô predict_batch 1 lần gọi GPU kèm cổng không gian R = 80 px (Chốt 3). Hệ thống Dual-Path đạt FAR = 0.0% tuyệt đối (0/2 GOOD) trên cả hai kịch bản, nâng AP@0.50 lớp open_circuit lên 0.9837 (+12.34% nhờ Box Fusion), bắt trúng 20/20 (100%) mẫu lỗi lạ Open-set, và giữ lại 35/36 (97.22%) bo mạch lỗi ở trạng thái cảnh báo dưới điều kiện nhiễu Camera thực tế.",
    )


def build_chapter_1(doc: Document, fig_dir: Path) -> None:
    add_h1(doc, "CHƯƠNG 1: ĐẶT VẤN ĐỀ & ĐẶC THÙ BÀI TOÁN KIỂM TRA MẠCH PCB TRẦN")

    add_h2(doc, "1.1. Đặc thù Vật lý và Quang học của Bảng mạch In Trần (Bare PCB)")
    add_p(
        doc,
        "Bảng mạch in trần (Bare PCB) trước công đoạn gắn linh kiện bề mặt (SMT) là một cấu trúc quang – cơ học có độ phức tạp hình học rất cao. Trên bộ dữ liệu chuẩn công nghiệp PKU-Market-PCB, mỗi bức ảnh chụp toàn bo mạch có độ phân giải nguyên bản khoảng 3034×1586 pixel (tương đương ~4.81 Megapixels, tỷ lệ khung hình chữ nhật ~1.91:1) hoặc lên tới 4K trong các trạm camera line-scan hiện đại. Ngược lại, các khuyết tật gia công đồng và khoan cơ khí lại có kích thước cực nhỏ, thường chỉ chiếm từ 15×15 pixel đến 50–80 pixel (tương đương 0.005% – 0.12% diện tích toàn bo mạch).",
    )
    add_p(
        doc,
        "Sáu lớp lỗi hình thái tiêu chuẩn trên mạch PCB trần được phân thành hai nhóm mức độ nghiêm trọng theo tiêu chuẩn IPC-A-600:",
    )
    add_bullet(
        doc,
        "Nhóm lỗi Nghiêm trọng cao (Critical / Electrical Failure — Mức FAIL): ",
        "Bao gồm open_circuit (hở mạch/đứt đường truyền tín hiệu), short (ngắn mạch/chập cầu đồng giữa hai đường mạch độc lập), và missing_hole (thiếu lỗ khoan xuyên tâm via/pad khiến linh kiện cắm không thể lắp ráp hoặc mất kết nối liên tầng). Bất kỳ bo mạch nào xuất hiện một trong ba lỗi này bắt buộc phải bị loại bỏ ngay lập tức (FAIL).",
    )
    add_bullet(
        doc,
        "Nhóm lỗi Tiềm ẩn / Suy giảm Độ bền (Latent / Cosmetic-Functional Defect — Mức REWORK): ",
        "Bao gồm mouse_bite (vết khuyết mẻ dạng răng chuột ở mép đường đồng làm thu hẹp tiết diện dẫn điện), spur (gai đồng nhọn nhô ra làm giảm khoảng cách cách điện an toàn), và spurious_copper (mảng đồng thừa cô lập sót lại sau quá trình ăn mòn hóa học). Các bo mạch này cần được đưa về trạm sửa chữa hoặc kiểm tra kỹ thuật (REWORK).",
    )
    add_p(
        doc,
        "Bên cạnh sự chênh lệch khổng lồ về tỷ lệ kích thước (scale mismatch), bề mặt PCB trần còn mang tính tuần hoàn mật độ cao: hàng trăm lỗ via, lưới chân IC (BGA/QFP footprints) và các bó đường mạch song song lặp đi lặp lại với khoảng cách chỉ vài pixel. Dưới ánh sáng đèn vòng công nghiệp, hiện tượng phản xạ gương trên bề mặt đồng bóng kết hợp với độ rọi không đều ở rìa thấu kính (vignetting) và sai số gá đặt cơ khí trên băng chuyền (dịch chuyển ±25 px, xoay ±2°, nghiêng phối cảnh 3D) khiến các thuật toán ngưỡng hóa cổ điển hoặc khớp mẫu cứng hoàn toàn thất bại.",
    )

    add_h2(doc, "1.2. Bài toán Thực tế trong Nhà máy — 'Zero-Defect-Training Boards' (Board 01 & 04)")
    add_p(
        doc,
        "Trong môi trường sản xuất điện tử gia công theo hợp đồng (EMS — Electronics Manufacturing Services), các dây chuyền sản xuất thường xuyên phải chuyển đổi sang gia công các mã thiết kế mạch mới (New Product Introduction — NPI). Tại thời điểm bắt đầu chạy loạt sản phẩm đầu tiên của một mã mạch mới (ví dụ mã mạch 01 và 04 trong đồ án này):",
    )
    add_bullet(
        doc,
        "Những gì nhà máy CÓ SẴN: ",
        "Nhà máy luôn luôn có bản thiết kế gốc CAD/Gerber hoặc ít nhất 01 bảng mạch mẫu chuẩn đầu tiên đã được kiểm định hoàn hảo (Golden Sample / Golden Template, tương ứng với tệp 01.JPG và 04.JPG trong thư mục assets/templates/).",
    )
    add_bullet(
        doc,
        "Những gì nhà máy CHƯA HỀ CÓ: ",
        "Nhà máy chưa từng có bất kỳ mẫu ảnh lỗi thực tế nào trên mã mạch mới 01 và 04 để huấn luyện lại (retrain) mô hình trí tuệ nhân tạo. Toàn bộ tập dữ liệu huấn luyện lịch sử chỉ bao gồm các mã mạch cũ đã sản xuất trước đó (các mã mạch 05, 06, 08, 09, 10, 11, 12 làm tập Train và mã mạch 07 làm tập Validation).",
    )

    add_callout(
        doc,
        "ĐỊNH NGHĨA HỌC THUẬT BẮT BUỘC: ZERO-DEFECT-TRAINING BOARDS (MÃ MẠCH 01 & 04)",
        [
            "• Mã mạch 01 và 04 được định nghĩa chính xác là 'Zero-defect-training Boards' (Bo mạch không có dữ liệu lỗi trong tập huấn luyện), tuyệt đối không gọi một cách mơ hồ là 'Unseen Boards cho toàn hệ thống'.",
            "• Đối với Nhánh A (YOLO11n): Thiết kế mạch 01 và 04 là hoàn toàn mới (Zero-shot cross-board layout). Mô hình chỉ được phép học đặc trưng hình thái cục bộ của 6 lớp lỗi từ các bo mạch cũ {05..12} và phải tự tổng quát hóa sang bố cục đường mạch mới của {01, 04}.",
            "• Đối với Nhánh B (Golden-Template Safety-Net): Hệ thống được phép sử dụng hợp pháp duy nhất 1 ảnh chuẩn không lỗi (01.JPG, 04.JPG) để đăng ký hình học và đối chiếu sai khác — mô phỏng đúng 100% quy trình thiết lập máy AOI trong nhà máy thực tế.",
        ],
    )

    add_h2(doc, "1.3. Nhu cầu Kết hợp Tổng quát hóa Hình thái Đóng (Closed-Set) và Bắt lỗi lạ Mở (Open-Set)")
    add_p(
        doc,
        "Một hệ thống AOI công nghiệp đạt chuẩn không chỉ dừng lại ở việc phân loại 6 lớp lỗi quen thuộc. Trong thực tế vận hành, bảng mạch luôn có nguy cơ xuất hiện các khuyết tật ngoại lai chưa từng có trong từ điển nhãn huấn luyện (Open-Set Anomalies): vết ố hóa chất ăn mòn, mảng cháy xém lớp phủ cách điện (solder mask burn), vết xước cơ khí diện rộng, dị vật bám trên bề mặt, hoặc linh kiện rơi vãi. Các bộ phát hiện vật thể thuần túy (như YOLO hay Faster R-CNN/TDD-Net) hoạt động theo cơ chế tập đóng (Closed-Set): mọi cấu trúc không khớp với hình thái của 6 lớp đã học đều bị gán nhãn là nền sạch (Background) với độ tự tin bằng 0, dẫn tới bỏ lọt lỗi nghiêm trọng (False Negative / Escape). Do đó, việc thiết kế kiến trúc hai nhánh song song — kết hợp khả năng định danh ngữ nghĩa 6 lớp của Nhánh A với lưới an toàn phát hiện mọi biến dạng vật lý so với Golden Template của Nhánh B — là yêu cầu tất yếu.",
    )


def build_chapter_2(doc: Document, fig_dir: Path) -> None:
    add_h1(doc, "CHƯƠNG 2: PHÂN TÍCH ĐỐI CHIẾU 3 HỆ THỐNG (2 PROJECT CŨ VS. PROJECT MỚI)")

    add_h2(doc, "2.1. Hệ thống 1 — Project Gốc Cũ (PCB_Anomaly_Inspector_Portable)")
    add_p(
        doc,
        "Hệ thống thế hệ đầu tiên (PCB_Anomaly_Inspector_Portable) được xây dựng theo hướng kết hợp mô hình YOLO ở độ phân giải chuẩn (640×640 hoặc cắt mảnh SAHI) với nhánh phát hiện dị thường không giám sát PatchCore (trích xuất đặc trưng tầng layer2 + layer3 của ResNet18 lưu vào Memory Bank) và hợp nhất quyết định bằng cổng logic OR cứng. Kết quả kiểm toán thực nghiệm (lưu tại results/baseline_day1.json và docs/AUDIT_vs_TDDNet.md) đã chỉ ra 4 nguyên nhân thất bại mang tính cấu trúc:",
    )
    add_bullet(
        doc,
        "1. Tại sao YOLO@640 trên ảnh toàn mạch thất bại hoàn toàn (AP50 = 0.0119 Standard / 0.0098 Camera): ",
        "Khi ép co bức ảnh toàn bo mạch kích thước 3034×1586 xuống kích thước 640×640 (tỷ lệ thu nhỏ ~4.74 lần mỗi chiều, tức giảm 22.5 lần diện tích), một vết đứt mạch hoặc lỗ kim kích thước 18×18 px bị bóp nghẹt xuống chỉ còn ~3.8×3.8 px. Khi đi qua các tầng tích chập có bước nhảy stride = 8, 16, 32 của mạng xương sống, tín hiệu đặc trưng của lỗi bị xóa sổ hoàn toàn vào nhiễu nền. Thực nghiệm ghi nhận YOLO@640 chỉ bắt trúng 3/105 lỗi ở Kịch bản 1 (Recall = 2.86%, FP = 129) và 1/105 lỗi ở Kịch bản 2 (Recall = 0.95%).",
    )
    add_bullet(
        doc,
        "2. Tại sao SAHI (Slicing Aided Hyper Inference 640×640, overlap 0.20) làm sụp đổ Precision và gây nghẽn độ trễ: ",
        "SAHI cắt ảnh lớn thành 18 mảnh 640×640 giúp giữ được Recall = 0.8095 (85/105 TP) trên Kịch bản 1, nhưng làm phát sinh tới 92 hộp báo giả (FP = 92), kéo Precision sụt giảm thảm hại xuống 0.4802 (và chỉ còn Precision = 0.4091, AP50 = 0.4882, FP = 78 trên Kịch bản 2). Khi cắt rời từng ô nhỏ, mạng mất hoàn toàn ngữ cảnh hình học toàn cục của đường mạch; các đường đồng và lỗ pad bị lưỡi cắt đi ngang qua mép ô biến thành các đoạn cụt giống hệt lỗi open_circuit hoặc mouse_bite. Hơn nữa, việc suy luận tuần tự trên 18 mảnh cắt khiến độ trễ vọt lên 605.3 ms/ảnh (Standard) và 656.4 ms/ảnh (Camera) — chậm gấp 4.1 đến 7.9 lần so với YOLO@3072 rect=True.",
    )
    add_bullet(
        doc,
        "3. Tại sao PatchCore (Unsupervised Memory-Bank) sụp đổ hoàn toàn khi đánh giá Cross-Board (AUROC = 0.3056, FPR = 100%): ",
        "Trong thiết kế cũ, Memory Bank của PatchCore được nạp đặc trưng từ các bo mạch huấn luyện (05–12). Khi đưa bo mạch kiểm thử độc lập 01 và 04 vào, cấu trúc đường đồng và bố cục chân IC hợp lệ của mã mạch 01/04 hoàn toàn khác biệt với các bo mạch trong ngân hàng nhớ. Khoảng cách láng giềng gần nhất của các vùng mạch lành trên board 01/04 vọt lên rất cao (0.7386 và 0.7643 > ngưỡng 0.6395). Ngoài ra, mã nguồn cũ còn mắc 2 lỗi toán học: gọi F.normalize(feats) làm mất biên độ kích hoạt và dùng điểm cực đại mức ảnh (patch_map.max(), p99 = 0.6425) đem so với ngưỡng phân vị mức patch (p99 = 0.5001), lệch 1.28 lần. Hệ quả là PatchCore báo giả FPR = 100% trên cả 2 ảnh GOOD 01/04 và FPR = 100% (264/264 ảnh GOOD bị phán FAIL) trên đợt kiểm toán 7 bo mạch!",
    )
    add_bullet(
        doc,
        "4. Tại sao cổng hợp nhất OR cứng (H0_OR) và bản vá AND làm tê liệt hệ thống: ",
        "Khi ghép YOLO và PatchCore bằng phép OR cứng (HIGH_SEVERITY OR is_anomaly -> FAIL), tỷ lệ báo giả toàn hệ thống bị chặn dưới bởi nhánh tệ nhất: FAR = 1 - (1 - FAR_A)(1 - FAR_B) >= FPR_PatchCore, đẩy tỷ lệ báo giả trong comparison.json lên tới FAR = 97.78% (F1 chỉ đạt 0.6716). Ngược lại, khi cố gắng chặn báo giả bằng điều kiện giao (anom_z >= 2.0 AND yolo_conf >= 0.30), hệ thống lại rơi vào 'Bẫy liệt Open-set': với mọi lỗi lạ ngoài 6 lớp mà YOLO không biết (yolo_conf = 0.0 < 0.30), vế AND luôn trả về False, khiến mọi lỗi lạ bị phán PASS!",
    )

    add_h2(doc, "2.2. Hệ thống 2 — Các Repository Đối chiếu từ GitHub (Test_github — TDD-Net & Baselines)")
    add_p(
        doc,
        "Hệ thống đối chiếu thứ hai là kho mã nguồn công bố kèm bài báo TDD-Net (Tiny Defect Detection Network, sử dụng mạng xương sống ResNet-101 kết hợp FPN tầng P2 stride 4, bộ neo nhỏ [15, 25, 40, 60, 80] và OHEM trên 10,668 ảnh cắt 600×600). Qua kiểm toán trực tiếp tệp mã nguồn chia dữ liệu (Test_github/data/io/divide_data.py, dòng 29–32), chúng tôi phát hiện lỗ hổng phương pháp luận nghiêm trọng về rò rỉ dữ liệu (Data Leakage):",
    )
    add_callout(
        doc,
        "BẰNG CHỨNG MÃ NGUỒN VỀ RÒ RỈ DỮ LIỆU TRONG REPO ĐỐI CHIẾU (Test_github/data/io/divide_data.py:29-32)",
        [
            "• Quy trình của TDD-Net: Cắt sẵn 693 ảnh gốc thuộc tất cả 10 mã bo mạch thành 10,668 mảnh nhỏ 600×600 chồng lấn nhau, sau đó gọi random.shuffle(xml_list) và chia ngẫu nhiên 80% Train (9,920 mảnh) / 20% Val (2,508 mảnh) ở cấp độ mảnh ảnh (Image-level Random Split).",
            "• Hệ quả toán học: Cùng một mã thiết kế bo mạch (bao gồm cả board 01 và 04), thậm chí các mảnh cắt chồng lấn trên cùng một bức ảnh chụp, vừa xuất hiện trong tập Train vừa xuất hiện trong tập Test!",
            "• Mạng ResNet-101 với hàng chục triệu tham số dễ dàng học thuộc lòng (memorize) hình nền đường mạch cụ thể của từng bo mạch thay vì học đặc trưng khuyết tật tổng quát. Do đó, con số mAP ~98% công bố trên tập chia ngẫu nhiên không phản ánh năng lực tổng quát hóa khi nhà máy đổi sang mã mạch mới.",
        ],
    )
    add_p(
        doc,
        "Đối chiếu trực diện với giao thức Chia tách Tuyệt đối theo Mã Bo Mạch (Board-Disjoint Split) của dự án mới: Chúng tôi khóa cứng tập Train = {05, 06, 08, 09, 10, 11, 12} (213 mảnh lỗi + 53 mảnh nền sạch = 266 mảnh), tập Val = {07} (55 mảnh lỗi + 10 mảnh nền sạch = 65 mảnh), và giữ nguyên toàn bộ ảnh gốc của mã mạch {01, 04} (36 ảnh lỗi chứa 105 hộp nhãn GT + 2 ảnh chuẩn GOOD) làm tập Kiểm thử độc lập (được kiểm증 tự động bằng bài test test_no_board_01_04_leakage_in_train_val). Ngoài ra, trên bộ kiểm thử nhanh 10 ảnh nguyên tấm, TDD-Net phát sinh báo giả có độ tự tin rất cao (0.9565 và 0.9902) trên 2/5 ảnh GOOD và hoàn toàn không có khả năng phát hiện lỗi lạ ngoài 6 lớp (Closed-set limitation).",
    )

    add_h2(doc, "2.3. Hệ thống 3 — Dự án Đề xuất (PCB_DualPath_AOI) & Bảng Đối chiếu Toàn diện 3 Hệ thống")
    add_p(
        doc,
        "Bảng 2.1 dưới đây tổng hợp đối chiếu trực diện kiến trúc, phương pháp luận và các chỉ số hiệu năng thực tế đã đo giữa 3 hệ thống:",
    )

    add_styled_table(
        doc,
        "Bảng 2.1: Tổng hợp Đối chiếu Kiến trúc, Phương pháp luận và Hiệu năng Thực nghiệm giữa 3 Hệ thống",
        [
            "Tiêu chí Đối chiếu",
            "Hệ thống 1: Project Cũ (PCB_Anomaly_Inspector_Portable)",
            "Hệ thống 2: Repo GitHub (Test_github / TDD-Net Res101)",
            "Hệ thống 3: Dự án Đề xuất (Dual-Path Industrial AOI)",
        ],
        [
            [
                "Giao thức Chia dữ liệu (Data Split Protocol)",
                "Hỗn hợp; Memory Bank PatchCore gộp đa board mâu thuẫn với Cross-board test",
                "Chia ngẫu nhiên 80/20 trên 10,668 patch 600×600 (Rò rỉ mã bo mạch — Data Leakage)",
                "Board-Disjoint tuyệt đối: Train {05..12}, Val {07}, Test Zero-defect-training {01, 04}",
            ],
            [
                "Cơ chế Xử lý Độ phân giải (Resolution Handling)",
                "Resize 640 (mất lỗi nhỏ) hoặc cắt mảnh SAHI 640 ov=0.2 (mất ngữ cảnh, nổ FP)",
                "Cắt patch tĩnh 600×600 thủ công từ trước; không có pipeline căn chỉnh toàn bo mạch",
                "Full-Res @3072 (rect=True, half=True, cắt 46.88% đệm vuông) + Batch ROI 640 quanh tâm blob",
            ],
            [
                "Khả năng Chống lệch Camera & Biến thiên Sáng",
                "Không có đăng ký hình học; sụp đổ khi dịch ±25 px, xoay ±2°, vignette",
                "Không có mô-đun đăng ký hình học hay chuẩn hóa quang độ cục bộ",
                "Vân tay Gradient 48×96 + ORB-3000 + RANSAC Homography (H, H^-1) + Low-freq Gain Norm + Valid Mask",
            ],
            [
                "Khả năng Bắt lỗi lạ (Open-Set Capability)",
                "PatchCore FPR = 100%; bản vá AND (yolo>=0.3) làm liệt hoàn toàn Open-set (0%)",
                "0% (Closed-set thuần túy 6 lớp; bỏ lọt 100% dị vật/vết ố ngoài từ điển)",
                "Rule 3b độc lập YOLO (Z_diff >= 3.5 & Area >= 100 px²): Bắt trúng 20/20 (100%) mẫu Open-set",
            ],
            [
                "AP@0.50 (Standard / Camera trên Board 01/04)",
                "YOLO640: 0.0119 / 0.0098\nSAHI640: 0.7644 / 0.4882",
                "Báo cáo ~0.98 trên tập rò rỉ patch; suy giảm mạnh và nổ FP trên ảnh nguyên tấm",
                "Nhánh A @3072: 0.8392 (Day2) / 0.6734 (Base)\nDual-Path (ROI-Gated): 0.7587 / 0.5130",
            ],
            [
                "Precision & Recall (Standard / Camera)",
                "YOLO640: P=0.0227, R=0.0286\nSAHI640: P=0.4802, R=0.8095 (Std) / P=0.4091, R=0.5143 (Cam)",
                "На 10 ảnh full-board đối chứng: bắt 5/5 lỗi nhưng báo giả 2/5 ảnh GOOD (conf 0.99)",
                "Nhánh A Day2 (Std): P=0.7946, R=0.8476\nDual-Path (Std): P=0.7928, R=0.8381 (Centroid B: 94.29% Std / 93.33% Cam)",
            ],
            [
                "F1-Score & Số hộp FP (Standard / Camera)",
                "YOLO640: F1=0.0253, FP=129\nSAHI640: F1=0.6028, FP=92 (Std) / F1=0.4557, FP=78 (Cam)\nH0_OR: F1=0.6716",
                "Không công bố F1/FP trên giao thức Board-Disjoint toàn bo mạch",
                "Nhánh A Day2 (Std): F1=0.8203, FP=23\nDual-Path (Std): F1=0.8148, FP=23\nDual-Path (Cam): F1=0.6359, FP=13",
            ],
            [
                "Tỷ lệ Báo giả trên Mạch Tốt (FAR / FPR trên GOOD)",
                "PatchCore: FPR = 100.0% (2/2 GOOD 01/04; 264/264 trên 7 boards)\nH0_OR: FAR = 97.78%",
                "40.0% trên tập thử nhanh (2/5 ảnh GOOD bị báo lỗi sai với conf > 0.95)",
                "FAR = 0.0% tuyệt đối (0/2 GOOD ở cả Standard và Camera; 0/4 mẫu GOOD kiểm định Nhánh B)",
            ],
            [
                "Độ trễ Suy luận Trung bình (Latency ms/ảnh)",
                "YOLO640: 124.8 ms\nSAHI640: 605.3 ms (Std) / 656.4 ms (Cam)\nPatchCore: 86.7 ms",
                "Faster R-CNN ResNet-101 trên ~18 patch/bo mạch: ~1200–2000 ms/bo mạch",
                "Nhánh A @3072: 77.0 ms (Day2) / 148.1 ms (Base)\nNhánh B Align: 43.1 ms | Diff: 232.2 ms\nFull Dual-Path: 379.5 ms (Std) / 333.4 ms (Cam)",
            ],
        ],
        highlight_keywords=("Dual-Path",),
    )


def build_chapter_3(doc: Document, fig_dir: Path) -> None:
    add_h1(doc, "CHƯƠNG 3: PHƯƠNG PHÁP ĐỀ XUẤT — KIẾN TRÚC DUAL-PATH INDUSTRIAL AOI")

    add_h2(doc, "3.1. Sơ đồ Khối Tổng thể và Luồng Dữ liệu Song song")
    add_p(
        doc,
        "Kiến trúc Dual-Path Industrial AOI được thiết kế gồm hai nhánh xử lý bổ trợ chặt chẽ cho nhau, hội tụ tại Mô-đun Đồng bộ Không gian qua ma trận nghịch đảo H^-1 và Khối Hợp nhất Quyết định 4 mức (4-Level Decision Logic):",
    )
    add_callout(
        doc,
        "LUỒNG DỮ LIỆU TỔNG THỂ CỦA KIẾN TRÚC DUAL-PATH INDUSTRIAL AOI",
        [
            "Ảnh đầu vào I_in (~3034×1586) ──┬──> [NHÁNH B: GOLDEN-TEMPLATE SAFETY-NET]",
            "                                │     1. Nhận diện mã mạch (Gradient Fingerprint 48×96 + ORB-3000 Shortlist + Escalation)",
            "                                │     2. Căn chỉnh RANSAC Homography (H: Input->Template, H^-1: Template->Input)",
            "                                │     3. Mặt nạ trường nhìn valid_mask_from_h (erode 11px) & fill_out_of_fov",
            "                                │     4. Chuẩn hóa quang độ tần số thấp (correct_lowfreq_gain, win=151, clip [0.5, 2.0])",
            "                                │     5. Phép trừ dung sai hình thái học (tau = 2 px) + Mặt nạ cấu trúc + Auto-Calib Median-MAD",
            "                                │     6. Phân đoạn CC (Area >= 60 px²), Safe Z-Score (sigma_floor = 5.0), lấy TOP_K = 8 blobs",
            "                                │",
            "                                └──> [CHỐT 1 & CHỐT 3: ĐỒNG BỘ TỌA ĐỘ H^-1 & NHÁNH A YOLO11n]",
            "                                      • Chế độ Toàn mạch (Full-Res): YOLO11n @3072 (rect=True, half=True, conf >= 0.20)",
            "                                      • Chế độ Đề cử ROI (ROI-Gated): Chiếu tâm blob qua H^-1 về I_in -> Cắt Batch ROI 640×640",
            "                                        -> Gọi predict_batch() 1 lần GPU duy nhất -> Cổng không gian R = 80 px",
            "                                      • Cộng hưởng Box Fusion: IoU(B_A, B_diff) >= 0.30 => S_final = min(1.0, S_A + 0.15)",
            "                                      • Phán quyết 4 mức độc lập: Level 1 FAIL -> Level 2 REWORK -> Level 3 REVIEW (3a/3b) -> Level 4 PASS",
        ],
        is_code=True,
    )

    add_h2(doc, "3.2. Nhánh A — Semantic Defect Detector (YOLO11n Full-Resolution)")
    add_h3(doc, "3.2.1. Giải quyết 'Bẫy Đệm Vuông' (Square Padding Trap) với rect=True, half=True ở @3072")
    add_p(
        doc,
        "Ảnh chụp bảng mạch PKU-Market-PCB có kích thước nguyên bản W0 × H0 ≈ 3034 × 1586 px (tỷ lệ khung hình ~1.91:1). Khi đưa vào mô hình YOLO11n ở độ phân giải cao S = 3072, nếu sử dụng cấu hình mặc định (rect=False), bộ tiền xử lý Letterbox sẽ ép ảnh vào một khung hình vuông 3072 × 3072 bằng cách đệm các dải xám vô nghĩa ở hai cạnh trên và dưới:",
    )
    add_callout(
        doc,
        "CHỨNG MINH TOÁN HỌC: TIẾT KIỆM 46.88% CHI PHÍ TÍNH TOÁN & TRÁNH BẪY CHÚ Ý BẬC HAI C2PSA",
        [
            "• Diện tích tensor khi bật đệm vuông (rect=False): A_square = 3072 × 3072 = 9,437,184 pixels.",
            "• Khi kích hoạt suy luận chữ nhật (rect=True): Chiều dài giữ ở W_rect = 3072, chiều cao co theo đúng tỷ lệ và làm tròn lên bội số của bước nhảy mạng (stride = 32):",
            "      H_rect = ceil((1586 × 3072 / 3034) / 32) × 32 = ceil(1605.87 / 32) × 32 = 51 × 32 = 1632 pixels.",
            "• Diện tích tensor thực tế: A_rect = 3072 × 1632 = 5,013,504 pixels.",
            "• Tỷ lệ diện tích tính toán thừa bị cắt bỏ hoàn toàn:",
            "      Eta_saved = 1 - (5,013,504 / 9,437,184) = 46.875% (~47%).",
            "• Tác động lên khối C2PSA (Cross-Stage Partial with Spatial Attention) ở tầng cuối mạng xương sống (stride 32): Số lượng token không gian giảm từ N_sq = 96 × 96 = 9,216 xuống N_rect = 96 × 51 = 4,896. Do ma trận tự chú ý (Self-Attention) có độ phức tạp bậc hai O(N²), số phép nhân ma trận giảm theo tỷ lệ (4896 / 9216)² = 0.2822 (tức giảm 71.78% chi phí chú ý!), kết hợp lượng tử hóa dấu phẩy động bán chính xác FP16 (half=True trên nhân Tensor Cores của RTX 3070) giúp kéo độ trễ suy luận toàn bo mạch từ 361.4 ms xuống chỉ còn 148.1 ms (Base) và 77.0 ms (Day 2 FT).",
        ],
    )

    add_h3(doc, "3.2.2. Cải tiến Đường ống Dữ liệu Ngày 2: Sửa lỗi Cắt nhãn Biên (clip_bbox_to_tile) & 25% Hard-Negative Tiles")
    add_p(
        doc,
        "Trong quá trình tạo dữ liệu huấn luyện bằng cửa sổ trượt 640×640 (scripts/prepare_yolo_dataset.py), các đường ống truyền thống thường loại bỏ mọi hộp nhãn nếu tâm của nó rơi ra ngoài ranh giới mảnh cắt, khiến các phần khuyết tật nằm sát mép mảnh cắt vẫn hiện diện trên ảnh nhưng không được gắn nhãn — vô tình dạy mạng coi lỗi ở mép là nền sạch. Chúng tôi đã thiết kế lại hàm clip_bbox_to_tile:",
    )
    add_callout(
        doc,
        "CÔNG THỨC CẮT HỘP NHÃN BIÊN (clip_bbox_to_tile) & TIÊM MẪU NỀN ÂM TÍNH KHÓ",
        [
            "• Với mỗi hộp nhãn gốc B = [x1, y1, x2, y2] và cửa sổ cắt T = [tx0, ty0, tx0 + 640, ty0 + 640]:",
            "      cx1 = clip(x1 - tx0, 0, 640),   cy1 = clip(y1 - ty0, 0, 640)",
            "      cx2 = clip(x2 - tx0, 0, 640),   cy2 = clip(y2 - ty0, 0, 640)",
            "• Điều kiện giữ lại nhãn biên hợp lệ:",
            "      w_clip = cx2 - cx1 >= 4.0 px,   h_clip = cy2 - cy1 >= 4.0 px,",
            "      (w_clip × h_clip) / ((x2 - x1) × (y2 - y1)) >= 0.25 (giữ lại ít nhất 25% diện tích gốc).",
            "• Kết quả thống kê thực tế (data/yolo_day2/stats.json): Cứu sống 35 hộp nhãn lỗi một phần ở biên (kept_partial_edge_boxes = 35).",
            "• Bổ sung 25.1% mảnh nền âm tính khó (Hard-Negative Background Tiles): Trích xuất các mảnh cắt 640×640 hoàn toàn không giao với bất kỳ hộp lỗi nào (IoU = 0) nhưng nằm trên vùng mật độ đường đồng cao từ chính các bo mạch Train/Val: bổ sung 53 mảnh nền vào tập Train (213 lỗi + 53 nền = 266 mảnh) và 10 mảnh nền vào tập Val (55 lỗi + 10 nền = 65 mảnh). Kết quả: Giảm 5 FP trên tập Test (từ 28 xuống 23 FP) và xóa sạch báo giả trên ảnh GOOD 04.JPG.",
        ],
    )

    add_h2(doc, "3.3. Nhánh B — Golden-Template Safety-Net (Coarse-to-Fine Registration & Tolerance-Aware Diff)")
    add_h3(doc, "3.3.1. Bước 1: Nhận diện Mã mạch (Board-ID) bằng Vân tay Gradient 48×96 & Căn chỉnh RANSAC Homography")
    add_p(
        doc,
        "Để đối chiếu ảnh kiểm tra I_in với mẫu chuẩn mà không cần công nhân nhập tay mã bo mạch, mô-đun src/reference/board_id_and_align.py thực hiện quy trình nhận diện kiêm căn chỉnh 2 tầng:",
    )
    add_bullet(
        doc,
        "Tầng lọc nhanh bằng Vân tay Gradient 48×96 (Gradient Magnitude Fingerprint Shortlist): ",
        "Ảnh đầu vào và 10 mẫu Golden Templates được thu nhỏ về kích thước 96×48, tính biên độ đạo hàm Sobel G = hypot(Gx, Gy) và chuẩn hóa vector đơn vị L2. Nhờ tính đạo hàm bậc nhất trước khi chuẩn hóa, vân tay này hoàn toàn bất biến trước sự thay đổi độ lợi sáng (gain alpha) và độ lệch nền (bias beta). Hệ thống chọn ra TOP_K = 2 mẫu có tích vô hướng cao nhất để đưa vào so khớp ORB.",
    )
    add_bullet(
        doc,
        "Đối sánh đặc trưng ORB-3000 + Lowe's Ratio Test (0.75) + RANSAC Homography 8 bậc tự do: ",
        "Hệ thống nạp bộ đệm đặc trưng đã tính sẵn tại assets/templates/orb_cache.npz (8,000 điểm khóa ORB mỗi template; khi chạy chế độ fast=True sẽ lấy mẫu cách đều 3,000 điểm đại diện kết hợp khôi phục đa luồng cv2.setNumThreads(CV_THREADS)). Khác với thuật toán tối ưu hóa độ dốc cục bộ ECC (cv2.findTransformECC) rất dễ bị kẹt vào cực trị địa phương trên bề mặt lưới lỗ khoan/pad lặp tuần hoàn và có nguy cơ sụp đổ trên OpenCV 4.11 nếu mặt nạ không thuộc kiểu CV_8UC1, thuật toán RANSAC Homography giải bài toán đồng thuận toàn cục với ngưỡng sai số chiếu lại 5.0 px, ước lượng trực tiếp ma trận phối cảnh 3×3 H (từ Input sang Template) và ma trận nghịch đảo H^-1 = inv(H) (từ Template về Input).",
    )
    add_bullet(
        doc,
        "Cơ chế Leo thang Tự động (Escalation Safeguard — Giải quyết BUG-DAY3-04): ",
        "Nếu mẫu đầu tiên trong danh sách rút gọn đạt >= 60 inliers (EARLY_EXIT_INLIERS = 60), thuật toán dừng sớm ngay ở ~43.1 ms. Nếu sau khi duyệt hết top-2 shortlist mà số lượng inliers cực đại vẫn < 60 (trường hợp bo mạch chịu dịch chuyển cơ khí lớn ±50 px làm trượt vân tay thu nhỏ), hệ thống tự động leo thang (escalate) quét nốt 8 mẫu chuẩn còn lại, đảm bảo độ chính xác nhận diện mã mạch đạt 100% tuyệt đối (98/98 trên benchmark và 80/80 trên stress test).",
    )
    add_bullet(
        doc,
        "Mặt nạ Trường nhìn Hợp lệ (valid_mask_from_h & fill_out_of_fov — Giải quyết BUG-DAY3-01): ",
        "Khi biến đổi phối cảnh ảnh bị dịch/xoay về hệ tọa độ Template, các vùng khuyết ở rìa ảnh được điền bằng BORDER_REPLICATE tạo ra dải sọc kéo dài gây ra 59% tổng số blob giả. Chúng tôi chiếu 4 đỉnh góc của ảnh đầu vào qua ma trận H, vẽ đa giác lồi và co biên hình thái học 11 px (BORDER_ERODE_PX = 11) để tạo mặt nạ hợp lệ M_valid, sau đó chép đè điểm ảnh của chính Golden Template vào vùng ngoài trường nhìn (thực thi bằng cv2.copyTo chỉ mất 2.2 ms, nhanh hơn 12.8 lần so với np.where).",
    )

    add_figure(
        doc,
        fig_dir / "fig1_homography_alignment.png",
        "Hình 3.1: Minh họa trực quan trước và sau khi căn chỉnh hình học RANSAC Homography kết hợp mặt nạ trường nhìn hợp lệ (valid_mask_from_h) trên Zero-defect-training Board 01 dưới nhiễu Camera (±25 px, ±2°, phối cảnh, lệch sáng). Sai số nền giảm từ 33.76 xuống 7.51 mức xám và khử sạch nhiễu rìa.",
    )

    add_h3(doc, "3.3.2. Bước 2 & 3: Chuẩn hóa Quang độ Cục bộ & Phép trừ Dung sai Hình thái học (Tolerance-Aware Diff)")
    add_p(
        doc,
        "Sau khi ảnh kiểm tra đã được đưa về cùng hệ quy chiếu với mẫu chuẩn T, mô-đun src/reference/tolerance_diff.py thực hiện chuẩn hóa quang độ và tính sai khác có dung sai:",
    )
    add_callout(
        doc,
        "CÔNG THỨC CHUẨN HÓA QUANG ĐỘ TẦN SỐ THẤP & PHÉP TRỪ DUNG SAI HÌNH THÁI HỌC (TAU = 2 PX)",
        [
            "1. Chuẩn hóa độ lợi tần số thấp (correct_lowfreq_gain, cửa sổ Gaussian W_g = 151×151):",
            "   Thay vì dùng chuẩn hóa Z-score cục bộ cửa sổ nhỏ (vốn khuếch đại nhiễu cảm biến lên 33 lần trên vùng phíp phẳng), hệ thống ước lượng trường chiếu sáng tần số thấp:",
            "      G_field(x,y) = clip( GaussianBlur(T, 151) / (GaussianBlur(I_w, 151) + 1.0), 0.5, 2.0 )",
            "      I_norm(x,y)  = clip( I_w(x,y) × G_field(x,y), 0, 255 ).",
            "",
            "2. Khoảng bao dung sai hình thái học [T_min, T_max] với phần tử cấu trúc K_tau = (2*tau + 1) × (2*tau + 1) (RUN_TAU_PX = 2, tức 5×5):",
            "      T_max(x,y) = dilate(T, K_tau),     T_min(x,y) = erode(T, K_tau)",
            "      R_raw(x,y) = max( 0, I_norm(x,y) - T_max(x,y), T_min(x,y) - I_norm(x,y) ).",
            "   Ý nghĩa vật lý: Mọi sai lệch vị trí biên đường đồng hoặc dịch chuyển nội suy sub-pixel trong bán kính tau = 2 px đều nằm trọn trong dải [T_min, T_max] và có sai số R_raw = 0!",
            "",
            "3. Mặt nạ cấu trúc đồng hợp nhất (M_union) & Khử nhiễu tự hiệu chuẩn Median-MAD:",
            "      M_struct(Z) = dilate_15x15( Otsu(Z) UNION (SobelGrad(Z) > 30.0) )",
            "      M_union     = (M_struct(T) OR M_struct(I_norm)) AND M_valid",
            "      tau_auto    = median(R_masked) + 3.0 × 1.4826 × median(|R_masked - median(R_masked)|)",
            "      R_final     = max(0, R_masked - tau_auto).",
        ],
    )

    add_figure(
        doc,
        fig_dir / "fig2_tolerance_diff_heatmap.png",
        "Hình 3.2: Đối chiếu bản đồ nhiệt sai khác tuyệt đối thông thường |I_w - T| (trái — tràn ngập nhiễu viền đồng giả do sai số đăng ký 1–2 px) và bản đồ nhiệt Tolerance-Aware Diff với tau = 2 px + Mặt nạ cấu trúc + Safe Z-Score (phải — nền phẳng bằng 0, cô lập chính xác 3 vết đứt mạch open_circuit với Z_diff = 6.05).",
    )

    add_h2(doc, "3.4. Giải quyết 3 'Cạm bẫy Kỹ thuật Ngầm' (Engineering Micro-Traps)")
    add_p(
        doc,
        "Một trong những đóng góp kỹ thuật quan trọng nhất của đồ án là việc phát hiện và giải quyết triệt để 3 cạm bẫy kỹ thuật ngầm thường khiến các hệ thống ghép nối CV + Deep Learning thất bại khi chạy thực tế:",
    )
    add_callout(
        doc,
        "BA CHỐT CHẶN KỸ THUẬT BẮT BUỘC (VERIFIED PASS TRONG FINAL_AUDIT_AND_COMPARISON.md)",
        [
            "• CHỐT KỸ THUẬT 1 — Đồng bộ Hệ tọa độ qua Ma trận Nghịch đảo H^-1 (src/models/ensemble.py:51-56 & src/reference/board_id_and_align.py:396-399):",
            "  Bản đồ sai khác R_final và các tâm vùng nghi ngờ c_tpl = (u_t, v_t) của Nhánh B nằm trong hệ quy chiếu của Golden Template, trong khi ảnh gốc I_in và hộp dự đoán của Nhánh A nằm trong hệ quy chiếu Camera chưa căn chỉnh. Nếu lấy trực tiếp tọa độ trên Template để cắt ảnh gốc hoặc tính IoU trên ảnh lệch ±25 px / xoay ±2°, hộp cắt sẽ trượt khỏi vị trí lỗi thật. Hệ thống bắt buộc chiếu ngược mọi tâm blob từ Template về không gian ảnh gốc qua H^-1:",
            "      [u_in, v_in, 1]^T ~ H^-1 × [u_t, v_t, 1]^T   (thực thi bằng cv2.perspectiveTransform, sai số khứ hồi < 10^-3 px),",
            "  và sử dụng cửa sổ cố định diff_box = [u_in - 80, v_in - 80, u_in + 80, v_in + 80] để đối sánh thay vì dùng hộp bao blob thô (giải quyết BUG-DAY3-03).",
            "",
            "• CHỐT KỸ THUẬT 2 — Chống 'Bẫy Tự Chuẩn Hóa' bằng Sàn Độ Lệch Chuẩn sigma_floor = 5.0 (src/reference/tolerance_diff.py:61, 261-277):",
            "  Trên một bảng mạch chuẩn không lỗi (GOOD board), sau phép trừ dung sai tau = 2 px, hầu hết điểm ảnh có R_final = 0 khiến độ lệch chuẩn nền tiến về sát 0 (thực nghiệm đo được sigma_bg ≈ 0.30). Nếu chia trực tiếp Z = (mu_blob - mu_bg) / sigma_bg, một hạt bụi cảm biến nhỏ có độ lệch mức xám bằng 2 sẽ bị phóng đại thành Z = 2 / 0.30 = 6.67 >> 3.5 và gây báo giả! Chúng tôi áp đặt sàn mẫu số vật lý sigma_floor = 5.0:",
            "      Z_diff(b) = (mean_res(b) - mu_bg) / max(sigma_bg, sigma_floor = 5.0, 10^-6).",
            "  Nhờ chốt này, cả 2 mẫu chuẩn GOOD 01.JPG và 04.JPG đều có max Z_diff = 0.00 (Standard) và 2.85–2.86 < 3.5 (Camera), đạt FAR = 0.0% tuyệt đối!",
            "",
            "• CHỐT KỸ THUẬT 3 — Suy luận Lô Batch ROI 1 lần gọi GPU & Cổng lọc Không gian R = 80 px (src/models/yolo_detector.py:101-169):",
            "  Thay vì gọi model.predict() rời rạc từng mảnh cắt trong vòng lặp for (gây nghẽn khởi tạo CUDA kernel), hệ thống gom tối đa TOP_K = 8 mảnh cắt 640×640 quanh các tâm blob đã chiếu qua H^-1 vào một danh sách duy nhất và gọi predict_batch() đúng 1 lần trên GPU (được khóa bởi unit test assert src.count('.predict(') == 1). Sau đó, cổng không gian spatial_center_gating(R = 80.0 px) loại bỏ mọi dự đoán nhiễu nằm cách tâm vùng sai khác quá 80 px, đồng thời dịch tọa độ hộp từ hệ mảnh cắt về hệ toàn cục (+x0, +y0) trước khi đưa vào bộ hợp nhất.",
        ],
    )

    add_h2(doc, "3.5. Cơ chế Hợp nhất Hộp (Box Fusion) và Cây Quyết định 4 Mức (4-Level Decision Logic)")
    add_p(
        doc,
        "Tại mô-đun src/models/ensemble.py và src/decision/logic.py, tập hộp dự đoán đã qua cổng không gian của Nhánh A và tập vùng sai khác của Nhánh B được hợp nhất và phân loại theo quy tắc ưu tiên nghiêm ngặt:",
    )
    add_callout(
        doc,
        "CÔNG THỨC CỘNG HƯỞNG BOX FUSION & QUY TẮC PHÁN QUYẾT 4 MỨC ĐỘC LẬP OPEN-SET",
        [
            "• Cơ chế Khớp và Cộng hưởng Độ tự tin (Box Fusion — src/models/ensemble.py:94-127):",
            "  Với mỗi vùng sai khác k của Nhánh B có cửa sổ chuẩn hóa B_diff^(k) = [u_in - 80, v_in - 80, u_in + 80, v_in + 80]:",
            "  - Nếu tồn tại dự đoán của Nhánh A có IoU(B_A, B_diff^(k)) >= 0.30 (FUSION_IOU = 0.30):",
            "        S_final = min(1.0, S_A + 0.15)   (CONF_BOOST = +0.15, đánh dấu source = 'fused').",
            "  - Nếu 0 < IoU(B_A, B_diff^(k)) < 0.30 (trường hợp lỗi rất nhỏ ~50 px nằm gọn trong cửa sổ 160×160 có IoU ~0.10–0.15):",
            "        Giữ nguyên lớp và độ tự tin S_final = S_A (đánh dấu source = 'yolo', không cộng thưởng nhưng không bị đánh mất hộp).",
            "  - Nếu vùng sai khác k không giao với bất kỳ hộp YOLO nào (IoU = 0):",
            "        Bảo toàn nguyên vẹn phần tử với source = 'branchb', lưu giữ (Z_diff, Area) để chuyển thẳng xuống xét Rule 3b (Open-set).",
            "",
            "• Cây Quyết định Công nghiệp 4 Mức (src/decision/logic.py — Ưu tiên: Level 1 > Level 2 > Level 3 > Level 4):",
            "  - MỨC 1 — FAIL (Loại bỏ bo mạch — Lỗi Nghiêm trọng Cao):",
            "        Có bất kỳ phần tử thuộc lớp {open_circuit (0), short (1), missing_hole (5)} đạt S_final >= 0.55.",
            "  - MỨC 2 — REWORK (Đưa về trạm sửa chữa — Lỗi Trung bình):",
            "        Không rơi vào Mức 1, và có phần tử thuộc lớp {mouse_bite (2), spur (3), spurious_copper (4)} đạt S_final >= 0.35,",
            "        HOẶC lớp nghiêm trọng cao {0, 1, 5} có độ tự tin trung bình 0.35 <= S_final < 0.55.",
            "  - MỨC 3 — REVIEW / UNKNOWN_DEFECT (Cảnh báo Kỹ thuật viên Kiểm tra — Gồm 2 nhánh con ĐỘC LẬP):",
            "        + Nhánh 3a (Cứu Recall cho YOLO độ tự tin thấp): 0.20 <= S_final < 0.35 VÀ có xác nhận từ Nhánh B với Z_diff >= 2.0.",
            "        + Nhánh 3b (Lưới An Toàn Bắt lỗi lạ Open-Set — ĐỘC LẬP HOÀN TOÀN VỚI YOLO):",
            "          Tồn tại vùng dị thường source == 'branchb' (YOLO không phát hiện hoặc S_A < 0.20) thỏa mãn đồng thời:",
            "              Z_diff >= 3.5   VÀ   Area >= 100 px².",
            "  - MỨC 4 — PASS (Bo mạch Đạt chuẩn):",
            "        Tất cả các trường hợp còn lại (mọi mẫu chuẩn GOOD 01.JPG và 04.JPG đều đạt Level 4 PASS).",
        ],
    )

    add_figure(
        doc,
        fig_dir / "fig3_box_fusion_recall_rescue.png",
        "Hình 3.3: Minh họa thực tế cơ chế Cứu Recall nhờ Cộng hưởng Box Fusion (+0.15 độ tự tin khi IoU >= 0.30) và Quy tắc 3a (0.20 <= S_final < 0.35 kèm Z_diff >= 2.0) trên các khuyết tật mờ có độ tự tin YOLO gốc dưới ngưỡng 0.55.",
    )

    add_figure(
        doc,
        fig_dir / "fig4_openset_safety_net.png",
        "Hình 3.4: Minh họa Lưới An Toàn Bắt lỗi lạ Open-Set (Rule 3b): Đối với mảng dị vật/vết cháy hóa chất nằm ngoài 6 lớp huấn luyện, Nhánh A (YOLO) hoàn toàn bỏ lọt (0 detections), nhưng Nhánh B lập tức cô lập vùng dị thường với Z_diff = 13.71 >= 3.5, Area = 22,500 px² >= 100 px² và kích hoạt độc lập cảnh báo Level 3 REVIEW (UNKNOWN_DEFECT).",
    )
