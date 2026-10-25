# CƠ SỞ PHƯƠNG PHÁP LUẬN VÀ MÔ HÌNH TOÁN HỌC HỆ THỐNG DUAL-PATH INDUSTRIAL AOI
*(Bản thảo chuẩn học thuật dùng để đưa trực tiếp vào Chương 3 — Phương pháp Đề xuất của Báo cáo Khóa luận Tốt nghiệp `.docx`)*

---

## 3.1. Phát biểu Bài toán và Thiết lập Giao thức Kiểm thử Công nghiệp (Zero-Defect-Training Boards)

Trong dây chuyền sản xuất bảng mạch in trần (*Bare Printed Circuit Board — Bare PCB*), một hệ thống Kiểm tra Quang học Tự động (*Automated Optical Inspection — AOI*) thực tế phải giải quyết đồng thời hai yêu cầu đối lập:
1. **Định danh và phân loại ngữ nghĩa chính xác (Closed-set Semantic Detection):** Phát hiện và phân loại đúng 6 nhóm khuyết tật hình thái tiêu chuẩn trên đường đồng và lỗ khoan, bao gồm: hở mạch (`open_circuit`), ngắn mạch/chập mạch (`short`), thiếu lỗ khoan (`missing_hole`), khuyết mẻ cạnh đồng (`mouse_bite`), gai đồng nhọn (`spur`), và đồng thừa trên nền phíp (`spurious_copper`). Các lỗi này có kích thước vi mô rất nhỏ (từ $15\times 15\text{ px}$ đến $60\times 60\text{ px}$) nằm lọt thỏm trên bức ảnh toàn bo mạch độ phân giải cao ($\sim 3034 \times 1586\text{ px}$ hoặc Camera 4K).
2. **Lưới an toàn bắt lỗi lạ mở (Open-set Safety-Net) và Triệt tiêu Báo giả trên mạch sạch (Zero False-Alarm on GOOD Boards):** Khi dây chuyền gặp các khuyết tật ngoài tập huấn luyện (*Open-set anomalies* như vết ố hóa chất ăn mòn, cháy xém lớp phủ, vết xước cơ khí diện rộng, hoặc dị vật rơi vào mạch), hệ thống không được phép bỏ lọt (False Escape). Đồng thời, trên các bo mạch đạt chuẩn (*GOOD boards*), tỷ lệ báo giả (*False Alarm Rate — FAR*) phải tiến về $0\%$ để không làm tắc nghẽn dây chuyền sản xuất.

### Định nghĩa "Zero-Defect-Training Boards" và Giao thức Chia tách Theo Mã Mạch (Board-Disjoint Protocol)
Một sai lầm phổ biến trong các công trình nghiên cứu trước đây trên bộ dữ liệu PKU-Market-PCB (điển hình là TDD-Net) là thực hiện tăng cường dữ liệu và cắt mảnh chồng lấn (*overlapping crops* $600\times 600\text{ px}$), sau đó chia ngẫu nhiên tập huấn luyện và tập kiểm thử theo từng ảnh (`random.shuffle` tỷ lệ 80/20). Cách chia này gây ra hiện tượng **rò rỉ dữ liệu không gian nghiêm trọng (*Spatial Data Leakage*)**, vì các mảnh cắt của cùng một thiết kế bo mạch — và thậm chí cùng một vết lỗi — xuất hiện ở cả tập huấn luyện lẫn tập kiểm thử.

Để khắc phục triệt để hạn chế này và phản ánh đúng thực tiễn nhà máy, khóa luận thiết lập giao thức chia tách độc lập theo mã thiết kế mạch (**Board-Disjoint Split**):
- **Tập huấn luyện ($\mathcal{D}_{\text{train}}$):** Gồm 7 mã thiết kế mạch $\{05, 06, 08, 09, 10, 11, 12\}$ (43 ảnh gốc chứa 213 hộp nhãn lỗi).
- **Tập kiểm định ($\mathcal{D}_{\text{val}}$):** Gồm 1 mã thiết kế mạch $\{07\}$ (11 ảnh gốc chứa 55 hộp nhãn lỗi).
- **Tập kiểm thử độc lập ($\mathcal{D}_{\text{test}}$ — Zero-defect-training Boards $\{01, 04\}$):** Gồm 2 mã thiết kế mạch `01` và `04` (36 ảnh lỗi chứa 105 hộp nhãn thuộc 6 lớp + 2 ảnh mẫu chuẩn `01.JPG`, `04.JPG`).

Trong khóa luận này, hai mã mạch `01` và `04` được định nghĩa chuẩn xác là **"Zero-defect-training Boards"** (thay vì gọi chung chung là *"Unseen Boards cho toàn hệ thống"*). Thuật ngữ này mô tả đúng kịch bản chuyển đổi sản phẩm (*New Product Introduction — NPI*) trong nhà máy điện tử: khi dây chuyền bắt đầu chạy một mã bo mạch mới (`01` hoặc `04`), kỹ sư luôn có sẵn bản vẽ thiết kế CAD/Gerber hoặc **duy nhất 1 bo mạch mẫu chuẩn đầu tiên (*Golden Sample / Golden Template*)**, nhưng **chưa từng có bất kỳ mẫu ảnh lỗi nào** của mã mạch đó để huấn luyện lại mạng học sâu. Khi đó:
- **Nhánh A (YOLO11n Full-Resolution):** Phải tự tổng quát hóa đặc trưng hình thái lỗi học được từ các mã mạch cũ (`05–12`) sang mã mạch mới (`01/04`).
- **Nhánh B (Golden-Template Safety-Net):** Đối chiếu trực tiếp ảnh chụp từ camera với ảnh mẫu chuẩn Golden Template (`01.JPG`, `04.JPG`) để phát hiện mọi biến dạng cấu trúc.

---

## 3.2. Mô hình Toán học Nhánh A: Suy luận Toàn Khung Hình Chữ Nhật (`YOLO11n Full-Res`) và Giải quyết Bẫy Đệm Vuông (`Square Padding Trap`)

### 3.2.1. Bài toán Suy giảm Độ phân giải và Sự Bùng nổ Báo giả khi Cắt mảnh (Tiling)
Giả sử ảnh chụp toàn bo mạch đầu vào là $I \in \mathbb{R}^{H \times W \times 3}$ với kích thước điển hình $W \approx 3034\text{ px}, H \approx 1586\text{ px}$ (tỷ lệ khung hình chữ nhật $W/H \approx 1.91 : 1$). Một khuyết tật nhỏ (như gai đồng `spur` hoặc ngắn mạch `short`) có hộp giới hạn kích thước $w_d \times h_d \approx 20 \times 20\text{ px}$.
- **Nếu co ảnh trực tiếp về kích thước tiêu chuẩn $S_0 = 640\text{ px}$:** Hệ số thu nhỏ là $\alpha = 640 / 3034 \approx 0.211$. Kích thước lỗi sau khi co chỉ còn $w'_d = \alpha w_d \approx 4.2\text{ px}$. Tại tầng đặc trưng P3 (stride $s_3 = 8$) và P4 (stride $s_4 = 16$) của kiến trúc YOLO11n, toàn bộ khuyết tật bị tiêu biến vào một điểm lưới duy nhất, dẫn đến tỷ lệ phát hiện sụp đổ ($\text{AP}_{50} = 0.0119$).
- **Nếu cắt ảnh thành các ô trượt $640\times 640$ chồng lấn (SAHI — Slicing Aided Hyper Inference):** Dù giữ được độ phân giải gốc, mỗi bức ảnh $3034 \times 1586\text{ px}$ với độ chồng lấn $20\%$ (`overlap = 0.20`) bị cắt thành $N_{\text{slice}} = 6 \times 3 = 18$ mảnh độc lập. Việc mất hoàn toàn ngữ cảnh toàn cục của bảng mạch khiến các cấu trúc lành tính lặp lại (chân IC, lỗ via, ký hiệu chữ in trên mạch) và các đường mạch bị cắt ngang ở mép mảnh bị nhận nhầm thành lỗi, làm số lượng báo giả (*False Positives*) tăng vọt từ $28$ lên $92$ hộp ($\text{Precision}$ tụt từ $0.7627$ xuống $0.4802$) và độ trễ tăng lên $605.3\text{ ms/ảnh}$.

### 3.2.2. Giải quyết Bẫy Đệm Vuông (`Square Padding Trap`) trên Khối Chú ý Không gian `C2PSA`
Khi thực hiện suy luận toàn khung hình ở độ phân giải cao $S_{\text{high}} = 3072\text{ px}$ (hoặc $3584\text{ px}$), cơ chế tiền xử lý mặc định của khung huấn luyện sẽ đệm (*letterbox padding*) bức ảnh chữ nhật tỷ lệ $1:2$ thành một ma trận vuông $S_{\text{high}} \times S_{\text{high}}$. Khi đó, chiều cao thực của bo mạch sau khi giữ nguyên tỷ lệ chỉ chiếm:
$$H_{\text{valid}} = \left\lceil \frac{H}{W} \cdot S_{\text{high}} \right\rceil_{32} = \left\lceil \frac{1586}{3034} \cdot 3072 \right\rceil_{32} = 1632\text{ px}$$
Diện tích vùng đệm xám vô nghĩa (*meaningless padding area*) chiếm tới:
$$\eta_{\text{pad}} = 1 - \frac{H_{\text{valid}} \cdot S_{\text{high}}}{S_{\text{high}}^2} = 1 - \frac{1632}{3072} \approx 46.88\%$$

Đáng chú ý, kiến trúc YOLO11n tích hợp khối **Cross-Stage Partial with Spatial Attention (`C2PSA`)** ở tầng đặc trưng sâu P5 (stride $s_5 = 32$). Gọi kích thước bản đồ đặc trưng tại tầng P5 là $H_5 = H_{\text{in}}/32$ và $W_5 = W_{\text{in}}/32$, số lượng token không gian là $N = H_5 \cdot W_5$. Phép tính ma trận tự chú ý không gian (*Spatial Self-Attention*) $\text{ Softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$ có độ phức tạp tính toán và băng thông bộ nhớ tăng theo **bậc hai của số lượng token**:
$$\mathcal{O}_{\text{C2PSA}}(H_{\text{in}}, W_{\text{in}}) = \mathcal{O}(N^2 \cdot d_k) = \mathcal{O}\!\left( \left(\frac{H_{\text{in}}}{32} \cdot \frac{W_{\text{in}}}{32}\right)^2 d_k \right)$$
Khi bị đệm vuông ($H_{\text{in}} = W_{\text{in}} = S_{\text{high}}$) thay vì giữ khung chữ nhật ($H_{\text{in}} \approx 0.53\, S_{\text{high}}$), chi phí tính toán trên khối `C2PSA` tăng gấp:
$$\frac{N_{\text{square}}^2}{N_{\text{rect}}^2} = \left( \frac{1}{1 - \eta_{\text{pad}}} \right)^2 \approx \left(\frac{1}{0.5312}\right)^2 \approx 3.54 \text{ lần}$$
Bằng cách bắt buộc kích hoạt suy luận chữ nhật **`rect=True`** kết hợp lượng tử hóa nửa độ chính xác **`half=True` (FP16 Tensor Cores)** trong `src/models/yolo_detector.py`, hệ thống loại bỏ hoàn toàn $46.88\%$ vùng đệm vô nghĩa, kéo độ trễ suy luận toàn bo mạch `@3072` xuống chỉ còn **$148.1\text{ ms}$** (trọng số gốc) và **$77.0\text{ ms}$** (trọng số tinh chỉnh Ngày 2).

### 3.2.3. Cải tiến Quy trình Cắt Nhãn Biên và Bổ sung Mẫu Âm Khó (Hard-Negative Background Tiles)
Trong quá trình chuẩn bị dữ liệu huấn luyện (`scripts/prepare_yolo_dataset.py`), khóa luận khắc phục hai khuyết tật cố hữu của quy trình cũ:
1. **Cắt hộp nhãn về biên mảnh cắt (`clip_bbox_to_tile`):** Với một hộp nhãn gốc $B = [x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ và mảnh cắt $T = [x_0, y_0, x_0 + L, y_0 + L]$ ($L = 640$), thay vì trả về `None` (vứt bỏ nhãn khi một phần hộp tràn ra ngoài biên mảnh, vô tình dạy mô hình bỏ qua lỗi ở rìa), tọa độ hộp được chiếu (clip) về biên mảnh:
   $$x'_1 = \max(x_{\min}, x_0), \quad y'_1 = \max(y_{\min}, y_0), \quad x'_2 = \min(x_{\max}, x_0 + L), \quad y'_2 = \min(y_{\max}, y_0 + L)$$
   Hộp sau khi cắt được giữ lại nếu thỏa mãn đồng thời tỷ lệ diện tích phần giao và kích thước tối thiểu:
   $$\frac{(x'_2 - x'_1)(y'_2 - y'_1)}{(x_{\max} - x_{\min})(y_{\max} - y_{\min})} \ge \gamma_{\text{area}} = 0.25 \quad \text{và} \quad \min(x'_2 - x'_1,\, y'_2 - y'_1) \ge 8\text{ px}$$
   Cải tiến này đã cứu lại **$35$ hộp nhãn lỗi ở biên** trong tập huấn luyện.
2. **Tiêm $25.1\%$ mảnh nền sạch (Hard-Negative Background Tiles):** Bổ sung $53$ mảnh cắt không lỗi từ các Golden Templates vào $213$ mảnh chứa lỗi (tổng $266$ mảnh train) và $10$ mảnh nền vào $55$ mảnh val (tổng $65$ mảnh val), giúp mô hình học cách ức chế báo giả trên các cấu trúc đồng lành tính (giảm $5$ FP trên Kịch bản 1 và triệt tiêu hoàn toàn FP trên mẫu chuẩn `04.JPG`).

---

## 3.3. Mô hình Toán học Nhánh B: Lưới An Toàn Đối Chiếu Mẫu Chuẩn (Golden-Template Safety-Net)

Nhánh B (`src/reference/board_id_and_align.py` và `src/reference/tolerance_diff.py`) được thiết kế theo kiến trúc 5 bước từ thô đến tinh (*Coarse-to-Fine Registration & Morphological Tolerance Difference*), hoàn toàn không phụ thuộc vào nhãn lỗi huấn luyện.

### Bước 1: Nhận diện Mã Mạch bằng Vân tay Gradient 48×96 và Đăng ký Ảnh Homography 8 Bậc Tự Do
Cho tập $M = 10$ ảnh mẫu chuẩn Golden Templates $\{T_m\}_{m=1}^M$ (`01, 04, 05..12.JPG`). Nếu thực hiện đối sánh đặc trưng ORB vét cạn (`knnMatch`) với cả 10 mẫu chuẩn ở độ phân giải đầy đủ, thời gian xử lý lên tới $\sim 180\text{--}250\text{ ms}$. Để đạt mục tiêu thời gian thực ($\le 45\text{ ms/ảnh}$), Nhánh B xây dựng cơ chế lọc hai tầng có bảo chứng an toàn:

1. **Tầng 1 — Xếp hạng ứng viên bằng Vân tay Gradient Bất biến Quang trắc (`fingerprint` & `shortlist_boards`):**  
   Ảnh đầu vào mức xám $I \in \mathbb{R}^{H \times W}$ được hạ mẫu tuyến tính nhanh (`cv2.INTER_LINEAR`, chỉ tốn $0.11\text{ ms}$) về kích thước chuẩn $H_s \times W_s = 48 \times 96$. Biên độ đạo hàm bậc nhất Sobel theo hai phương được tính và chuẩn hóa Z-score:
   $$G(x,y) = \sqrt{\left(\nabla_x I_s(x,y)\right)^2 + \left(\nabla_y I_s(x,y)\right)^2}, \qquad \mathbf{f}(I) = \text{vec}\!\left( \frac{G - \mu_G}{\max(\sigma_G, 10^{-6})} \right) \in \mathbb{R}^{4608}$$
   Do chỉ sử dụng biên độ đạo hàm đã chuẩn hóa trung bình – phương sai, vector vân tay $\mathbf{f}(I)$ **bất biến hoàn toàn trước phép biến đổi độ sáng tuyến tính $I' = \alpha I + \beta$ và giảm thiểu ảnh hưởng của hiệu ứng tối góc (vignette)**. Độ tương đồng Cosine giữa ảnh đầu vào và mẫu chuẩn thứ $m$ là:
   $$s_{\cos}(I, T_m) = \frac{\langle \mathbf{f}(I),\, \mathbf{f}(T_m) \rangle}{\|\mathbf{f}(I)\|_2 \, \|\mathbf{f}(T_m)\|_2}$$
   Hệ thống chọn ra $K_{\text{short}} = 2$ mẫu chuẩn có $s_{\cos}$ cao nhất để đưa vào đối sánh ORB + RANSAC.

2. **Tầng 2 — Đối sánh ORB Cách đều (`FAST_TPL_MAX_DESC = 3000`), Ước lượng Homography RANSAC và Cơ chế Leo thang (`Escalation` — Khắc phục `BUG-DAY3-04`):**  
   Với mỗi mẫu chuẩn ứng viên $T_m$, tập đặc trưng nhị phân ORB 32-byte được lấy mẫu cách đều (`np.linspace`) giữ lại $3000/8000$ điểm khóa nhằm phân bố đều khắp diện tích bo mạch. Các cặp điểm khớp qua kiểm định tỷ số Lowe ($\frac{d_1}{d_2} < 0.75$) được đưa vào thuật toán RANSAC (`ransacReprojThreshold = 5.0 px`) để ước lượng ma trận biến đổi phối cảnh 8 bậc tự do $H_m \in \mathbb{R}^{3\times 3}$ (ánh xạ từ không gian Ảnh đầu vào $\to$ Không gian Template):
   $$\tilde{\mathbf{p}}_{\text{tpl}} \sim H_m \, \tilde{\mathbf{p}}_{\text{in}}, \qquad \text{với } \tilde{\mathbf{p}} = [x, y, 1]^\top$$
   **Cơ chế leo thang bảo toàn độ chính xác 100% (`BUG-DAY3-04`):** Khi ảnh đầu vào bị dịch chuyển cơ khí lớn ($\pm 25\text{ px}$ đến $\pm 50\text{ px}$) kèm xoay $\pm 2^\circ$, vân tay lưới thô $48\times 96$ có thể xếp mã mạch đúng ở vị trí thứ 3 hoặc thứ 4. Do đó, danh sách rút gọn top-2 chỉ được dùng làm **gợi ý ưu tiên (*soft heuristic*)**, không phải bộ lọc cứng. Nếu số lượng điểm nội suy (*inliers*) tốt nhất sau khi xét top-2 chưa đạt ngưỡng tin cậy thoát sớm:
   $$\max_{m \in \text{Top-2}} N_{\text{inliers}}(m) < N_{\text{early}} = 60$$
   hệ thống tự động **leo thang (*escalate*)** đối sánh tiếp các mẫu chuẩn còn lại theo thứ tự $s_{\cos}$ giảm dần và dừng ngay khi tìm được mẫu chuẩn đạt $N_{\text{inliers}} \ge 60$. Ngưỡng $N_{\text{early}} = 60$ được xác lập từ phân phối thực nghiệm trên 80 ảnh nhiễu mạnh: mã mạch đúng luôn có $\min N_{\text{inliers}} = 107$ (trung vị $176$), trong khi mã mạch sai cạnh tranh cao nhất chỉ có $\max N_{\text{inliers}} = 13$ (biên an toàn gấp $4.6\times$).

3. **Khử Nhiễu Vùng Ngoài Trường Nhìn (`valid_mask_from_h` & `fill_out_of_fov` — Khắc phục `BUG-DAY3-01`):**  
   Khi biến đổi phối cảnh ảnh đầu vào về không gian Template bằng $W = \text{warpPerspective}(I, H, (W_T, H_T))$, phần diện tích khung Template nằm ngoài trường nhìn (*Out-of-FOV*) của ảnh đầu vào bị điền bằng các pixel kéo dài ở biên (`BORDER_REPLICATE`), sinh ra sai số giả lớn (chiếm tới **59% tổng số blob giả**, tức 170/288 blobs trên Kịch bản 2). Để triệt tiêu hiện tượng này, Nhánh B chiếu 4 đỉnh của khung ảnh đầu vào qua $H$ để tạo đa giác lồi hợp lệ, co biên $e = 11\text{ px}$ để loại bỏ vùng nội suy mép, thu được mặt nạ hợp lệ $\mathcal{M}_{\text{valid}}(x,y) \in \{0, 1\}$. Toàn bộ vùng ngoài FOV được thay thế bằng chính giá trị cường độ sáng của Golden Template:
   $$W_{\text{filled}}(x,y) = \begin{cases} W(x,y), & \text{nếu } \mathcal{M}_{\text{valid}}(x,y) = 1 \\ T(x,y), & \text{nếu } \mathcal{M}_{\text{valid}}(x,y) = 0 \end{cases}$$
   Nhờ đó, phần diện tích không có bằng chứng quang học từ camera luôn có sai số bằng $0$ tuyệt đối, kéo độ lệch chuẩn nền $\sigma_{\text{bg}}$ từ $6.66$ xuống $2.52$ mức xám.

---

### Bước 2: Chuẩn hóa Quang trắc Tần số Thấp (`correct_lowfreq_gain`)
Dưới điều kiện chiếu sáng công nghiệp không đồng đều và hiệu ứng tối góc thấu kính (*vignette*), ảnh đã căn chỉnh $W(x,y)$ chịu một trường biến thiên độ sáng tần số thấp.
- **Tại sao không dùng chuẩn hóa Z-score cục bộ cửa sổ nhỏ ($31\times 31\text{ px}$)?** Nếu chuẩn hóa theo công thức $(W - \mu_{31})/\sigma_{31}$, một vết lỗi có kích thước $20\times 20\text{ px}$ nằm lọt trong cửa sổ $31\times 31\text{ px}$ sẽ làm nhiễm bẩn trực tiếp kỳ vọng cục bộ $\mu_{31}$ và phương sai $\sigma_{31}$, dẫn đến việc **tự triệt tiêu tín hiệu của chính vết lỗi đó** (thực nghiệm đo được sai số tại tâm lỗi tụt về $\approx 0$).
- **Giải pháp chia tỷ số làm trơn băng thông thấp ($151\times 151\text{ px}$):** Hệ thống ước lượng trường chiếu sáng tần số thấp bằng bộ lọc trung bình hộp kích thước lớn $K_{\text{smooth}} = 151\times 151\text{ px}$ (lớn hơn nhiều lần kích thước lỗi cục bộ) và hiệu chỉnh hệ số khuếch đại bị chặn trong đoạn $[g_{\min}, g_{\max}] = [0.5, 2.0]$:
  $$W_{\text{corr}}(x,y) = W(x,y) \cdot \text{clip}\!\left( \frac{(T * K_{151})(x,y)}{\max\big((W * K_{151})(x,y),\, 1.0\big)},\; 0.5,\; 2.0 \right)$$
  Trong đó bản làm trơn của mẫu chuẩn $(T * K_{151})$ chỉ phụ thuộc vào Template nên được tính 1 lần duy nhất và lưu vào bộ nhớ đệm (`template_priors`), tiết kiệm $15\text{ ms}$ mỗi ảnh.

---

### Bước 3: Phép Trừ Dung Sai Hình Thái Học (`Tolerance-Aware Diff`), Mặt Nạ Cấu Trúc Hợp Nhất và Tự Hiệu Chuẩn Median–MAD
Phép trừ tuyệt đối ngây thơ $|W_{\text{corr}}(x,y) - T(x,y)|$ luôn thất bại trên mạch PCB thực tế vì chỉ cần sai số đăng ký ảnh hoặc nội suy sub-pixel cỡ $1\text{--}2\text{ px}$, toàn bộ các cạnh có độ dốc mức xám lớn (viền đường đồng, viền pad hàn) sẽ phát sáng giả.

1. **Khoảng bao hình thái học $[T_{\min}, T_{\max}]$ với bán kính dung sai $\tau = 2\text{ px}$:**  
   Thay vì so sánh $W_{\text{corr}}(x,y)$ với một giá trị điểm ảnh duy nhất $T(x,y)$, hệ thống cho phép điểm ảnh khớp với bất kỳ giá trị nào trong vùng lân cận không gian $(2\tau + 1)\times(2\tau + 1)$ của Template. Bằng cách sử dụng phép co (*erosion*) và phép giãn (*dilation*) hình thái học mức xám với phần tử cấu trúc phẳng $K_{2\tau+1}$ (thực thi bằng C++ SIMD trong OpenCV chỉ tốn $\sim 3\text{ ms}$, hoàn toàn không dùng vòng lặp Python):
   $$T_{\min}(x,y) = \min_{(u,v) \in [-\tau, \tau]^2} T(x+u, y+v) = \text{erode}(T, K_{2\tau+1})$$
   $$T_{\max}(x,y) = \max_{(u,v) \in [-\tau, \tau]^2} T(x+u, y+v) = \text{dilate}(T, K_{2\tau+1})$$
   Phần dư vượt ngoài khoảng bao dung sai (*Raw Morphological Residual*) được định nghĩa là khoảng cách từ $W_{\text{corr}}(x,y)$ tới đoạn $[T_{\min}(x,y), T_{\max}(x,y)]$:
   $$R_{\text{raw}}(x,y) = \max\!\big(T_{\min}(x,y) - W_{\text{corr}}(x,y),\, 0\big) + \max\!\big(W_{\text{corr}}(x,y) - T_{\max}(x,y),\, 0\big)$$

2. **Mặt nạ cấu trúc hợp nhất ($\mathcal{M}_{\text{union}}$):**  
   Nếu chỉ giới hạn phép kiểm tra trên mặt nạ đồng nhị phân nghiêm ngặt (Otsu) của Template, các khuyết tật mọc thêm ra ngoài nền phíp cách đường đồng $1\text{--}3\text{ px}$ (như `spur`, `short`, `spurious_copper`) sẽ bị mặt nạ cắt bỏ. Do đó, khóa luận xây dựng mặt nạ cấu trúc từ cả Template lẫn ảnh đầu vào bằng hợp của phân ngưỡng Otsu và biên độ đạo hàm Sobel ($> 30$), sau đó giãn nở hình thái $d = 7\text{ px}$:
   $$\mathcal{M}_{\text{struct}}(I) = \text{dilate}_{7\times 7}\!\Big( \mathbb{I}\big(I > \theta_{\text{Otsu}}(I)\big) \;\cup\; \mathbb{I}\big(\|\nabla I\|_2 > 30\big) \Big)$$
   $$\mathcal{M}_{\text{union}}(x,y) = \big( \mathcal{M}_{\text{struct}}(T) \;\cup\; \mathcal{M}_{\text{struct}}(W_{\text{corr}}) \big) \;\cap\; \mathcal{M}_{\text{valid}}(x,y)$$

3. **Tự hiệu chuẩn Bản đồ Dung sai bằng Thống kê Bền vững Median–MAD (`AUTO_CALIB_TOL`):**  
   Để thích nghi động với mức nhiễu cảm biến của từng bức ảnh camera mà không cần bộ ảnh chuẩn hiệu chuẩn trước, hệ thống ước lượng độ lệch tuyệt đối trung vị (*Median Absolute Deviation — MAD*) của phần dư thô trên vùng mặt nạ cấu trúc $\Omega = \{(x,y) \mid \mathcal{M}_{\text{union}}(x,y) = 1\}$:
   $$m_{\text{res}} = \text{median}_{(x,y) \in \Omega} R_{\text{raw}}(x,y), \qquad \text{MAD}_{\text{res}} = \text{median}_{(x,y) \in \Omega} \big| R_{\text{raw}}(x,y) - m_{\text{res}} \big|$$
   $$\tau_{\text{tol}} = \max\!\big( m_{\text{res}} + 3.0 \cdot \text{MAD}_{\text{res}},\; 0.5 \big), \qquad D(x,y) = \frac{R_{\text{raw}}(x,y) \cdot \mathbb{I}\big((x,y) \in \Omega\big)}{\tau_{\text{tol}}}$$
   Vì trung vị (`median`) và `MAD` có điểm đổ vỡ thống kê (*breakdown point*) bằng $50\%$, các vùng lỗi cục bộ (chỉ chiếm $< 5\%$ diện tích mạch) hoàn toàn không làm lệch ngưỡng $\tau_{\text{tol}}$, trong khi nhiễu hạt toàn ảnh trên Kịch bản 2 được chuẩn hóa triệt để (giúp loại bỏ hoàn toàn báo giả trên ảnh GOOD nhiễu Camera).

---

### Bước 4: Phân tích Thành phần Liên thông và Chốt chặn Kỹ thuật 2 (`Safe Z-Score` với $\sigma_{\text{floor}}$)
Từ bản đồ sai khác chuẩn hóa $D(x,y)$, kỳ vọng nền $\mu_{\text{bg}}$ và độ lệch chuẩn nền $\sigma_{\text{bg}}$ trên toàn ảnh được tính trong một lần duyệt mảng bằng `cv2.meanStdDev(D)`. Ngưỡng hạt giống thích ứng dùng để nhị phân hóa bản đồ sai khác là:
$$\theta_{\text{seed}} = \mu_{\text{bg}} + k_{\text{seed}} \cdot \max(\sigma_{\text{bg}},\, \sigma_{\text{seed\_floor}}), \qquad \text{với } k_{\text{seed}} = 1.25,\; \sigma_{\text{seed\_floor}} = 0.5$$
Các thành phần liên thông 8-láng giềng (`cv2.connectedComponentsWithStats`) có diện tích $\text{Area}_i \ge A_{\min} = 60\text{ px}^2$ được trích xuất.

**Chốt kỹ thuật 2 — Chống "Ảo giác Z-Score" trên ảnh sạch (*Self-Normalization Trap*):**  
Nếu tính điểm bất thường của mỗi vùng liên thông thứ $i$ theo công thức Z-score tiêu chuẩn $Z_{\text{naive}} = (\bar{D}_i - \mu_{\text{bg}}) / \sigma_{\text{bg}}$, một nghịch lý toán học sẽ xảy ra trên các bức ảnh mạch chuẩn (GOOD): do ảnh GOOD khớp hoàn hảo với Template nên phần dư nền gần như bằng không ($\mu_{\text{bg}} \approx 0.05, \sigma_{\text{bg}} \approx 0.30$). Khi xuất hiện một hạt bụi cực nhỏ hoặc nhiễu lượng tử hóa có cường độ dư trung bình $\bar{D}_i = 2.1$ mức xám, phép chia cho mẫu số siêu nhỏ $\sigma_{\text{bg}} = 0.30$ sẽ phóng đại điểm số lên $Z_{\text{naive}} = (2.1 - 0.05)/0.30 = 6.83 \gg 3.5$, gây báo giả liên tục trên mạch sạch!  
Để khắc phục triệt để bẫy tự chuẩn hóa này, khóa luận áp đặt một **sàn độ lệch chuẩn vật lý tối thiểu $\sigma_{\text{floor}} = 5.0$**:
$$Z_{\text{diff}}(i) = \frac{\bar{D}_i - \mu_{\text{bg}}}{\max(\sigma_{\text{bg}},\, \sigma_{\text{floor}},\, 10^{-6})}, \qquad \text{với } \bar{D}_i = \frac{1}{\text{Area}_i} \sum_{(x,y) \in \mathcal{C}_i} D(x,y)$$
Với $\sigma_{\text{floor}} = 5.0$, hạt bụi $2.1$ mức xám trên ảnh sạch chỉ có $Z_{\text{diff}} = (2.1 - 0.05)/5.0 = 0.41 \ll 3.5$ (bị loại bỏ hoàn toàn), trong khi các khuyết tật hình thái thật ($\bar{D}_i \in [20, 120]$) vẫn đạt $Z_{\text{diff}} \in [3.8, 22.6]$. Các vùng liên thông được sắp xếp giảm dần theo $Z_{\text{diff}}$ và giới hạn tối đa $\text{TOP\_K} = 8$ vùng đứng đầu.

---

### Bước 5: Chốt kỹ thuật 1 (Chiếu ngược Hệ quy chiếu qua $H^{-1}$), Chốt kỹ thuật 3 (Suy luận Lô Batch ROI) và Cổng Không gian $R = 80\text{ px}$

1. **Chốt kỹ thuật 1 — Đồng bộ Hệ quy chiếu Không gian qua Ma trận Nghịch đảo $H^{-1}$:**  
   Toàn bộ phép trừ ở Bước 2–4 diễn ra trên hệ tọa độ của Golden Template. Tâm hình học của mỗi blob $\mathbf{c}_i^{\text{tpl}} = [c_{x,i}^{\text{tpl}}, c_{y,i}^{\text{tpl}}]^\top$ bắt buộc phải được chiếu ngược về hệ tọa độ của Ảnh đầu vào gốc thông qua ma trận nghịch đảo $H^{-1}$ (`cv2.perspectiveTransform`) **trước khi** cắt mảnh ROI `640×640` và **trước khi** tính độ giao cắt $\text{IoU}$ với Nhánh A:
   $$\mathbf{c}_i^{\text{in}} = \begin{bmatrix} c_{x,i}^{\text{in}} \\ c_{y,i}^{\text{in}} \end{bmatrix} = \Pi\!\left( H^{-1} \begin{bmatrix} c_{x,i}^{\text{tpl}} \\ c_{y,i}^{\text{tpl}} \\ 1 \end{bmatrix} \right), \qquad \text{với } \Pi([u, v, w]^\top) = \left[\frac{u}{w},\, \frac{v}{w}\right]^\top$$

2. **Chốt kỹ thuật 3 — Suy luận Batch 1 lần gọi GPU (`predict_batch`):**  
   Quanh mỗi tâm chiếu ngược $\mathbf{c}_i^{\text{in}}$ ($i = 1, \dots, K$, $K \le 8$), hệ thống cắt một mảnh ROI kích thước $640\times 640\text{ px}$ có gốc tọa độ $(x_{0,i}, y_{0,i}) = \big(\text{clip}(c_{x,i}^{\text{in}} - 320, 0, W-640),\, \text{clip}(c_{y,i}^{\text{in}} - 320, 0, H-640)\big)$. Toàn bộ $K$ mảnh cắt được gom vào một tensor duy nhất và thực thi `model.predict(batch_crops, imgsz=640, rect=True, half=True)` trong **đúng 1 lần gọi GPU**, tránh hoàn toàn hiện tượng chồng chất độ trễ (*latency cascade*) của vòng lặp `for` (độ trễ suy luận lô 3–4 mảnh sau khi làm nóng GPU chỉ tốn $18.7\text{--}69.1\text{ ms}$).

3. **Cổng lọc Tâm Không gian (`spatial_center_gating` với $R = 80\text{ px}$) và Khắc phục `BUG-DAY3-03`:**  
   Một hộp dự đoán $B_A^{\text{crop}} = [x_1, y_1, x_2, y_2]$ của YOLO trên mảnh cắt thứ $i$ trước hết phải được dịch chuyển về hệ tọa độ toàn cục của ảnh đầu vào:
   $$B_A^{\text{global}} = [x_1 + x_{0,i},\; y_1 + y_{0,i},\; x_2 + x_{0,i},\; y_2 + y_{0,i}]$$
   Gọi tâm hộp dự đoán là $\mathbf{b}_A = \big[\frac{gx_1+gx_2}{2},\, \frac{gy_1+gy_2}{2}\big]^\top$ và cửa sổ dung sai quanh tâm diff là $W_{\text{diff}}(i) = [c_{x,i}^{\text{in}} - R,\, c_{y,i}^{\text{in}} - R,\, c_{x,i}^{\text{in}} + R,\, c_{y,i}^{\text{in}} + R]$ với bán kính cổng $R = 80\text{ px}$. Hộp $B_A^{\text{global}}$ chỉ được chấp nhận nếu thỏa mãn:
   $$\big\|\mathbf{b}_A - \mathbf{c}_i^{\text{in}}\big\|_2 \le R = 80\text{ px} \quad \text{hoặc} \quad \text{IoU}\!\big(B_A^{\text{global}},\, W_{\text{diff}}(i)\big) > 0$$
   Mọi hộp báo giả phát sinh ở rìa mảnh cắt `640×640` đều bị cổng không gian này triệt tiêu hoàn toàn. Đồng thời, việc sử dụng cửa sổ $W_{\text{diff}}(i)$ kích thước $2R \times 2R = 160\times 160\text{ px}$ quanh tâm chiếu ngược giải quyết triệt để hiện tượng **`BUG-DAY3-03`** (do hộp bao chặt AABB của nhân blob sau ngưỡng hóa thường nhỏ hơn hộp nhãn chuẩn của YOLO, nếu so khớp trực tiếp AABB chặt với ngưỡng $\text{IoU} \ge 0.30$ thì chỉ đạt $13/105$ trên Kịch bản 1).

---

## 3.4. Mô hình Hợp Nhất Không Gian (`Box Fusion`) và Máy Trạng Thái Phán Quyết 4 Mức

### 3.4.1. Cộng hưởng Độ tự tin Không gian (Spatial Box Fusion)
Tại module hợp nhất (`src/models/ensemble.py`), với mỗi vùng dị thường $i$ từ Nhánh B (đại diện bởi cửa sổ cổng $W_{\text{diff}}(i)$ và cặp thuộc tính hình thái $(Z_{\text{diff}}(i), \text{Area}_i)$), hệ thống tìm hộp dự đoán chưa ghép cặp $B_A(j^*)$ từ Nhánh A có độ giao cắt $\text{IoU}\big(B_A(j), W_{\text{diff}}(i)\big)$ lớn nhất:
- **Trường hợp 1 — Đồng thuận chặt ($\text{IoU} \ge \tau_{\text{fuse}} = 0.30$, `source = "fused"`):** Cả hai nhánh cùng khẳng định sự tồn tại của khuyết tật tại cùng một vị trí vật lý. Độ tự tin cuối cùng được cộng hưởng thêm $\Delta S = +0.15$:
  $$S_{\text{final}} = \min\!\big(1.0,\; S_A(j^*) + 0.15\big)$$
- **Trường hợp 2 — Đồng thuận trong cổng ($0 < \text{IoU} < 0.30$, `source = "yolo"`):** Hộp dự đoán của YOLO rất nhỏ (ví dụ `missing_hole` hoặc `spur` cỡ $35\times 35\text{ px}$ nằm hoàn toàn bên trong cửa sổ $160\times 160\text{ px}$ nên $\text{IoU} \approx 0.05\text{--}0.20 < 0.30$). Hệ thống giữ nguyên nhãn lớp $c_{j^*}$, độ tự tin $S_{\text{final}} = S_A(j^*)$ và gắn kèm bằng chứng $(Z_{\text{diff}}(i), \text{Area}_i)$ của Nhánh B để xét duyệt ở các mức 1, 2 và 3a.
- **Trường hợp 3 — Dị thường thuần túy từ Nhánh B ($\text{IoU} = 0$, `source = "branchb"`):** Không có hộp dự đoán nào của YOLO khớp với vùng biến dạng của Nhánh B. Hệ thống tạo một mục bằng chứng độc lập mang tọa độ $W_{\text{diff}}(i)$, $S_{\text{final}} = \text{None}$, giữ nguyên $(Z_{\text{diff}}(i), \text{Area}_i)$ để chuyển xuống tầng kiểm tra lỗi lạ Open-set (Mức 3b).

### 3.4.2. Máy Trạng Thái Phán Quyết Phân Cấp 4 Mức (Khắc phục Lỗi "Liệt Open-Set")
Trong kiến trúc tiền nhiệm, điều kiện đưa vào diện kiểm tra lại (`REVIEW`) được viết dưới dạng hội logic cứng: `anom_z >= 2.0 AND yolo_conf_max >= 0.30`. Biểu thức này chứa một **mâu thuẫn logic nghiêm trọng**: khi gặp khuyết tật lạ ngoài 6 lớp huấn luyện (*Open-set*), mạng YOLO không nhận diện được nên `yolo_conf_max = 0.0 < 0.30`, làm mệnh đề `AND` luôn nhận giá trị `False` và hệ thống kết luận `PASS` (vô hiệu hóa hoàn toàn vai trò lưới an toàn của Nhánh B).

Để khắc phục triệt để, module `src/decision/logic.py` định nghĩa hai tập lớp theo mức độ nghiêm trọng cơ điện:
$$\mathcal{C}_{\text{FAIL}} = \{\text{open\_circuit (0)},\; \text{short (1)},\; \text{missing\_hole (5)}\}, \qquad \mathcal{C}_{\text{REWORK}} = \{\text{mouse\_bite (2)},\; \text{spur (3)},\; \text{spurious\_copper (4)}\}$$
và thực thi máy trạng thái 4 mức theo thứ tự ưu tiên nghiêm ngặt từ cao xuống thấp (trong đó Mức 3 được tách thành hai quy tắc độc lập **3a** và **3b**):

1. **Mức 1 — `FAIL / HIGH` (Lỗi chí mạng làm hỏng chức năng dẫn điện của mạch):**
   $$\exists\, e \in \mathcal{E} \text{ sao cho } c(e) \in \mathcal{C}_{\text{FAIL}} \;\land\; S_{\text{final}}(e) \ge 0.55$$
2. **Mức 2 — `REWORK / MEDIUM` (Lỗi hình thái mức trung bình có thể sửa chữa/gia công lại):**
   $$\exists\, e \in \mathcal{E} \text{ sao cho } c(e) \in \mathcal{C}_{\text{REWORK}} \;\land\; S_{\text{final}}(e) \ge 0.35$$
3. **Mức 3a — `REVIEW / RECALL_RESCUE` (Cứu Recall cho 6 lớp lỗi đã biết khi YOLO có độ tự tin thấp):**
   $$\exists\, e \in \mathcal{E} \text{ sao cho } c(e) \in \mathcal{C}_{\text{FAIL}} \cup \mathcal{C}_{\text{REWORK}} \;\land\; 0.20 \le S_{\text{final}}(e) < 0.35 \;\land\; Z_{\text{diff}}(e) \ge 2.0$$
4. **Mức 3b — `REVIEW / UNKNOWN_DEFECT` (Bắt lỗi lạ Open-set thuần túy — Hoàn toàn độc lập với `yolo_conf`):**
   $$\exists\, e \in \mathcal{E} \text{ sao cho } \text{source}(e) = \text{"branchb"} \;\land\; Z_{\text{diff}}(e) \ge 3.5 \;\land\; \text{Area}(e) \ge 100\text{ px}^2$$
5. **Mức 4 — `PASS / NONE` (Bo mạch đạt chuẩn):** Khi không có bất kỳ phần tử bằng chứng nào kích hoạt các mức 1, 2, 3a hoặc 3b.
