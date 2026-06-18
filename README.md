# INT1418-HCSDLDPT

> **Note:** Source code bài tập lớn Bird Image Search System - Hệ thống Tìm kiếm Ảnh Chim (Bird Image Search System) : [https://github.com/b22dccn256/bird-search-system](https://github.com/b22dccn256/bird-search-system)

> [!CAUTION]
> ### 🚨 LƯU Ý ĐẶC BIỆT QUAN TRỌNG KHI BÁO CÁO (THẦY HÓA)
> 
> *   **Quy tắc sống còn:** **Chỉ trình bày và nói những phần mình hiểu thật rõ.** Nếu không nắm vững bản chất, tuyệt đối không tự ý giải thích lan man (Thầy vặn hỏi sâu mà không trả lời được là sẽ bị đánh giá rất thấp $\rightarrow$ nguy cơ **OUT** trực tiếp).
> 
> ---
> 
> #### 📸 Câu 1: Dữ liệu (Data) là gì? Tách nền bằng cách nào? Minh họa trực quan?
> *   **Data của dự án:** Show thư mục `dataset/` chứa **1.239 ảnh chim** đã chuẩn hóa kích thước $224 \times 224$ pixels thuộc **122 loài**.
> *   **Tách nền thế nào:** Sử dụng thuật toán phân đoạn ảnh **GrabCut** (với bounding box tự động từ YOLOv8) kết hợp xử lý hình thái học (**Morphological Opening/Closing** bằng kernel Ellipse $5 \times 5$) để tách chim (foreground) ra khỏi nền (background). Có cơ chế tự động fallback sang phân ngưỡng **Otsu** nếu GrabCut lỗi hoặc trả về mặt nạ rỗng.
> *   **Minh họa tách nền:** 
>     *   Cách 1: Mở trực tiếp thư mục ảnh trung gian lưu ảnh debug tại `data/result_grabcut/` để thầy thấy trực quan (ảnh gốc, mặt nạ nhị phân và ảnh tách nền).
>     *   Cách 2: Chạy trực tiếp script preview bằng lệnh: `python debug_grabcut_preview.py` để biểu diễn sinh động quá trình phân đoạn ảnh chim.
> 
> ---
> 
> #### 🗄️ Câu 2: Mô hình Cơ sở dữ liệu (Database) sử dụng cấu trúc gì? Tại sao? Cách liên kết và lưu trữ vector?
> *   **Mô hình sử dụng:** Sử dụng **Kiến trúc lưu trữ lai (Hybrid Storage)** kết hợp giữa CSDL quan hệ siêu dữ liệu **SQLite** (`database.db`) và CSDL vector dạng tệp nhị phân phẳng **NumPy** (`features.npy`) kết hợp chỉ mục tìm kiếm nhanh **FAISS** (`faiss.index`).
> *   **Tại sao lại chọn mô hình Hybrid này?** 
>     *   SQLite cực kỳ nhẹ, không cần cài đặt máy chủ, rất tốt để quản lý thông tin hình ảnh và đường dẫn (`ImagePath`) tránh trùng lặp bản ghi (ràng buộc `UNIQUE`).
>     *   Tuy nhiên, các hệ quản trị CSDL quan hệ như SQL không tối ưu cho việc tính toán ma trận khoảng cách vector lớn. Việc tách ma trận đặc trưng nén PCA 512D ra file NumPy nhị phân `.npy` giúp nạp thẳng lên bộ nhớ RAM dưới dạng mảng đa chiều để thực hiện tính toán vector hóa (vectorized operations) song song ở tốc độ tối đa.
> *   **Cách liên kết và thứ tự lưu trữ:**
>     *   SQLite chỉ lưu bảng `Images` (gồm `ImageID` khóa chính tự tăng và `ImagePath` tương đối).
>     *   *Quan hệ logic:* Dòng thứ $i$ (bắt đầu từ index 0) trong ma trận `features.npy` và chỉ mục FAISS tương ứng chính xác với bản ghi thứ $i$ khi truy vấn danh sách sắp xếp theo ID: `SELECT ImageID, ImagePath FROM Images ORDER BY ImageID`.
> *   **Giải thích cách tính và thứ tự lưu một cột (Ví dụ: Dim_0):**
>     *   Vector thô lúc đầu trích xuất có **1.460 chiều** gồm: *Color (210D)*, *Texture (179D)*, *Shape (527D)*, *Spatial (544D)*.
>     *   Hệ thống dùng thuật toán nén thông tin tuyến tính **PCA** huấn luyện offline nén từ 1.460 chiều xuống còn **512 chiều** (lưu trong ma trận `.npy` từ cột `Dim_0` đến `Dim_511`).
>     *   *Cách tính:* Mỗi cột `Dim_j` (ví dụ `Dim_0`) là kết quả chiếu tuyến tính của vector 1.460 chiều ban đầu lên hướng trục phương sai lớn thứ $j$ được xác định bởi ma trận trọng số (eigenvectors) học từ tập huấn luyện. Giá trị cột này phản ánh một thành phần đặc trưng cô đọng kết hợp cả màu sắc, kết cấu và hình dạng của ảnh chim.
> 
> ---
> 
> #### 💻 Câu 3: Luồng chạy của code (Upload $\rightarrow$ Trả kết quả) & Công thức so khớp Cosine
> *   **Show code luồng chạy (Vừa chỉ code vừa giải thích):**
>     1.  Mở `2_app.py`: Chỉ vào phần Streamlit `st.file_uploader` nhận tệp upload.
>     2.  Hệ thống ghi tệp ảnh tạm thời ra đĩa và truyền đường dẫn vào hàm trích xuất đặc trưng `extract_raw_features` trong tệp `feature_extractors.py`.
>     3.  Sau khi nhận được vector đặc trưng thô 1.460 chiều $\rightarrow$ đi qua hàm giảm chiều `pca.transform(raw)` để đưa về vector 512 chiều.
>     4.  Truyền vector 512 chiều này vào hàm `search_topk` so khớp với ma trận CSDL.
> *   **Cách tính công thức Cosine Similarity:**
>     *   Khoảng cách Cosine giữa vector truy vấn $\mathbf{q}$ và vector ảnh CSDL $\mathbf{x}_i$:
>         \[
>         d_i = 1 - \cos(\theta) = 1 - \frac{\mathbf{q} \cdot \mathbf{x}_i}{\|\mathbf{q}\|_2 \|\mathbf{x}_i\|_2}
>         \]
>     *   Độ tương đồng hiển thị (%) trên web app: $\text{similarity}_i = (1 - d_i) \times 100\%$.
> *   **Hiểu rõ thông số thiết lập (Settings):**
>     *   `use_faiss`: Hệ thống tự kiểm tra thư viện FAISS cài đặt trên máy. Nếu có FAISS (`faiss.index`), nó chuẩn hóa L2 các vector đặc trưng và thực hiện so khớp exact cosine tốc độ cao qua lớp `IndexFlatIP` (Tích vô hướng).
>     *   *Fallback:* Nếu môi trường máy chạy thiếu FAISS, code tự động dùng hàm brute-force `cdist(..., metric="cosine")` của SciPy.
>     *   `top_k`: Tham số slider trên sidebar Streamlit cho phép người dùng điều chỉnh số kết quả hiển thị mong muốn (mặc định là Top-5).

---

## Tài liệu liên quan

*   **Giáo trình chính:** `Giáo trình - 2011.pdf`
*   **Slide bài giảng:** Thư mục `Slide/` chứa các slide lý thuyết về Cơ sở dữ liệu Đa phương tiện.
*   **Sách tham khảo:** Thư mục `Sách tham khảo/` chứa các sách nghiên cứu chuyên sâu về quản lý và khai phá dữ liệu đa phương tiện.
*   **Tài liệu chuẩn bị báo cáo:** Thư mục `review_question/` chứa các câu hỏi, hình ảnh và video dùng để trình bày báo cáo bảo vệ.
*   **Báo cáo kết quả của nhóm:** `file_support/baocao_nhom11_hecsdldpt.txt`
*   **Báo cáo đánh giá tổng hợp:** `results/quantitative_summary.md`