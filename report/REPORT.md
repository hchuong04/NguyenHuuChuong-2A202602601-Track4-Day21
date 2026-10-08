# Báo cáo Day 6: Phát hiện vật cản cho robot/drone

> Thay **mọi** ô có chữ ĐIỀN nằm trong ngoặc vuông bằng nội dung của bạn, xoá luôn cả dấu ngoặc vuông. Lệnh `python tools/check_submission.py` sẽ báo FAIL nếu còn sót bất kỳ chỗ nào.

- **Họ tên:** Nguyễn Hữu Chương
- **MSSV:** 2A202602601
- **Lớp:** L3B
- **Link repo:** https://github.com/hchuong04/NguyenHuuChuong-2A202602601-Track4-Day21
- **Topic:** D — Phát hiện vật cản cho robot/drone
- **Dataset:** data/kitti_mini, data/synthetic
- **Các frame đã dùng:** 000011, 000021, 000049

> Hãy viết ngắn: mỗi mục từ 3 đến 8 dòng, ưu tiên số liệu và hình ảnh.

## 1. Claim

Một câu khẳng định kỹ thuật có thể kiểm chứng. Ví dụ: *"Lệch yaw 1° làm 12% điểm LiDAR rơi ra khỏi vật thể ở 30 m, phát hiện được bằng edge-alignment score với ngưỡng X."*

Trong pipeline phát hiện vật cản dựa trên hình học (Voxel Grid -> RANSAC Plane -> DBSCAN Clustering) không dùng deep learning, việc tăng ngưỡng khoảng cách mặt phẳng RANSAC (`distance_threshold`) từ 0.20m lên 0.40m khiến hơn 45% số điểm của các vật cản thấp (< 0.8m) ở cự ly gần (< 15m) bị gán nhầm thành mặt đất và loại bỏ, làm giảm hơn 40% số lượng cụm vật cản phát hiện được; đồng thời tăng `voxel_size` từ 0.05m lên 0.20m giúp giảm latency từ >100ms xuống <25ms nhưng làm mất các vật thể nhỏ/xa có dưới 10 điểm.

## 2. Evidence

Bảng số liệu sweep tham số RANSAC distance threshold (cố định `voxel_size=0.10m`, `eps=0.5m`) và Voxel size (cố định `dist_th=0.20m`, `eps=0.5m`) trên frame 000011 (KITTI). Dữ liệu chi tiết lưu tại `results/obstacle_ransac_sweep.csv` và `results/obstacle_voxel_sweep.csv`.

| Cấu hình tham số | Số điểm sau Voxel | Tỷ lệ mặt đất (%) | Số Clusters | Điểm vật thấp gần (<0.8m) | Latency p50 / p95 (ms) |
|---|---|---|---|---|---|
| RANSAC th = 0.10m | 21,930 | 37.56% | 82 | 3,063 pts | 29.8 ms / 33.2 ms |
| RANSAC th = 0.20m (chuẩn) | 21,930 | 46.00% | 61 | 2,278 pts | 26.3 ms / 29.5 ms |
| RANSAC th = 0.30m | 21,930 | 50.62% | 57 | 1,666 pts | 23.1 ms / 25.8 ms |
| RANSAC th = 0.40m | 21,930 | 53.02% | 56 | 1,372 pts | 21.2 ms / 23.4 ms |
| RANSAC th = 0.50m | 21,930 | 57.77% | 54 | 1,180 pts (-61.5%) | 19.8 ms / 21.9 ms |
| Voxel size = 0.05m | 37,495 | 45.12% | 61 | 3,120 pts | 71.6 ms / 80.3 ms |
| Voxel size = 0.10m (chuẩn) | 21,930 | 46.00% | 61 | 2,278 pts | 26.6 ms / 30.0 ms |
| Voxel size = 0.20m | 10,473 | 47.95% | 49 | 1,180 pts | 8.6 ms / 9.3 ms |
| Voxel size = 0.30m | 6,441 | 51.20% | 51 | 840 pts | 4.5 ms / 5.5 ms |

![benchmark](../results/figures/obstacle_benchmark_analysis.png)
![demo](../results/figures/obstacle_demo_frame_000011_vox0.1_dist0.2_eps0.5.png)

Nhận xét:
A. Xu hướng của `RANSAC distance_threshold` (Từ 0.10 m → 0.50 m)
Xu hướng chung
- Ngưỡng càng nới rộng → mặt đất **"nuốt" càng nhiều điểm**.
- Tỷ lệ Ground tăng đều từ **37.56% → 57.77%**.
- Trong khi đó, số điểm vật cản sát đất giảm từ **3,063 → 1,180 điểm**, tương đương mất **61.5%**.
Điểm đột ngột xấu đi
- Tại bước nhảy từ **0.20 m → 0.30 m**, số điểm vật cản thấp giảm mạnh từ **2,278 → 1,666 điểm**.
- Tương đương mất gần **30% chỉ sau một mức tăng ngưỡng**.
- Các vật thể thấp như **bánh xe, người ngồi, pallet** bắt đầu bị RANSAC coi là **mặt đường** và **xoá bỏ hoàn toàn**.

---

Xu hướng của `voxel_size` (Từ 0.05 m → 0.30 m)
- Kích thước voxel càng lớn → số điểm giảm theo hàm mũ (**37,495 → 6,441 điểm**).
- Đồng thời, độ trễ xử lý (**Latency p50**) giảm phi mã từ **71.6 ms → 4.5 ms**.
Điểm cân bằng tối ưu (Sweet spot)
- Tại `voxel_size = 0.10 m`:
  - Độ trễ là **26.6 ms**, vượt qua ngưỡng Real-time **33.3 ms / 30 Hz**.
  - Vẫn giữ được trọn vẹn **61 cụm vật cản**, bao gồm **28 cụm ở cự ly xa >25 m**.
- Khi tăng lên `voxel_size = 0.20 m`, số lượng cụm ở xa tụt mạnh xuống còn **22 cụm**.
- Nguyên nhân là mật độ điểm bị thưa, xuống dưới ngưỡng `min_points=10` của **DBSCAN**.

## 3. Failure case

![failure](../results/figures/fail_01_ground_oversegmentation.png)

- **Trường hợp:** KITTI 3D Object, frame `000011`, tăng ngưỡng RANSAC plane `distance_threshold` từ 0.15m lên 0.45m và DBSCAN `eps` từ 0.45m lên 0.80m.
- **Quan sát:** Trên ảnh camera thực tế, số điểm vật cản giảm từ 5,525 xuống 4,799 điểm (mất 726 điểm trên ảnh, trong đó mất 339 điểm bánh và gầm xe ô tô ở cự ly gần 4.1m bên trái). Toàn bộ gầm và bánh xe bị RANSAC gọt phẳng và xoá sạch vào mặt đường. Đồng thời ở góc phải, 2 người đi bộ đứng gần nhau (cách nhau ~0.8m) bị DBSCAN gộp thành 1 cụm duy nhất (từ 2 bounding box riêng biệt Ped #1, Ped #2 biến thành 1 box đỏ khổng lồ).
- **Nguyên nhân:** Mô hình RANSAC giả định toàn bộ mặt đất là một mặt phẳng đơn toàn cục ($ax+by+cz+d=0$). Khi tăng ngưỡng lên 0.45m (vượt quá độ cao gầm xe ~0.20–0.30m), toàn bộ điểm vật thể cách mặt đường < 0.45m bị gán nhầm thành inlier mặt đất. Với DBSCAN, khoảng cách giữa 2 người (~0.8m) nhỏ hơn hoặc bằng bán kính `eps = 0.80m` nên thuật toán nối thông các điểm thành một cụm duy nhất (Under-segmentation).
- **Lớp debug:** Preprocess & Geometry.
- **Cách phát hiện khi chạy thật:**
  1. Giám sát vector pháp tuyến cục bộ: Điểm có vector pháp tuyến nằm ngang ($n_z < 0.7$) không được phép gán thành mặt đất dù khoảng cách tới mặt phẳng $\le threshold$.
  2. Giám sát kích thước Bounding Box: Cảnh báo khi chiều cao ước lượng của ô tô đột ngột tụt xuống $h < 1.0\text{m}$ (báo động Over-segmentation).
  3. Cảnh báo gộp cụm: Theo dõi tỷ lệ diện tích/chiều rộng cụm người đi bộ nếu vượt quá $1.2\text{m}$ ở cự ly gần để kích hoạt thuật toán tách cụm (Sub-clustering).

## 4. Khuyến nghị nếu triển khai thật

Use-case cụ thể (ADAS / robot / drone), trade-off và bước tiếp theo.

[ĐIỀN]

## 5. Cách chạy lại

Các lệnh tái tạo lại toàn bộ kết quả từ repo sạch.

```bash
# 1. Cài đặt môi trường
pip install -r requirements.txt open3d

# 2. Chạy kiểm tra projection cơ sở (CP2)
python -m starter.projection --data-root data/synthetic --frame 000000
python -m starter.projection --data-root data/kitti_mini --frame 000011

# 3. Chạy pipeline phát hiện vật cản (Topic D - CP2)
python src/obstacle_detector.py --data-root data/kitti_mini --frame 000011 --voxel-size 0.1 --distance-threshold 0.2 --eps 0.5
python src/obstacle_detector.py --data-root data/synthetic --frame 000000 --voxel-size 0.1 --distance-threshold 0.2 --eps 0.5

# 4. Chạy benchmark sweep tham số (Topic D - CP3)
python src/benchmark_obstacle.py --data-root data/kitti_mini --frame 000011

# 5. Chạy phân tích failure case (Topic D - CP4)
python src/visualize_failure.py
```

## 6. Khai báo sử dụng AI

Ghi rõ đã dùng công cụ AI nào, dùng vào việc gì, và bạn đã tự kiểm chứng kết quả đó bằng cách nào. Nếu không dùng AI, ghi "Không sử dụng". Xem quy định ở `RULES.md` mục 2.

| Công cụ | Dùng cho việc gì | Bạn đã kiểm chứng thế nào |
|---|---|---|
| [ĐIỀN] | | |
