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

Bảng hoặc plot số liệu, kèm ảnh/video demo. Ghi rõ đường dẫn file trong `results/`.

| Cấu hình / mức perturb | Metric 1 | Metric 2 | Ghi chú |
|---|---|---|---|
| [ĐIỀN] | | | |

![demo](../results/figures/[ĐIỀN].png)

## 3. Failure case

Nêu khi nào hệ thống hoặc phương pháp fail, vì sao fail, và liên hệ tới lớp nào trong 6 lớp debug: I/O, Geometry, Time, Preprocess, Model, Metric.

![failure](../results/figures/fail_[ĐIỀN].png)

[ĐIỀN]

## 4. Khuyến nghị nếu triển khai thật

Use-case cụ thể (ADAS / robot / drone), trade-off và bước tiếp theo.

[ĐIỀN]

## 5. Cách chạy lại

Các lệnh tái tạo lại toàn bộ kết quả từ repo sạch.

```bash
[ĐIỀN]
```

## 6. Khai báo sử dụng AI

Ghi rõ đã dùng công cụ AI nào, dùng vào việc gì, và bạn đã tự kiểm chứng kết quả đó bằng cách nào. Nếu không dùng AI, ghi "Không sử dụng". Xem quy định ở `RULES.md` mục 2.

| Công cụ | Dùng cho việc gì | Bạn đã kiểm chứng thế nào |
|---|---|---|
| [ĐIỀN] | | |
