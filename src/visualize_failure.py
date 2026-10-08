"""Tạo ảnh minh họa Failure Case cực kỳ trực quan cho Topic D (Robot/Drone Obstacle Detection).

Áp dụng đúng chuẩn hướng dẫn:
  - Đặt ảnh ĐÚNG (Baseline) và ảnh SAI (Failure) cạnh nhau bằng np.hstack / np.vstack.
  - Chiếu trực tiếp lên ảnh Camera thật.
  - Khoanh vùng khoanh tròn (cv2.circle) và hình chữ nhật (cv2.rectangle) màu đỏ/vàng làm nổi bật vị trí lỗi.
  - Ghi chú tiếng Anh / không dấu rõ ràng bằng cv2.putText:
      + "FAIL: -339 WHEEL PTS ERASED AS ROAD"
      + "FAIL: 2 PEDS MERGED INTO 1 BOX"
  - Có khung hình phóng to (Zoom-in) chi tiết xe ô tô và 2 người đi bộ.

Lưu kết quả tại: results/figures/fail_01_ground_oversegmentation.png
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import open3d as o3d

from starter.datasets import load_frame
from starter.projection import project_velo_to_image
from src.obstacle_detector import filter_roi, downsample_voxel, segment_ground_plane, cluster_obstacles


def create_failure_comparison_image(data_root: str = "data/kitti_mini",
                                    frame_id: str = "000011",
                                    out_path: Path = ROOT / "results" / "figures" / "fail_01_ground_oversegmentation.png") -> None:
    frame_data = load_frame(data_root, frame_id)
    raw_img = frame_data["image"]  # shape (375, 1242, 3)
    calib = frame_data["calib"]

    roi_points = filter_roi(frame_data["points"], x_range=(0, 50), y_range=(-20, 20), z_range=(-2.5, 2.0))
    pcd, down_pts = downsample_voxel(roi_points, voxel_size=0.10)

    # 1. Pipeline Baseline (ĐÚNG): th=0.15m, eps=0.45m
    _, in_base, _, obs_base, obs_pcd_base = segment_ground_plane(pcd, distance_threshold=0.15, num_iterations=100)
    labels_base, cl_base = cluster_obstacles(obs_pcd_base, eps=0.45, min_points=8)

    # 2. Pipeline Failure (SAI): th=0.45m, eps=0.80m
    _, in_fail, _, obs_fail, obs_pcd_fail = segment_ground_plane(pcd, distance_threshold=0.45, num_iterations=100)
    labels_fail, cl_fail = cluster_obstacles(obs_pcd_fail, eps=0.80, min_points=8)

    # Các điểm bị RANSAC "nuốt" thành mặt đất
    lost_indices = list(set(in_fail) - set(in_base))
    lost_pts = down_pts[lost_indices]

    # Chiếu điểm lên ảnh
    uv_base, _, _ = project_velo_to_image(obs_base, calib, raw_img.shape)
    uv_fail, _, _ = project_velo_to_image(obs_fail, calib, raw_img.shape)
    uv_lost, _, _ = project_velo_to_image(lost_pts, calib, raw_img.shape)

    # ==========================
    # ẢNH ĐÚNG (img_ok)
    # ==========================
    img_ok = raw_img.copy()
    # Vẽ điểm vật cản chuẩn (Xanh lá sáng)
    for u, v in uv_base.astype(int):
        cv2.circle(img_ok, (u, v), 2, (0, 255, 0), -1)

    # Vẽ box xe ô tô gần
    cv2.rectangle(img_ok, (0, 217), (86, 374), (0, 255, 0), 2)
    cv2.rectangle(img_ok, (0, 192), (180, 216), (0, 0, 0), -1)
    cv2.putText(img_ok, "Car (Wheels intact)", (5, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Vẽ 2 box người đi bộ riêng biệt
    cv2.rectangle(img_ok, (873, 145), (905, 258), (0, 255, 0), 2)
    cv2.putText(img_ok, "Ped #1", (860, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.rectangle(img_ok, (908, 148), (938, 258), (0, 255, 255), 2)
    cv2.putText(img_ok, "Ped #2", (915, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # Banner tiêu đề ảnh đúng
    cv2.rectangle(img_ok, (10, 10), (540, 50), (0, 0, 0), -1)
    cv2.putText(img_ok, "[NORMAL] BASELINE (th=0.15m, eps=0.45m)", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ==========================
    # ẢNH SAI (img_fail)
    # ==========================
    img_fail = raw_img.copy()
    # Vẽ điểm vật cản còn sót (Xanh da trời)
    for u, v in uv_fail.astype(int):
        cv2.circle(img_fail, (u, v), 2, (255, 200, 0), -1)

    # Vẽ các điểm bị gọt mất (Màu đỏ rực rỡ)
    for u, v in uv_lost.astype(int):
        cv2.circle(img_fail, (u, v), 2, (0, 0, 255), -1)

    # KHOANH VÙNG LỖI 1: Bánh xe và gầm xe bị xoá mất
    cv2.circle(img_fail, (45, 330), 45, (0, 0, 255), 3)
    cv2.rectangle(img_fail, (0, 345), (290, 373), (0, 0, 0), -1)
    cv2.putText(img_fail, "FAIL: WHEELS ERASED (-339 pts)", (5, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # KHOANH VÙNG LỖI 2: Hai người đi bộ bị dính thành 1 box duy nhất
    cv2.rectangle(img_fail, (868, 142), (942, 260), (0, 0, 255), 3)
    cv2.rectangle(img_fail, (740, 110), (1040, 138), (0, 0, 0), -1)
    cv2.putText(img_fail, "FAIL: 2 PEDS MERGED (eps=0.8m)", (745, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # Banner tiêu đề ảnh lỗi
    cv2.rectangle(img_fail, (10, 10), (620, 50), (0, 0, 0), -1)
    cv2.putText(img_fail, "[FAIL] OVER-SEGMENTED ROAD (th=0.45m, eps=0.8m)", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ==========================
    # CẮT ZOOM-IN CHI TIẾT TRÊN ẢNH GỐC ĐỂ KHÔNG BỊ TRÙNG CHỮ
    # ==========================
    # Zoom xe ô tô (H: 190->375, W: 0->150)
    crop_base_car = raw_img[190:375, 0:150].copy()
    crop_fail_car = raw_img[190:375, 0:150].copy()

    # Vẽ điểm lên crop xe
    for u, v in uv_base.astype(int):
        if 0 <= u < 150 and 190 <= v < 375:
            cv2.circle(crop_base_car, (u, v - 190), 3, (0, 255, 0), -1)

    for u, v in uv_fail.astype(int):
        if 0 <= u < 150 and 190 <= v < 375:
            cv2.circle(crop_fail_car, (u, v - 190), 3, (255, 200, 0), -1)

    for u, v in uv_lost.astype(int):
        if 0 <= u < 150 and 190 <= v < 375:
            cv2.circle(crop_fail_car, (u, v - 190), 4, (0, 0, 255), -1)

    # Khoanh vùng trên crop xe
    cv2.rectangle(crop_base_car, (2, 27), (86, 184), (0, 255, 0), 2)
    cv2.circle(crop_fail_car, (45, 140), 40, (0, 0, 255), 3)

    zoom_car_ok = cv2.resize(crop_base_car, (350, 260))
    zoom_car_fail = cv2.resize(crop_fail_car, (350, 260))

    cv2.rectangle(zoom_car_ok, (0, 0), (350, 35), (0, 0, 0), -1)
    cv2.putText(zoom_car_ok, "1. CAR BASELINE (WHEELS OK)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.rectangle(zoom_car_fail, (0, 0), (350, 35), (0, 0, 0), -1)
    cv2.putText(zoom_car_fail, "2. CAR FAIL (WHEELS LOST -339P)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)

    # Zoom người đi bộ (H: 130->270, W: 860->950)
    crop_base_ped = raw_img[130:270, 860:950].copy()
    crop_fail_ped = raw_img[130:270, 860:950].copy()

    for u, v in uv_base.astype(int):
        if 860 <= u < 950 and 130 <= v < 270:
            cv2.circle(crop_base_ped, (u - 860, v - 130), 3, (0, 255, 0), -1)

    for u, v in uv_fail.astype(int):
        if 860 <= u < 950 and 130 <= v < 270:
            cv2.circle(crop_fail_ped, (u - 860, v - 130), 3, (255, 200, 0), -1)

    for u, v in uv_lost.astype(int):
        if 860 <= u < 950 and 130 <= v < 270:
            cv2.circle(crop_fail_ped, (u - 860, v - 130), 4, (0, 0, 255), -1)

    # Vẽ box trên crop người
    cv2.rectangle(crop_base_ped, (13, 15), (45, 128), (0, 255, 0), 2)
    cv2.rectangle(crop_base_ped, (48, 18), (78, 128), (0, 255, 255), 2)
    cv2.rectangle(crop_fail_ped, (8, 12), (82, 130), (0, 0, 255), 3)

    zoom_ped_ok = cv2.resize(crop_base_ped, (350, 260))
    zoom_ped_fail = cv2.resize(crop_fail_ped, (350, 260))

    cv2.rectangle(zoom_ped_ok, (0, 0), (350, 35), (0, 0, 0), -1)
    cv2.putText(zoom_ped_ok, "3. PEDS BASELINE (2 BOXES)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.rectangle(zoom_ped_fail, (0, 0), (350, 35), (0, 0, 0), -1)
    cv2.putText(zoom_ped_fail, "4. PEDS FAIL (1 MERGED BOX)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Ghép hàng dưới (Zoom-in bar): 4 ảnh zoom ghép ngang
    zoom_row = np.hstack([zoom_car_ok, zoom_car_fail, zoom_ped_ok, zoom_ped_fail])  # width: 1400, height: 260

    # Ghép hàng trên (Full Camera views): resize mỗi ảnh về width=700 để tổng width=1400
    top_ok = cv2.resize(img_ok, (700, 215))
    top_fail = cv2.resize(img_fail, (700, 215))
    top_row = np.hstack([top_ok, top_fail])  # width: 1400, height: 215

    # Tạo thanh phân cách chú thích ở giữa
    banner_mid = np.zeros((35, 1400, 3), dtype=np.uint8)
    banner_mid[:] = (20, 20, 20)
    cv2.putText(banner_mid, "CHI TIET PHONG TO VUNG XU LY (ZOOM-IN: BASELINE vs FAILURE CASE)", (340, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # Ghép toàn bộ thành 1 ảnh duy nhất (1400 x 510)
    final_img = np.vstack([top_row, banner_mid, zoom_row])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), final_img)
    print(f"-> Đã lưu ảnh Failure Case trực quan: {out_path} ({final_img.shape[1]}x{final_img.shape[0]})")


if __name__ == "__main__":
    create_failure_comparison_image()
