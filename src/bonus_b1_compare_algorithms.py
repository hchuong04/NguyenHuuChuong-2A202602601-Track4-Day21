"""Bonus B1 (+4 điểm): So sánh 2 thuật toán tách mặt đất cho Robot/Drone:
  1. RANSAC Plane Segmentation (Thuật toán động, thích nghi mặt phẳng ax+by+cz+d=0)
  2. Fixed Height Thresholding (Cắt độ cao cố định z < -1.5m - baseline truyền thống)

Đánh giá trên cùng dữ liệu frame 000011 (KITTI):
  - Số điểm mặt đất / vật cản
  - Số cụm vật cản phát hiện được (DBSCAN eps=0.5m, min_points=10)
  - Độ nhạy với vật cản thấp sát đất (< 0.8m)
  - Độ trễ tính toán (Latency p50)
  - Xuất bảng so sánh CSV và ảnh đối sánh trực quan.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd

from starter.datasets import load_frame
from starter.projection import project_velo_to_image
from src.obstacle_detector import filter_roi, downsample_voxel, segment_ground_plane, cluster_obstacles


def run_fixed_height_segmentation(pcd: o3d.geometry.PointCloud,
                                  ground_z_threshold: float = -1.50) -> tuple[np.ndarray, np.ndarray, o3d.geometry.PointCloud]:
    """Tách mặt đất bằng ngưỡng độ cao cố định z < ground_z_threshold."""
    pts = np.asarray(pcd.points)
    ground_mask = pts[:, 2] < ground_z_threshold
    ground_indices = np.where(ground_mask)[0]

    ground_pcd = pcd.select_by_index(ground_indices)
    obstacle_pcd = pcd.select_by_index(ground_indices, invert=True)

    return np.asarray(ground_pcd.points), np.asarray(obstacle_pcd.points), obstacle_pcd


def main() -> None:
    data_root = "data/kitti_mini"
    frame_id = "000011"
    frame_data = load_frame(data_root, frame_id)
    calib = frame_data["calib"]
    raw_img = frame_data["image"]

    roi_points = filter_roi(frame_data["points"], x_range=(0, 50), y_range=(-20, 20), z_range=(-2.5, 2.0))
    pcd, down_pts = downsample_voxel(roi_points, voxel_size=0.10)

    # 1. Thuật toán A: RANSAC Plane (th = 0.20m)
    t0 = time.perf_counter()
    for _ in range(10):
        _, _, g_pts_ransac, obs_pts_ransac, obs_pcd_ransac = segment_ground_plane(pcd, distance_threshold=0.20, num_iterations=100)
    lat_ransac = (time.perf_counter() - t0) / 10 * 1000
    labels_ransac, cl_ransac = cluster_obstacles(obs_pcd_ransac, eps=0.5, min_points=10)

    # 2. Thuật toán B: Fixed Height Threshold (z < -1.50m)
    t0 = time.perf_counter()
    for _ in range(10):
        g_pts_fixed, obs_pts_fixed, obs_pcd_fixed = run_fixed_height_segmentation(pcd, ground_z_threshold=-1.50)
    lat_fixed = (time.perf_counter() - t0) / 10 * 1000
    labels_fixed, cl_fixed = cluster_obstacles(obs_pcd_fixed, eps=0.5, min_points=10)

    # Đếm điểm vật cản thấp sát đất (< 0.8m)
    near_mask_ransac = (obs_pts_ransac[:, 0] <= 15) & (obs_pts_ransac[:, 2] <= -0.8)
    near_mask_fixed = (obs_pts_fixed[:, 0] <= 15) & (obs_pts_fixed[:, 2] <= -0.8)

    records = [
        {
            "Thuat_toan": "RANSAC Plane (th=0.20m)",
            "Nguyen_ly": "Uoc luong mat phang thich ung",
            "Diem_mat_dat": len(g_pts_ransac),
            "Diem_vat_can": len(obs_pts_ransac),
            "So_clusters": len(cl_ransac),
            "Vat_thap_gan (<0.8m)": int(np.sum(near_mask_ransac)),
            "Vat_gan_nhat (m)": round(cl_ransac[0]["distance"], 2),
            "Latency_plane (ms)": round(lat_ransac, 2),
            "Uu_diem": "Thich nghi voi mat duong nghieng/doc",
            "Nhuoc_diem": "Cham hon do can 100 vong lap RANSAC",
        },
        {
            "Thuat_toan": "Fixed Height (z < -1.50m)",
            "Nguyen_ly": "Cat phang cung theo do cao cam bien",
            "Diem_mat_dat": len(g_pts_fixed),
            "Diem_vat_can": len(obs_pts_fixed),
            "So_clusters": len(cl_fixed),
            "Vat_thap_gan (<0.8m)": int(np.sum(near_mask_fixed)),
            "Vat_gan_nhat (m)": round(cl_fixed[0]["distance"], 2),
            "Latency_plane (ms)": round(lat_fixed, 2),
            "Uu_diem": "Sieu nhanh (phep so sanh mang numpy)",
            "Nhuoc_diem": "Sup do khi duong doc; nhan nham mat duong thanh vat can",
        }
    ]

    df = pd.DataFrame(records)
    out_dir = ROOT / "results" / "bonus"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "b1_algorithm_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"-> Đã lưu bảng so sánh B1: {csv_path}")

    # Vẽ biểu đồ so sánh trực quan
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # BEV RANSAC
    ax1 = axes[0]
    ax1.set_facecolor("#1e1e1e")
    ax1.scatter(g_pts_ransac[:, 1], g_pts_ransac[:, 0], s=0.8, c="#7f8c8d", alpha=0.4, label="Mat dat (Ground)")
    ax1.scatter(obs_pts_ransac[:, 1], obs_pts_ransac[:, 0], s=2.0, c="#2ecc71", alpha=0.9, label="Vat can (Obstacles)")
    ax1.set_title(f"A. RANSAC Plane Segmentation ({len(cl_ransac)} clusters | {lat_ransac:.1f}ms)", fontsize=12, fontweight="bold")
    ax1.set_xlim(15, -15)
    ax1.set_ylim(0, 40)
    ax1.set_xlabel("Y: Trai (+) / Phai (-) [m]")
    ax1.set_ylabel("X: Tien toi [m]")
    ax1.legend(loc="upper right", facecolor="#2c3e50", labelcolor="white")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # BEV Fixed Height
    ax2 = axes[1]
    ax2.set_facecolor("#1e1e1e")
    ax2.scatter(g_pts_fixed[:, 1], g_pts_fixed[:, 0], s=0.8, c="#7f8c8d", alpha=0.4, label="Mat dat (Ground)")
    ax2.scatter(obs_pts_fixed[:, 1], obs_pts_fixed[:, 0], s=2.0, c="#e67e22", alpha=0.9, label="Vat can (Obstacles)")
    ax2.set_title(f"B. Fixed Height Crop z < -1.5m ({len(cl_fixed)} clusters | {lat_fixed:.1f}ms)", fontsize=12, fontweight="bold")
    ax2.set_xlim(15, -15)
    ax2.set_ylim(0, 40)
    ax2.set_xlabel("Y: Trai (+) / Phai (-) [m]")
    ax2.set_ylabel("X: Tien toi [m]")
    ax2.legend(loc="upper right", facecolor="#2c3e50", labelcolor="white")
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle("BONUS [B1]: SO SANH RANSAC PLANE VS FIXED HEIGHT THRESHOLDING", fontsize=14, fontweight="bold")
    plt.tight_layout()
    chart_path = ROOT / "results" / "figures" / "bonus_b1_comparison.png"
    plt.savefig(str(chart_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> Đã lưu ảnh so sánh B1: {chart_path}")


if __name__ == "__main__":
    main()
