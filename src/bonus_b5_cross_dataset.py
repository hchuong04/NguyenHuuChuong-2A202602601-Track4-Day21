"""Bonus B5 (+2 điểm): So sánh pipeline phát hiện vật cản trên 2 dataset thật:
  1. KITTI 3D Object (Velodyne HDL-64E, 64 tia, ~108,000 điểm/frame, ban ngày)
  2. nuScenes v1.0-mini scene-0103 (LiDAR 32 tia, 34,720 điểm/frame, ban ngày)
  3. nuScenes v1.0-mini scene-1094 (LiDAR 32 tia, 34,720 điểm/frame, ban đêm sau mưa)

Đánh giá các khía cạnh:
  - Mật độ điểm và số điểm rơi vào ROI
  - Số cụm vật cản phát hiện được (DBSCAN)
  - Khả năng phát hiện vật cản xa (>20m) với mật độ tia thưa (32 tia vs 64 tia)
  - Xuất bảng so sánh CSV và biểu đồ trực quan đa cảm biến.
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd

from starter.datasets import load_frame
from src.obstacle_detector import filter_roi, downsample_voxel, segment_ground_plane, cluster_obstacles


def evaluate_dataset_frame(data_root: str,
                           frame_id: str,
                           dataset_name: str,
                           condition: str,
                           beams: int,
                           voxel_size: float = 0.10,
                           dist_th: float = 0.20,
                           eps: float = 0.50,
                           min_points: int = 10) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
    frame_data = load_frame(data_root, frame_id)
    raw_points = frame_data["points"]

    roi_points = filter_roi(raw_points, x_range=(0, 50), y_range=(-20, 20), z_range=(-2.5, 2.0))
    pcd, down_pts = downsample_voxel(roi_points, voxel_size=voxel_size)

    _, _, g_pts, obs_pts, obs_pcd = segment_ground_plane(pcd, distance_threshold=dist_th, num_iterations=100)
    labels, clusters = cluster_obstacles(obs_pcd, eps=eps, min_points=min_points)

    clusters_gt20m = sum(1 for c in clusters if c["distance"] > 20.0)
    nearest_dist = clusters[0]["distance"] if len(clusters) > 0 else 0.0

    stat = {
        "Dataset": dataset_name,
        "Frame_ID": frame_id,
        "Dieu_kien": condition,
        "So_beam_LiDAR": beams,
        "Tong_diem_tho": len(raw_points),
        "Diem_sau_Voxel": len(down_pts),
        "Diem_mat_dat": len(g_pts),
        "Diem_vat_can": len(obs_pts),
        "So_clusters": len(clusters),
        "Clusters_xa (>20m)": clusters_gt20m,
        "Vat_gan_nhat (m)": round(nearest_dist, 2),
    }

    return stat, g_pts, obs_pts, clusters


def main() -> None:
    test_cases = [
        ("data/kitti_mini", "000011", "KITTI HDL-64E", "Ban ngày, trời quang", 64),
        ("data/nuscenes_mini_subset", "scene-0103_010", "nuScenes 32-beam", "Ban ngày, đô thị Singapore", 32),
        ("data/nuscenes_mini_subset", "scene-1094_010", "nuScenes 32-beam", "Ban đêm sau mưa, đường ướt", 32),
    ]

    records = []
    results_vis = []
    for data_root, frame_id, ds_name, cond, beams in test_cases:
        stat, g_pts, obs_pts, clusters = evaluate_dataset_frame(data_root, frame_id, ds_name, cond, beams)
        records.append(stat)
        results_vis.append((ds_name, cond, g_pts, obs_pts, clusters))
        print(f"[{ds_name}] Frame {frame_id}: Thô={stat['Tong_diem_tho']} | Cụm={stat['So_clusters']} | Cụm xa={stat['Clusters_xa (>20m)']} | Gần={stat['Vat_gan_nhat (m)']}m")

    df = pd.DataFrame(records)
    out_dir = ROOT / "results" / "bonus"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "b5_kitti_vs_nuscenes.csv"
    df.to_csv(csv_path, index=False)
    print(f"-> Đã lưu bảng so sánh B5: {csv_path}")

    # Vẽ biểu đồ 3 cột trực quan
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (ds_name, cond, g_pts, obs_pts, clusters) in enumerate(results_vis):
        ax = axes[i]
        ax.set_facecolor("#1e1e1e")
        if len(g_pts) > 0:
            ax.scatter(g_pts[:, 1], g_pts[:, 0], s=0.5, c="#7f8c8d", alpha=0.3, label="Mặt đất")
        if len(obs_pts) > 0:
            ax.scatter(obs_pts[:, 1], obs_pts[:, 0], s=1.5, c="#e74c3c", alpha=0.8, label="Vật cản")

        # Vẽ bounding box cho các cụm
        cmap = plt.get_cmap("tab20")
        for cid, c in enumerate(clusters):
            min_b, max_b = c["min_bound"], c["max_bound"]
            rect = plt.Rectangle((min_b[1], min_b[0]), max_b[1] - min_b[1], max_b[0] - min_b[0],
                                 fill=False, edgecolor=cmap(cid % 20), linewidth=1.2)
            ax.add_patch(rect)

        ax.set_title(f"{i+1}. {ds_name}\n[{cond}]\n({len(clusters)} cụm | Gần: {clusters[0]['distance']:.1f}m)",
                     fontsize=11, fontweight="bold")
        ax.set_xlim(15, -15)
        ax.set_ylim(0, 45)
        ax.set_xlabel("Y: Trái (+) / Phải (-) [m]")
        ax.set_ylabel("X: Tiến tới [m]")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", facecolor="#2c3e50", labelcolor="white")

    plt.suptitle("BONUS [B5]: SO SÁNH HIỆU QUẢ PHÁT HIỆN VẬT CẢN TRÊN KITTI (64 BEAM) VÀ NUSCENES (32 BEAM)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    chart_path = ROOT / "results" / "figures" / "bonus_b5_cross_dataset.png"
    plt.savefig(str(chart_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> Đã lưu ảnh so sánh B5: {chart_path}")


if __name__ == "__main__":
    main()
