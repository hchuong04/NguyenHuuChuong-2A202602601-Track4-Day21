"""Bonus B2 (+3 điểm): Stress test suy giảm dữ liệu cảm biến cho Topic D.

Áp dụng 2 loại suy giảm dữ liệu từ starter/perturb.py:
  1. Nhiễu Gaussian (mô phỏng mưa tuyết / cảm biến LiDAR bị rung lắc):
     sigma_xyz = 0.00m (gốc), 0.02m, 0.05m, 0.10m, 0.15m.
  2. Bỏ bớt điểm ngẫu nhiên (Random Dropout - mô phỏng cảm biến bị bám bụi / công suất yếu):
     keep_ratio = 1.0 (gốc), 0.9, 0.7, 0.5, 0.3.

Đo đạc:
  - Số lượng cluster DBSCAN phát hiện được
  - Khoảng cách tới vật cản gần nhất (độ ổn định của phanh khẩn cấp)
  - Số điểm vật cản còn giữ lại
  - Vẽ biểu đồ suy giảm và xuất CSV.
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
from starter.perturb import gaussian_noise, random_dropout
from src.obstacle_detector import filter_roi, downsample_voxel, segment_ground_plane, cluster_obstacles


def main() -> None:
    data_root = "data/kitti_mini"
    frame_id = "000011"
    frame_data = load_frame(data_root, frame_id)
    raw_points = frame_data["points"]

    records = []
    print("\n--- BẮT ĐẦU STRESS TEST 1: GAUSSIAN NOISE ---")
    sigmas = [0.00, 0.02, 0.05, 0.10, 0.15]
    for sig in sigmas:
        noisy_pts = gaussian_noise(raw_points, sigma_xyz_m=sig, seed=42) if sig > 0 else raw_points
        roi_pts = filter_roi(noisy_pts, x_range=(0, 50), y_range=(-20, 20), z_range=(-2.5, 2.0))
        pcd, down_pts = downsample_voxel(roi_pts, voxel_size=0.10)
        _, _, g_pts, obs_pts, obs_pcd = segment_ground_plane(pcd, distance_threshold=0.20, num_iterations=100)
        labels, clusters = cluster_obstacles(obs_pcd, eps=0.5, min_points=10)
        nearest = clusters[0]["distance"] if len(clusters) > 0 else 0.0

        records.append({
            "Perturbation_type": "Gaussian_Noise",
            "Level_param": f"sigma={sig:.2f}m",
            "Level_value": sig,
            "Total_down_points": len(down_pts),
            "Obstacle_points": len(obs_pts),
            "Num_clusters": len(clusters),
            "Nearest_dist_m": round(nearest, 2),
        })
        print(f"Noise sigma={sig:.2f}m -> Clusters: {len(clusters)} | Nearest: {nearest:.2f}m | Obs pts: {len(obs_pts)}")

    print("\n--- BẮT ĐẦU STRESS TEST 2: RANDOM DROPOUT ---")
    keep_ratios = [1.0, 0.9, 0.7, 0.5, 0.3]
    for kr in keep_ratios:
        dropped_pts = random_dropout(raw_points, keep_ratio=kr, seed=42) if kr < 1.0 else raw_points
        roi_pts = filter_roi(dropped_pts, x_range=(0, 50), y_range=(-20, 20), z_range=(-2.5, 2.0))
        pcd, down_pts = downsample_voxel(roi_pts, voxel_size=0.10)
        _, _, g_pts, obs_pts, obs_pcd = segment_ground_plane(pcd, distance_threshold=0.20, num_iterations=100)
        labels, clusters = cluster_obstacles(obs_pcd, eps=0.5, min_points=10)
        nearest = clusters[0]["distance"] if len(clusters) > 0 else 0.0

        records.append({
            "Perturbation_type": "Random_Dropout",
            "Level_param": f"keep_ratio={kr:.1f}",
            "Level_value": kr,
            "Total_down_points": len(down_pts),
            "Obstacle_points": len(obs_pts),
            "Num_clusters": len(clusters),
            "Nearest_dist_m": round(nearest, 2),
        })
        print(f"Dropout keep_ratio={kr:.1f} -> Clusters: {len(clusters)} | Nearest: {nearest:.2f}m | Obs pts: {len(obs_pts)}")

    df = pd.DataFrame(records)
    out_dir = ROOT / "results" / "bonus"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "b2_stress_test.csv"
    df.to_csv(csv_path, index=False)
    print(f"-> Đã lưu kết quả Stress Test B2: {csv_path}")

    # Vẽ biểu đồ suy giảm
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    df_noise = df[df["Perturbation_type"] == "Gaussian_Noise"]
    df_drop = df[df["Perturbation_type"] == "Random_Dropout"]

    # Plot 1: Gaussian Noise
    ax1 = axes[0]
    ax1.plot(df_noise["Level_value"], df_noise["Num_clusters"], "o-", color="#c0392b", linewidth=2.5, label="Số Clusters")
    ax1.set_xlabel("Độ lệch chuẩn nhiễu Gaussian sigma (m)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Số lượng Clusters phát hiện", color="#c0392b", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#c0392b")

    ax1_twin = ax1.twinx()
    ax1_twin.plot(df_noise["Level_value"], df_noise["Nearest_dist_m"], "s--", color="#2980b9", linewidth=2.5, label="Khoảng cách gần nhất")
    ax1_twin.set_ylabel("Khoảng cách vật gần nhất (m)", color="#2980b9", fontsize=11, fontweight="bold")
    ax1_twin.tick_params(axis="y", labelcolor="#2980b9")
    ax1.set_title("A. Ảnh hưởng của Nhiễu Gaussian (Mưa/Rung lắc)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Random Dropout
    ax2 = axes[1]
    ax2.plot(df_drop["Level_value"], df_drop["Num_clusters"], "^-", color="#27ae60", linewidth=2.5, label="Số Clusters")
    ax2.set_xlabel("Tỷ lệ điểm giữ lại (keep_ratio)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Số lượng Clusters phát hiện", color="#27ae60", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#27ae60")

    ax2_twin = ax2.twinx()
    ax2_twin.plot(df_drop["Level_value"], df_drop["Obstacle_points"], "d--", color="#8e44ad", linewidth=2.5, label="Số điểm vật cản")
    ax2_twin.set_ylabel("Số điểm vật cản còn sót", color="#8e44ad", fontsize=11, fontweight="bold")
    ax2_twin.tick_params(axis="y", labelcolor="#8e44ad")
    ax2.set_title("B. Ảnh hưởng của Dropout (Bụi bám/Mất tia)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("BONUS [B2]: STRESS TEST SUY GIẢM DỮ LIỆU CẢM BIẾN (NOISE & DROPOUT)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    chart_path = ROOT / "results" / "figures" / "bonus_b2_degradation.png"
    plt.savefig(str(chart_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> Đã lưu biểu đồ Stress test B2: {chart_path}")


if __name__ == "__main__":
    main()
