"""Thực hiện thí nghiệm Benchmark & Stress Test cho Topic D (Robot/Drone Obstacle Detection).

Thí nghiệm gồm 2 phần quét tham số (Sweep):
  1. Quét ngưỡng khoảng cách RANSAC plane (distance_threshold từ 0.10m đến 0.50m):
     Đo lường sự thay đổi của số lượng cụm vật cản, điểm vật cản thấp gần đất, và khoảng cách tới vật cản gần nhất.
  2. Quét kích thước Voxel Grid (voxel_size từ 0.05m đến 0.30m):
     Đo lường sự đánh đổi giữa thời gian xử lý (Latency p50/p95 qua 20 lần lặp) và số lượng cụm phát hiện được.

Kết quả xuất ra:
  - CSV: results/obstacle_benchmark_summary.csv
  - Biểu đồ: results/figures/obstacle_benchmark_analysis.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Đảm bảo in tiếng Việt trên console Windows không lỗi charmap
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


def measure_latency_distribution(pcd: o3d.geometry.PointCloud,
                                 distance_threshold: float,
                                 eps: float,
                                 min_points: int,
                                 num_runs: int = 20) -> tuple[float, float, float, float]:
    """Đo độ trễ p50 và p95 qua num_runs lần chạy (bỏ lần đầu tiên) - Chuẩn Bonus B3."""
    times_ransac = []
    times_dbscan = []
    times_total = []

    # Warm-up (bỏ lần chạy đầu để loại bỏ overhead khởi tạo)
    segment_ground_plane(pcd, distance_threshold=distance_threshold, num_iterations=80)

    for _ in range(num_runs):
        t0 = time.perf_counter()
        _, _, _, _, obs_pcd = segment_ground_plane(pcd, distance_threshold=distance_threshold, num_iterations=80)
        t1 = time.perf_counter()
        _ = np.array(obs_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        t2 = time.perf_counter()

        times_ransac.append((t1 - t0) * 1000)
        times_dbscan.append((t2 - t1) * 1000)
        times_total.append((t2 - t0) * 1000)

    p50_total = float(np.percentile(times_total, 50))
    p95_total = float(np.percentile(times_total, 95))
    p50_ransac = float(np.percentile(times_ransac, 50))
    p50_dbscan = float(np.percentile(times_dbscan, 50))

    return p50_total, p95_total, p50_ransac, p50_dbscan


def run_ransac_sweep(roi_points: np.ndarray,
                     fixed_voxel_size: float = 0.10,
                     dist_thresholds: list[float] | None = None,
                     eps: float = 0.5,
                     min_points: int = 10) -> pd.DataFrame:
    """Quét distance_threshold của RANSAC plane."""
    if dist_thresholds is None:
        dist_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    np.random.seed(42)  # Cố định seed
    pcd, down_pts = downsample_voxel(roi_points, voxel_size=fixed_voxel_size)

    records = []
    print(f"\n--- BẮT ĐẦU SWEEP 1: RANSAC distance_threshold ({len(dist_thresholds)} mức) ---")
    for dist_th in dist_thresholds:
        plane_model, inliers, ground_pts, obstacle_pts, obstacle_pcd = segment_ground_plane(
            pcd, distance_threshold=dist_th, num_iterations=100
        )
        labels, clusters = cluster_obstacles(obstacle_pcd, eps=eps, min_points=min_points)

        # Đếm số điểm vật cản thấp (< 0.8m so với mặt đất) ở cự ly gần (< 15m)
        # Trong hệ KITTI, mặt đất phẳng thường nằm ở z ~ -1.5m đến -1.7m
        # Điểm sát đất có z trong khoảng [-1.6, -0.8]
        near_mask = (obstacle_pts[:, 0] <= 15.0) & (obstacle_pts[:, 0] >= 0.0) & \
                    (obstacle_pts[:, 1] >= -10.0) & (obstacle_pts[:, 1] <= 10.0) & \
                    (obstacle_pts[:, 2] <= -0.8) & (obstacle_pts[:, 2] >= -1.6)
        low_obstacle_pts_near = int(np.sum(near_mask))

        nearest_dist = float(clusters[0]["distance"]) if len(clusters) > 0 else 0.0
        p50_tot, p95_tot, _, _ = measure_latency_distribution(pcd, dist_th, eps, min_points, num_runs=20)

        record = {
            "dist_threshold_m": dist_th,
            "voxel_size_m": fixed_voxel_size,
            "total_down_points": len(down_pts),
            "ground_points": len(ground_pts),
            "obstacle_points": len(obstacle_pts),
            "ground_ratio_pct": round(len(ground_pts) / len(down_pts) * 100, 2),
            "num_clusters": len(clusters),
            "nearest_dist_m": round(nearest_dist, 2),
            "low_obstacle_pts_near": low_obstacle_pts_near,
            "latency_p50_ms": round(p50_tot, 2),
            "latency_p95_ms": round(p95_tot, 2),
        }
        records.append(record)
        print(f"dist_th={dist_th:.2f}m -> Ground: {record['ground_ratio_pct']}% | Clusters: {len(clusters)} | "
              f"Vật thấp gần: {low_obstacle_pts_near} pts | Nearest: {record['nearest_dist_m']}m | Latency: {p50_tot:.1f}ms")

    return pd.DataFrame(records)


def run_voxel_sweep(roi_points: np.ndarray,
                    fixed_dist_th: float = 0.20,
                    voxel_sizes: list[float] | None = None,
                    eps: float = 0.5,
                    min_points: int = 10) -> pd.DataFrame:
    """Quét voxel_size downsample."""
    if voxel_sizes is None:
        voxel_sizes = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30]

    np.random.seed(42)  # Cố định seed
    records = []
    print(f"\n--- BẮT ĐẦU SWEEP 2: Voxel Size ({len(voxel_sizes)} mức) ---")
    for vox in voxel_sizes:
        pcd, down_pts = downsample_voxel(roi_points, voxel_size=vox)
        plane_model, inliers, ground_pts, obstacle_pts, obstacle_pcd = segment_ground_plane(
            pcd, distance_threshold=fixed_dist_th, num_iterations=100
        )
        labels, clusters = cluster_obstacles(obstacle_pcd, eps=eps, min_points=min_points)

        # Đếm cụm nhỏ hoặc xa (> 25m)
        distant_clusters = sum(1 for c in clusters if c["distance"] > 25.0)

        p50_tot, p95_tot, p50_ran, p50_db = measure_latency_distribution(pcd, fixed_dist_th, eps, min_points, num_runs=20)

        record = {
            "voxel_size_m": vox,
            "dist_threshold_m": fixed_dist_th,
            "downsampled_points": len(down_pts),
            "ground_points": len(ground_pts),
            "obstacle_points": len(obstacle_pts),
            "num_clusters": len(clusters),
            "distant_clusters_gt25m": distant_clusters,
            "ransac_p50_ms": round(p50_ran, 2),
            "dbscan_p50_ms": round(p50_db, 2),
            "latency_p50_ms": round(p50_tot, 2),
            "latency_p95_ms": round(p95_tot, 2),
        }
        records.append(record)
        print(f"voxel={vox:.2f}m -> Points: {len(down_pts)} | Clusters: {len(clusters)} | "
              f"Xa >25m: {distant_clusters} | Latency p50: {p50_tot:.1f}ms (p95: {p95_tot:.1f}ms)")

    return pd.DataFrame(records)


def plot_benchmark_results(df_ransac: pd.DataFrame,
                           df_voxel: pd.DataFrame,
                           out_chart_path: Path) -> None:
    """Vẽ biểu đồ phân tích 4 khung hình biểu diễn sự đánh đổi của các tham số."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. RANSAC dist vs Số Cluster & Điểm vật cản thấp sát đất
    ax1 = axes[0, 0]
    color1 = "#2980b9"
    color2 = "#c0392b"
    ax1.set_xlabel("RANSAC Plane distance_threshold (m)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Số lượng Clusters phát hiện", color=color1, fontsize=11, fontweight="bold")
    line1 = ax1.plot(df_ransac["dist_threshold_m"], df_ransac["num_clusters"], "o-", color=color1, linewidth=2.5, label="Số Clusters")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel("Điểm vật cản thấp sát đất (< 0.8m)", color=color2, fontsize=11, fontweight="bold")
    line2 = ax1_twin.plot(df_ransac["dist_threshold_m"], df_ransac["low_obstacle_pts_near"], "s--", color=color2, linewidth=2.5, label="Điểm vật cản thấp gần")
    ax1_twin.tick_params(axis="y", labelcolor=color2)
    ax1.set_title("1. Ngưỡng RANSAC vs Khả năng giữ lại vật cản thấp", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. RANSAC dist vs Tỷ lệ mặt đất & Khoảng cách vật gần nhất
    ax2 = axes[0, 1]
    color3 = "#27ae60"
    color4 = "#e67e22"
    ax2.set_xlabel("RANSAC Plane distance_threshold (m)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Tỷ lệ điểm mặt đất Ground (%)", color=color3, fontsize=11, fontweight="bold")
    ax2.plot(df_ransac["dist_threshold_m"], df_ransac["ground_ratio_pct"], "^-", color=color3, linewidth=2.5)
    ax2.tick_params(axis="y", labelcolor=color3)

    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel("Khoảng cách vật cản gần nhất (m)", color=color4, fontsize=11, fontweight="bold")
    ax2_twin.plot(df_ransac["dist_threshold_m"], df_ransac["nearest_dist_m"], "d-.", color=color4, linewidth=2.5)
    ax2_twin.tick_params(axis="y", labelcolor=color4)
    ax2.set_title("2. Ngưỡng RANSAC vs Nhầm lẫn vật cản cự ly gần", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Voxel size vs Latency p50 và p95
    ax3 = axes[1, 0]
    ax3.plot(df_voxel["voxel_size_m"], df_voxel["latency_p50_ms"], "o-", color="#8e44ad", linewidth=2.5, label="Latency p50 (ms)")
    ax3.plot(df_voxel["voxel_size_m"], df_voxel["latency_p95_ms"], "x--", color="#d35400", linewidth=2.0, label="Latency p95 (ms)")
    ax3.axhline(y=33.3, color="red", linestyle=":", label="Ngưỡng Realtime 30 FPS (33.3ms)")
    ax3.set_xlabel("Voxel Size (m)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Độ trễ xử lý (ms) [CPU]", fontsize=11, fontweight="bold")
    ax3.set_title("3. Voxel Size vs Độ trễ xử lý (Latency Benchmark)", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right")
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Voxel size vs Số lượng điểm & Số cụm ở xa (> 25m)
    ax4 = axes[1, 1]
    color5 = "#16a085"
    color6 = "#e74c3c"
    ax4.set_xlabel("Voxel Size (m)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Số điểm sau Voxel", color=color5, fontsize=11, fontweight="bold")
    ax4.plot(df_voxel["voxel_size_m"], df_voxel["downsampled_points"], "o-", color=color5, linewidth=2.5)
    ax4.tick_params(axis="y", labelcolor=color5)

    ax4_twin = ax4.twinx()
    ax4_twin.set_ylabel("Số cụm ở xa (> 25m)", color=color6, fontsize=11, fontweight="bold")
    ax4_twin.plot(df_voxel["voxel_size_m"], df_voxel["distant_clusters_gt25m"], "s--", color=color6, linewidth=2.5)
    ax4_twin.tick_params(axis="y", labelcolor=color6)
    ax4.set_title("4. Voxel Size vs Mất mát vật thể ở cự ly xa", fontsize=12, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("ĐÁNH GIÁ ĐÁNH ĐỔI THAM SỐ (BENCHMARK SWEEP) CHO ROBOT OBSTACLE PIPELINE",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_chart_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_chart_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> Đã lưu biểu đồ phân tích: {out_chart_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy Benchmark và Stress Test cho Topic D")
    parser.add_argument("--data-root", default="data/kitti_mini", help="Đường dẫn dữ liệu")
    parser.add_argument("--frame", default="000011", help="ID frame dùng để benchmark")
    parser.add_argument("--out-dir", default="results", help="Thư mục xuất kết quả")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== BẮT ĐẦU BENCHMARK THÍ NGHIỆM CHÍNH (FRAME {args.frame}) ===")
    frame_data = load_frame(args.data_root, args.frame)
    roi_points = filter_roi(frame_data["points"], x_range=(0, 50), y_range=(-20, 20), z_range=(-2.5, 2.0))
    print(f"Tổng số điểm trong ROI: {len(roi_points)}")

    # Sweep 1: RANSAC distance threshold
    df_ransac = run_ransac_sweep(roi_points, fixed_voxel_size=0.10)
    df_ransac.to_csv(out_dir / "obstacle_ransac_sweep.csv", index=False)
    print(f"-> Đã lưu {out_dir / 'obstacle_ransac_sweep.csv'}")

    # Sweep 2: Voxel size
    df_voxel = run_voxel_sweep(roi_points, fixed_dist_th=0.20)
    df_voxel.to_csv(out_dir / "obstacle_voxel_sweep.csv", index=False)
    print(f"-> Đã lưu {out_dir / 'obstacle_voxel_sweep.csv'}")

    # Tạo bảng tổng hợp để điền vào REPORT.md
    df_ransac["experiment"] = "RANSAC_Threshold_Sweep"
    df_voxel["experiment"] = "Voxel_Size_Sweep"
    df_summary = pd.concat([df_ransac, df_voxel], ignore_index=True)
    df_summary.to_csv(out_dir / "obstacle_benchmark_summary.csv", index=False)

    # Vẽ biểu đồ
    chart_path = fig_dir / "obstacle_benchmark_analysis.png"
    plot_benchmark_results(df_ransac, df_voxel, chart_path)

    print("\n=== HOÀN TẤT BENCHMARK CHÍNH (CP3) ===")


if __name__ == "__main__":
    main()
