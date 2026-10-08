"""Pipeline phát hiện vật cản (Obstacle Detection) cho robot/drone bằng LiDAR không dùng Deep Learning.

Các bước trong pipeline:
  1. Lọc vùng quan tâm (ROI / Range cropping)
  2. Giảm mẫu điểm bằng Voxel Grid Downsampling
  3. Tách mặt đất bằng RANSAC Plane Segmentation (open3d.segment_plane)
  4. Gom cụm vật cản bằng DBSCAN Clustering (open3d.cluster_dbscan)
  5. Trích xuất Bounding Box cho từng cụm và chiếu lên ảnh camera (nếu có).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Đảm bảo in tiếng Việt trên console Windows không lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Đảm bảo import được module từ thư mục gốc của repo
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from starter.datasets import load_frame, dataset_type
from starter.projection import project_velo_to_image


def filter_roi(points: np.ndarray,
               x_range: tuple[float, float] = (0.0, 50.0),
               y_range: tuple[float, float] = (-20.0, 20.0),
               z_range: tuple[float, float] = (-2.5, 2.0)) -> np.ndarray:
    """Lọc vùng không gian quan tâm (ROI) phía trước xe/robot."""
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )
    return points[mask]


def downsample_voxel(points: np.ndarray, voxel_size: float = 0.1) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Giảm mẫu đám mây điểm theo kích thước voxel."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if voxel_size > 0:
        down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    else:
        down_pcd = pcd
    return down_pcd, np.asarray(down_pcd.points)


def segment_ground_plane(pcd: o3d.geometry.PointCloud,
                         distance_threshold: float = 0.2,
                         ransac_n: int = 3,
                         num_iterations: int = 100) -> tuple[list[float], list[int], np.ndarray, np.ndarray, o3d.geometry.PointCloud]:
    """Tách mặt đất bằng RANSAC Plane Segmentation.
    
    Trả về:
      plane_model: [a, b, c, d] phương trình mặt phẳng ax + by + cz + d = 0
      inliers: danh sách chỉ số các điểm thuộc mặt đất
      ground_pts: tọa độ các điểm mặt đất
      obstacle_pts: tọa độ các điểm vật cản (outliers)
      obstacle_pcd: Open3D PointCloud của vật cản
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    ground_pcd = pcd.select_by_index(inliers)
    obstacle_pcd = pcd.select_by_index(inliers, invert=True)

    ground_pts = np.asarray(ground_pcd.points)
    obstacle_pts = np.asarray(obstacle_pcd.points)

    return plane_model, inliers, ground_pts, obstacle_pts, obstacle_pcd


def cluster_obstacles(obstacle_pcd: o3d.geometry.PointCloud,
                      eps: float = 0.5,
                      min_points: int = 10) -> tuple[np.ndarray, list[dict]]:
    """Gom cụm các điểm vật cản bằng thuật toán DBSCAN."""
    pts = np.asarray(obstacle_pcd.points)
    if len(pts) == 0:
        return np.zeros((0,), dtype=int), []

    labels = np.array(obstacle_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    max_label = labels.max() if len(labels) > 0 else -1

    clusters = []
    for cid in range(max_label + 1):
        c_pts = pts[labels == cid]
        if len(c_pts) == 0:
            continue
        min_bound = c_pts.min(axis=0)
        max_bound = c_pts.max(axis=0)
        center = c_pts.mean(axis=0)
        dimensions = max_bound - min_bound
        distance = np.linalg.norm(center[:2])  # khoảng cách phẳng 2D tới ego

        clusters.append({
            "id": cid,
            "num_points": len(c_pts),
            "center": center,
            "min_bound": min_bound,
            "max_bound": max_bound,
            "dimensions": dimensions,
            "distance": distance,
            "points": c_pts
        })

    # Sắp xếp các cụm theo khoảng cách gần ego nhất trước
    clusters.sort(key=lambda c: c["distance"])
    return labels, clusters


def visualize_obstacle_pipeline(raw_points: np.ndarray,
                                ground_pts: np.ndarray,
                                obstacle_pts: np.ndarray,
                                labels: np.ndarray,
                                clusters: list[dict],
                                frame_data: dict,
                                out_path: Path,
                                title_extra: str = "") -> None:
    """Tạo ảnh 4 panel trực quan hóa toàn bộ pipeline phát hiện vật cản."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel 1: Raw Point Cloud BEV
    ax1 = axes[0, 0]
    ax1.set_facecolor("#1e1e1e")
    bev_pts = filter_roi(raw_points, x_range=(0, 50), y_range=(-20, 20), z_range=(-3, 3))
    sc1 = ax1.scatter(bev_pts[:, 1], bev_pts[:, 0], s=0.8, c=bev_pts[:, 2], cmap="viridis", alpha=0.7)
    plt.colorbar(sc1, ax=ax1, label="Height z (m)")
    ax1.set_title(f"1. Raw Point Cloud (BEV) - {len(raw_points)} điểm", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Y: Left (+) / Right (-) [m]")
    ax1.set_ylabel("X: Forward [m]")
    ax1.set_xlim(20, -20)
    ax1.set_ylim(0, 50)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Panel 2: Ground vs Obstacle Separation
    ax2 = axes[0, 1]
    ax2.set_facecolor("#1e1e1e")
    if len(ground_pts) > 0:
        ax2.scatter(ground_pts[:, 1], ground_pts[:, 0], s=0.6, c="#7f8c8d", alpha=0.4, label=f"Mặt đất ({len(ground_pts)})")
    if len(obstacle_pts) > 0:
        ax2.scatter(obstacle_pts[:, 1], obstacle_pts[:, 0], s=1.2, c="#e74c3c", alpha=0.8, label=f"Vật cản ({len(obstacle_pts)})")
    ax2.set_title(f"2. Tách mặt đất bằng RANSAC ({len(obstacle_pts)} vật cản)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Y: Left (+) / Right (-) [m]")
    ax2.set_ylabel("X: Forward [m]")
    ax2.set_xlim(20, -20)
    ax2.set_ylim(0, 50)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper right", facecolor="#2c3e50", edgecolor="none", labelcolor="white")

    # Panel 3: DBSCAN Clusters with Bounding Boxes
    ax3 = axes[1, 0]
    ax3.set_facecolor("#1e1e1e")
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(clusters):
        c_pts = c["points"]
        color = cmap(i % 20)
        ax3.scatter(c_pts[:, 1], c_pts[:, 0], s=3.0, color=color, alpha=0.9)
        # Vẽ 2D BEV Bounding Box
        min_b, max_b = c["min_bound"], c["max_bound"]
        rect = plt.Rectangle((min_b[1], min_b[0]), max_b[1] - min_b[1], max_b[0] - min_b[0],
                             fill=False, edgecolor=color, linewidth=1.5, linestyle="-")
        ax3.add_patch(rect)
        if i < 8:  # Hiển thị khoảng cách cho 8 vật cản gần nhất
            ax3.text(c["center"][1], c["center"][0] + 0.8, f"#{c['id']}: {c['distance']:.1f}m",
                     color="yellow", fontsize=8, fontweight="bold", ha="center")

    nearest_dist = clusters[0]["distance"] if len(clusters) > 0 else 0.0
    ax3.set_title(f"3. DBSCAN Clustered Obstacles ({len(clusters)} cụm, Gần nhất: {nearest_dist:.1f}m)",
                  fontsize=12, fontweight="bold")
    ax3.set_xlabel("Y: Left (+) / Right (-) [m]")
    ax3.set_ylabel("X: Forward [m]")
    ax3.set_xlim(20, -20)
    ax3.set_ylim(0, 50)
    ax3.grid(True, linestyle="--", alpha=0.3)

    # Panel 4: Projection of Obstacles on Camera Image
    ax4 = axes[1, 1]
    img = frame_data["image"].copy()
    if len(obstacle_pts) > 0 and "calib" in frame_data:
        calib = frame_data["calib"]
        uv, depth, mask = project_velo_to_image(obstacle_pts, calib, img.shape)
        # Tô màu từng cluster trên ảnh
        cluster_labels_on_mask = labels[mask] if len(labels) == len(obstacle_pts) else np.zeros(len(uv), dtype=int)
        for i, (u, v) in enumerate(uv.astype(int)):
            cid = cluster_labels_on_mask[i]
            if cid >= 0:
                color_rgb = [int(255 * x) for x in cmap(cid % 20)[:3]]
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            else:
                color_bgr = (120, 120, 120)
            cv2.circle(img, (u, v), 2, color_bgr, -1)

    # Chuyển BGR sang RGB cho Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax4.imshow(img_rgb)
    ax4.set_title("4. Chiếu các cụm vật cản lên ảnh Camera", fontsize=12, fontweight="bold")
    ax4.axis("off")

    plt.suptitle(f"Pipeline phát hiện vật cản (Robot/Drone Obstacle Detection) {title_extra}",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> Đã lưu ảnh trực quan hóa: {out_path}")


def run_pipeline(data_root: str,
                 frame_id: str,
                 voxel_size: float = 0.1,
                 distance_threshold: float = 0.2,
                 eps: float = 0.5,
                 min_points: int = 10,
                 out_dir: str = "results/figures") -> dict:
    """Chạy toàn bộ pipeline trên một frame và trả về thống kê & kết quả."""
    t0 = time.perf_counter()
    frame_data = load_frame(data_root, frame_id)
    raw_points = frame_data["points"]

    # 1. ROI filtering
    roi_points = filter_roi(raw_points, x_range=(0, 50), y_range=(-20, 20), z_range=(-2.5, 2.0))
    t_roi = time.perf_counter()

    # 2. Voxel Downsample
    pcd, down_pts = downsample_voxel(roi_points, voxel_size=voxel_size)
    t_down = time.perf_counter()

    # 3. Ground plane segmentation
    plane_model, inliers, ground_pts, obstacle_pts, obstacle_pcd = segment_ground_plane(
        pcd, distance_threshold=distance_threshold, ransac_n=3, num_iterations=100
    )
    t_plane = time.perf_counter()

    # 4. DBSCAN Clustering
    labels, clusters = cluster_obstacles(obstacle_pcd, eps=eps, min_points=min_points)
    t_cluster = time.perf_counter()

    total_time_ms = (t_cluster - t0) * 1000
    voxel_time_ms = (t_down - t_roi) * 1000
    plane_time_ms = (t_plane - t_down) * 1000
    cluster_time_ms = (t_cluster - t_plane) * 1000

    nearest_dist = clusters[0]["distance"] if len(clusters) > 0 else 0.0

    tag = f"frame_{frame_id}_vox{voxel_size}_dist{distance_threshold}_eps{eps}"
    out_img = Path(out_dir) / f"obstacle_demo_{tag}.png"
    visualize_obstacle_pipeline(
        raw_points, ground_pts, obstacle_pts, labels, clusters,
        frame_data, out_img, title_extra=f"[Frame: {frame_id} | {len(clusters)} cụm | {total_time_ms:.1f}ms]"
    )

    stats = {
        "frame_id": frame_id,
        "voxel_size": voxel_size,
        "distance_threshold": distance_threshold,
        "eps": eps,
        "min_points": min_points,
        "raw_points": len(raw_points),
        "roi_points": len(roi_points),
        "downsampled_points": len(down_pts),
        "ground_points": len(ground_pts),
        "obstacle_points": len(obstacle_pts),
        "num_clusters": len(clusters),
        "nearest_obstacle_dist": nearest_dist,
        "voxel_time_ms": voxel_time_ms,
        "plane_time_ms": plane_time_ms,
        "cluster_time_ms": cluster_time_ms,
        "total_time_ms": total_time_ms,
        "image_path": str(out_img),
    }

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline phát hiện vật cản cho robot/drone (Topic D)")
    parser.add_argument("--data-root", default="data/kitti_mini", help="Đường dẫn thư mục dữ liệu")
    parser.add_argument("--frame", default="000011", help="ID frame cần chạy (ví dụ 000011)")
    parser.add_argument("--voxel-size", type=float, default=0.1, help="Kích thước voxel downsample (m)")
    parser.add_argument("--distance-threshold", type=float, default=0.2, help="Ngưỡng khoảng cách RANSAC plane (m)")
    parser.add_argument("--eps", type=float, default=0.5, help="Bán kính lân cận DBSCAN (m)")
    parser.add_argument("--min-points", type=int, default=10, help="Số điểm tối thiểu để tạo cụm DBSCAN")
    parser.add_argument("--out-dir", default="results/figures", help="Thư mục lưu ảnh kết quả")

    args = parser.parse_args()

    stats = run_pipeline(
        data_root=args.data_root,
        frame_id=args.frame,
        voxel_size=args.voxel_size,
        distance_threshold=args.distance_threshold,
        eps=args.eps,
        min_points=args.min_points,
        out_dir=args.out_dir
    )

    print("\n=== KẾT QUẢ PIPELINE PHÁT HIỆN VẬT CẢN ===")
    print(f"Frame ID:                {stats['frame_id']}")
    print(f"Tổng số điểm thô:        {stats['raw_points']}")
    print(f"Số điểm sau Voxel:       {stats['downsampled_points']} (voxel={stats['voxel_size']}m)")
    print(f"Số điểm mặt đất (RANSAC):{stats['ground_points']} ({stats['ground_points']/stats['downsampled_points']*100:.1f}%)")
    print(f"Số điểm vật cản:         {stats['obstacle_points']} ({stats['obstacle_points']/stats['downsampled_points']*100:.1f}%)")
    print(f"Số cụm vật cản (DBSCAN): {stats['num_clusters']}")
    print(f"Khoảng cách vật gần nhất:{stats['nearest_obstacle_dist']:.2f} m")
    print(f"Thời gian xử lý:         {stats['total_time_ms']:.1f} ms (RANSAC: {stats['plane_time_ms']:.1f}ms, DBSCAN: {stats['cluster_time_ms']:.1f}ms)")


if __name__ == "__main__":
    main()
