"""Bonus B4 (+3 điểm): Unified Reusable CLI Toolkit for LiDAR Obstacle Detection.

Công cụ dòng lệnh tái sử dụng cho các bài lab và dự án robot/xe tự hành tiếp theo:
  - Hỗ trợ đầy đủ tham số dòng lệnh bằng argparse với --help chi tiết.
  - Tự động nhận diện cấu trúc dataset (KITTI hoặc nuScenes).
  - Có các chế độ:
      + detect: Chạy pipeline phát hiện vật cản trên 1 frame và lưu ảnh demo 4-panel.
      + compare: So sánh RANSAC vs Fixed Height Threshold (Bonus B1).
      + stress: Chạy suy giảm dữ liệu Gaussian Noise / Dropout (Bonus B2).
      + cross-eval: Đánh giá chéo KITTI vs nuScenes (Bonus B5).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.obstacle_detector import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.obstacle_toolkit",
        description="Unified Toolkit for LiDAR 3D Obstacle Detection & Benchmarking (VinUni AI20K - Track 4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--mode", choices=["detect", "compare", "stress", "cross-eval"], default="detect",
                        help="Chế độ hoạt động: detect (phát hiện cơ bản), compare (so sánh thuật toán B1), stress (stress test B2), cross-eval (đánh giá chéo B5)")
    parser.add_argument("--data-root", default="data/kitti_mini",
                        help="Đường dẫn thư mục dữ liệu (data/kitti_mini, data/synthetic, data/nuscenes_mini_subset)")
    parser.add_argument("--frame", default="000011",
                        help="Mã frame cần xử lý (ví dụ 000011 trên KITTI hoặc scene-0103_010 trên nuScenes)")
    parser.add_argument("--voxel-size", type=float, default=0.10,
                        help="Kích thước voxel downsampling (m)")
    parser.add_argument("--distance-threshold", type=float, default=0.20,
                        help="Ngưỡng khoảng cách RANSAC plane segmentation (m)")
    parser.add_argument("--eps", type=float, default=0.50,
                        help="Bán kính láng giềng DBSCAN clustering (m)")
    parser.add_argument("--min-points", type=int, default=10,
                        help="Số điểm tối thiểu để tạo thành 1 cụm DBSCAN")
    parser.add_argument("--out-dir", default="results/figures",
                        help="Thư mục xuất ảnh kết quả")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print(f"=== KHỞI CHẠY OBSTACLE TOOLKIT [Chế độ: {args.mode.upper()}] ===")
    if args.mode == "detect":
        stats = run_pipeline(
            data_root=args.data_root,
            frame_id=args.frame,
            voxel_size=args.voxel_size,
            distance_threshold=args.distance_threshold,
            eps=args.eps,
            min_points=args.min_points,
            out_dir=args.out_dir
        )
        print("\n[THÀNH CÔNG] Kết quả phát hiện:")
        print(f"  Frame: {stats['frame_id']} | Điểm: {stats['downsampled_points']} | Vật cản: {stats['num_clusters']} cụm | Gần nhất: {stats['nearest_obstacle_dist']:.2f}m")
        print(f"  Thời gian: {stats['total_time_ms']:.1f}ms -> Ảnh: {stats['image_path']}")

    elif args.mode == "compare":
        from src.bonus_b1_compare_algorithms import main as run_compare
        run_compare()

    elif args.mode == "stress":
        from src.bonus_b2_stress_test import main as run_stress
        run_stress()

    elif args.mode == "cross-eval":
        from src.bonus_b5_cross_dataset import main as run_crosseval
        run_crosseval()


if __name__ == "__main__":
    main()
