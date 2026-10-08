"""Bonus B3 (+2 điểm): Đo độ trễ Latency p50/p95 chuẩn qua 20 lần chạy và lưu thông số phần cứng.

Quy chuẩn đo lường:
  - Chạy lặp lại 21 lần trên frame 000011 (KITTI).
  - Loại bỏ lần chạy đầu tiên (warm-up, cấp phát bộ nhớ).
  - Báo trung vị p50 và phân vị p95.
  - Ghi nhận thông tin phần cứng: CPU, RAM, OS.
  - Xuất ra file CSV results/bonus/b3_latency_hardware.csv.
"""
from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import open3d as o3d
import pandas as pd

from starter.datasets import load_frame
from src.obstacle_detector import filter_roi, downsample_voxel, segment_ground_plane, cluster_obstacles


def get_hardware_info() -> dict:
    cpu_name = "11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz"
    try:
        import subprocess
        out = subprocess.run(["powershell", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True).stdout.strip()
        if out:
            cpu_name = out
    except Exception:
        pass

    return {
        "CPU": cpu_name,
        "RAM": "8.0 GB",
        "OS": f"{platform.system()} {platform.release()} ({platform.architecture()[0]})",
        "Python": platform.python_version(),
        "Backend": "Pure CPU (Open3D + NumPy)",
    }


def main() -> None:
    hw = get_hardware_info()
    print("=== THÔNG SỐ PHẦN CỨNG HỆ THỐNG ===")
    for k, v in hw.items():
        print(f"  {k}: {v}")

    data_root = "data/kitti_mini"
    frame_id = "000011"
    frame_data = load_frame(data_root, frame_id)
    roi_points = filter_roi(frame_data["points"], x_range=(0, 50), y_range=(-20, 20), z_range=(-2.5, 2.0))

    runs = 21
    latencies_voxel = []
    latencies_ransac = []
    latencies_dbscan = []
    latencies_total = []

    print(f"\nBắt đầu đo Latency qua {runs} lần chạy (bỏ lần đầu)...")
    for r in range(runs):
        t0 = time.perf_counter()
        pcd, down_pts = downsample_voxel(roi_points, voxel_size=0.10)
        t1 = time.perf_counter()
        _, _, g_pts, obs_pts, obs_pcd = segment_ground_plane(pcd, distance_threshold=0.20, num_iterations=100)
        t2 = time.perf_counter()
        labels, clusters = cluster_obstacles(obs_pcd, eps=0.5, min_points=10)
        t3 = time.perf_counter()

        vox_ms = (t1 - t0) * 1000
        ran_ms = (t2 - t1) * 1000
        dbs_ms = (t3 - t2) * 1000
        tot_ms = (t3 - t0) * 1000

        latencies_voxel.append(vox_ms)
        latencies_ransac.append(ran_ms)
        latencies_dbscan.append(dbs_ms)
        latencies_total.append(tot_ms)

    # Loại bỏ lần chạy đầu tiên
    valid_voxel = np.array(latencies_voxel[1:])
    valid_ransac = np.array(latencies_ransac[1:])
    valid_dbscan = np.array(latencies_dbscan[1:])
    valid_total = np.array(latencies_total[1:])

    p50_tot = np.percentile(valid_total, 50)
    p95_tot = np.percentile(valid_total, 95)
    p50_ran = np.percentile(valid_ransac, 50)
    p50_dbs = np.percentile(valid_dbscan, 50)

    print(f"\n=== KẾT QUẢ ĐO LATENCY (20 LẦN HỢP LỆ) ===")
    print(f"Tổng Pipeline: p50 = {p50_tot:.2f} ms | p95 = {p95_tot:.2f} ms")
    print(f"RANSAC Plane:  p50 = {p50_ran:.2f} ms")
    print(f"DBSCAN:        p50 = {p50_dbs:.2f} ms")

    # Tạo bảng CSV chi tiết
    records = []
    for i in range(1, runs):
        records.append({
            "Run_Iteration": i,
            "Voxel_time_ms": round(latencies_voxel[i], 2),
            "RANSAC_time_ms": round(latencies_ransac[i], 2),
            "DBSCAN_time_ms": round(latencies_dbscan[i], 2),
            "Total_latency_ms": round(latencies_total[i], 2),
            "Hardware_CPU": hw["CPU"],
            "Hardware_RAM": hw["RAM"],
            "Hardware_OS": hw["OS"],
        })

    df = pd.DataFrame(records)
    out_dir = ROOT / "results" / "bonus"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "b3_latency_hardware.csv"
    df.to_csv(csv_path, index=False)
    print(f"-> Đã lưu chi tiết 20 lần chạy: {csv_path}")


if __name__ == "__main__":
    main()
