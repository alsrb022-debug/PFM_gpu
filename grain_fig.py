import math
import configparser
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# input.txt 읽기
# ============================================================
def read_input(input_path):
    cfg = configparser.ConfigParser()
    cfg.read(input_path, encoding="utf-8")

    p = {}

    p["im"] = cfg.getint("domain", "im")
    p["jm"] = cfg.getint("domain", "jm")
    p["km"] = cfg.getint("domain", "km")

    p["dx"] = cfg.getfloat("grid", "dx")
    p["dy"] = cfg.getfloat("grid", "dy")
    p["dz"] = cfg.getfloat("grid", "dz")

    p["nmax"] = cfg.getint("pf", "nmax")
    p["nph"] = cfg.getint("pf", "nph")
    p["pss"] = cfg.getfloat("pf", "pss")

    p["nstep"] = cfg.getint("time", "nstep")
    p["nout"] = cfg.getint("time", "nout")

    p["init_type"] = cfg.get("init", "type")
    p["bc_type"] = cfg.get("bc", "type")

    return p


# ============================================================
# snapshot 파일 목록
# ============================================================
def find_snapshot_files(data_dir: Path):
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    return files


# ============================================================
# 2D grain label field 추출
# id[..., 0]을 대표 grain id로 사용
# ============================================================
def load_label_and_meta(npz_path: Path, id_slot=0):
    data = np.load(npz_path)

    if "id" not in data:
        raise KeyError(f"'id' not found in {npz_path}")

    ids = data["id"]
    if ids.ndim != 4:
        raise ValueError(
            f"{npz_path.name}: expected id.ndim == 4, got {ids.ndim}"
        )

    label = ids[:, :, :, id_slot]

    step = int(data["step"]) if "step" in data else None
    time = float(data["time"]) if "time" in data else None

    return label, step, time


# ============================================================
# 2D 등가 반경 계산
# R = sqrt(A / pi)
# ============================================================
def grain_radii_from_label_2d(label_2d: np.ndarray, dx: float, min_pixels: int = 1):
    valid = label_2d[label_2d >= 0]
    if valid.size == 0:
        return np.array([], dtype=np.float64)

    _, counts = np.unique(valid, return_counts=True)

    counts = counts[counts >= min_pixels]
    if counts.size == 0:
        return np.array([], dtype=np.float64)

    area = counts.astype(np.float64) * (dx * dx)
    radii = np.sqrt(area / math.pi)
    return radii


# ============================================================
# 전체 분석
# ============================================================
def analyze_growth_2d(data_dir: Path, dx: float, id_slot=0, min_pixels=1):
    files = find_snapshot_files(data_dir)

    steps = []
    times = []
    mean_r = []
    mean_r2 = []
    ngrain = []

    for f in files:
        label, step, time = load_label_and_meta(f, id_slot=id_slot)

        if label.shape[2] != 1:
            raise ValueError(
                f"{f.name}: expected km=1 for 2D analysis, got shape {label.shape}"
            )

        label_2d = label[:, :, 0]
        radii = grain_radii_from_label_2d(label_2d, dx=dx, min_pixels=min_pixels)

        if radii.size == 0:
            continue

        steps.append(step if step is not None else len(steps))
        times.append(time if time is not None else float(len(times)))
        mean_r.append(np.mean(radii))
        mean_r2.append(np.mean(radii ** 2))
        ngrain.append(len(radii))

    if len(times) == 0:
        raise RuntimeError("No valid snapshots were analyzed.")

    return {
        "step": np.array(steps, dtype=np.int64),
        "time": np.array(times, dtype=np.float64),
        "mean_r": np.array(mean_r, dtype=np.float64),
        "mean_r2": np.array(mean_r2, dtype=np.float64),
        "ngrain": np.array(ngrain, dtype=np.int64),
    }


# ============================================================
# log-log slope fit
# ============================================================
def fit_log_slope(x, y):
    mask = (x > 0.0) & (y > 0.0)
    if np.count_nonzero(mask) < 2:
        return None, None, None

    lx = np.log(x[mask])
    ly = np.log(y[mask])

    coef = np.polyfit(lx, ly, 1)
    slope = coef[0]
    intercept = coef[1]

    xfit = np.linspace(lx.min(), lx.max(), 200)
    yfit = slope * xfit + intercept

    return slope, xfit, yfit


# ============================================================
# 그림 저장
# ============================================================
def save_plots(result, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    t = result["time"]
    R = result["mean_r"]
    R2 = result["mean_r2"]
    N = result["ngrain"]

    # 1) Mean grain size vs time
    plt.figure(figsize=(10, 6))
    plt.plot(t, R, marker="o")
    plt.xlabel("time")
    plt.ylabel("mean grain radius [m]")
    plt.title("Mean grain size vs time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "mean_grain_size_vs_time.png", dpi=200)
    plt.close()

    # 2) Mean grain size squared vs time
    plt.figure(figsize=(10, 6))
    plt.plot(t, R2, marker="o")
    plt.xlabel("time")
    plt.ylabel("mean grain radius squared [m^2]")
    plt.title("Mean grain size squared vs time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "mean_grain_size_squared_vs_time.png", dpi=200)
    plt.close()

    # 3) Grain count vs time
    plt.figure(figsize=(10, 6))
    plt.plot(t, N, marker="o")
    plt.xlabel("time")
    plt.ylabel("number of grains")
    plt.title("Grain count vs time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "grain_count_vs_time.png", dpi=200)
    plt.close()

    # 4) log(mean grain size) vs log(time)
    slope_R, xfit_R, yfit_R = fit_log_slope(t, R)
    mask_R = (t > 0.0) & (R > 0.0)

    plt.figure(figsize=(10, 6))
    plt.plot(np.log(t[mask_R]), np.log(R[mask_R]), marker="o", linestyle="-", label="data")
    if slope_R is not None:
        plt.plot(xfit_R, yfit_R, linestyle="--", label=f"fit slope = {slope_R:.4f}")
    plt.xlabel("log(time)")
    plt.ylabel("log(mean grain radius)")
    plt.title("log(mean grain size) vs log(time)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "log_mean_grain_size_vs_log_time.png", dpi=200)
    plt.close()

    # 5) log(grain count) vs log(time)
    slope_N, xfit_N, yfit_N = fit_log_slope(t, N.astype(np.float64))
    mask_N = (t > 0.0) & (N > 0)

    plt.figure(figsize=(10, 6))
    plt.plot(np.log(t[mask_N]), np.log(N[mask_N]), marker="o", linestyle="-", label="data")
    if slope_N is not None:
        plt.plot(xfit_N, yfit_N, linestyle="--", label=f"fit slope = {slope_N:.4f}")
    plt.xlabel("log(time)")
    plt.ylabel("log(number of grains)")
    plt.title("log(grain count) vs log(time)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "log_grain_count_vs_log_time.png", dpi=200)
    plt.close()

    # 6) csv 저장
    arr = np.column_stack([result["step"], t, R, R2, N])
    header = "step,time,mean_radius_m,mean_radius_squared_m2,grain_count"
    np.savetxt(out_dir / "grain_growth_summary.csv", arr, delimiter=",", header=header, comments="")

    return slope_R, slope_N


# ============================================================
# main
# ============================================================
def main():
    base = Path(__file__).resolve().parent

    input_path = base / "input.txt"
    data_dir = base / "p_out"
    out_dir = base / "post_out"

    p = read_input(input_path)

    if p["km"] != 1:
        raise ValueError("This post-processing script is for 2D data only (km = 1).")

    dx = p["dx"]

    result = analyze_growth_2d(
        data_dir=data_dir,
        dx=dx,
        id_slot=0,
        min_pixels=1,
    )

    slope_R, slope_N = save_plots(result, out_dir)

    print("Post-processing finished.")
    print(f"input file : {input_path}")
    print(f"data dir   : {data_dir}")
    print(f"output dir : {out_dir}")
    print(f"n snapshots: {len(result['time'])}")
    print(f"time range : {result['time'][0]:.6e} ~ {result['time'][-1]:.6e}")
    print(f"mean R     : {result['mean_r'][0]:.6e} -> {result['mean_r'][-1]:.6e}")
    print(f"grain count: {result['ngrain'][0]} -> {result['ngrain'][-1]}")
    if slope_R is not None:
        print(f"log(mean R) vs log(t) slope = {slope_R:.6f}")
    if slope_N is not None:
        print(f"log(N) vs log(t) slope      = {slope_N:.6f}")


if __name__ == "__main__":
    main()
