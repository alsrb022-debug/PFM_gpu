import argparse
import configparser
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Input / Snapshot
# ============================================================
def read_input(path="input.txt"):
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.read(path, encoding="utf-8")

    data = {}

    data["im"] = cfg.getint("domain", "im")
    data["jm"] = cfg.getint("domain", "jm")
    data["km"] = cfg.getint("domain", "km")

    data["dx"] = cfg.getfloat("grid", "dx")
    data["dy"] = cfg.getfloat("grid", "dy")
    data["dz"] = cfg.getfloat("grid", "dz")

    data["nmax"] = cfg.getint("pf", "nmax")
    data["emob"] = cfg.getfloat("pf", "emob")
    data["sigma"] = cfg.getfloat("pf", "sigma")
    data["xi_in"] = cfg.getfloat("pf", "xi_in")
    data["tifac"] = cfg.getfloat("pf", "tifac")
    data["pss"] = cfg.getfloat("pf", "pss")

    data["nstep"] = cfg.getint("time", "nstep")
    data["nout"] = cfg.getint("time", "nout")

    init_type = cfg.get("init", "type", fallback="voronoi").strip().lower()
    data["init_type"] = init_type

    data["seed"] = cfg.getint("init", "seed", fallback=1234)
    data["nphase"] = cfg.getint("init", "nphase", fallback=0)
    data["radius"] = cfg.getfloat("init", "radius", fallback=-1.0)
    data["cx"] = cfg.getfloat("init", "cx", fallback=-1.0)
    data["cy"] = cfg.getfloat("init", "cy", fallback=-1.0)
    data["cz"] = cfg.getfloat("init", "cz", fallback=-1.0)
    data["ngrain"] = cfg.getint("init", "ngrain", fallback=0)

    data["bc_type"] = cfg.get("bc", "type", fallback="pbc").strip().lower()

    return data


def load_snapshot(path):
    data = np.load(path)

    phi = data["phi"]     # shape = (im, jm, km, nmax)
    gid = data["id"]      # shape = (im, jm, km, nmax)
    step = int(data["step"])
    time = float(data["time"])

    return phi, gid, step, time


def detect_ndim_from_shape(phi):
    im, jm, km, _ = phi.shape

    ndim = 0
    if im > 1:
        ndim += 1
    if jm > 1:
        ndim += 1
    if km > 1:
        ndim += 1

    return ndim


# ============================================================
# PF coefficients
# ============================================================
def compute_pf_coefficients(params):
    xi = params["xi_in"] * params["dx"]
    www = 2.0 * params["sigma"] / xi
    cep = 4.0 / math.pi * math.sqrt(xi * params["sigma"])
    emm = params["emob"] * params["sigma"] / (cep ** 2.0)

    return xi, www, cep, emm


# ============================================================
# Common field processing
# ============================================================
def dominant_gid_map(phi, gid):
    """
    Return dominant actual gid map using slot-wise phi and id.
    phi shape: (im, jm, km, nmax)
    gid shape: (im, jm, km, nmax)
    output   : (im, jm, km)
    """
    dom_slot = np.argmax(phi, axis=3)
    gmap = np.take_along_axis(gid, dom_slot[..., None], axis=3)[..., 0]
    return gmap.astype(np.int32)


def dominant_phi_map(phi):
    return np.max(phi, axis=3)


def boundary_map_2d(gmap):
    """
    boundary-only map from 2D gid map
    output: 1 boundary, 0 interior
    """
    im, jm = gmap.shape
    bmap = np.zeros((im, jm), dtype=np.uint8)

    for i in range(im):
        for j in range(jm):
            c = gmap[i, j]

            il = im - 1 if i == 0 else i - 1
            ir = 0 if i == im - 1 else i + 1
            jl = jm - 1 if j == 0 else j - 1
            jr = 0 if j == jm - 1 else j + 1

            if (
                gmap[il, j] != c
                or gmap[ir, j] != c
                or gmap[i, jl] != c
                or gmap[i, jr] != c
            ):
                bmap[i, j] = 1

    return bmap


def boundary_image_from_map(bmap):
    img = np.full(bmap.shape, 255, dtype=np.uint8)
    img[bmap == 1] = 0
    return img


def extract_phi_profile_1d(phi, gid, pss=0.0):
    """
    Extract 1D phi profiles by actual gid.
    Returns dict: actual_gid -> profile array
    phi shape: (im, jm, km, nmax)
    gid shape: (im, jm, km, nmax)
    """
    im, jm, km, nmax = phi.shape
    if not (jm == 1 and km == 1):
        raise ValueError("1D profile extraction requires jm=1 and km=1")

    gid_line_all = gid[:, 0, 0, :]
    valid_gid = gid_line_all[gid_line_all >= 0]

    if valid_gid.size == 0:
        return {}

    uniq_gid = np.unique(valid_gid)
    profiles = {}

    for g in uniq_gid:
        prof = np.zeros(im, dtype=np.float64)

        for i in range(im):
            val = 0.0
            for s in range(nmax):
                if gid[i, 0, 0, s] == g:
                    val += phi[i, 0, 0, s]
            prof[i] = val

        if np.max(prof) > pss:
            profiles[int(g)] = prof

    return profiles


# ============================================================
# Grain size statistics (Voronoi only, 2D/3D)
# ============================================================
def equivalent_radii_from_gmap(gmap, dx, dy, dz, min_cells):
    """
    Compute equivalent grain radius from dominant gid map.

    2D:
        A_g = N_g * dx * dy
        R_g = sqrt(A_g / pi)

    3D:
        V_g = N_g * dx * dy * dz
        R_g = (3V_g / 4pi)^(1/3)
    """
    ndim = 0
    if gmap.shape[0] > 1:
        ndim += 1
    if gmap.shape[1] > 1:
        ndim += 1
    if gmap.shape[2] > 1:
        ndim += 1

    gids, counts = np.unique(gmap, return_counts=True)

    # safety: ignore invalid gid if any
    valid = gids >= 0
    gids = gids[valid]
    counts = counts[valid]

    # remove too small grains
    keep = counts >= min_cells
    gids = gids[keep]
    counts = counts[keep]

    if gids.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int32)

    if ndim == 2:
        areas = counts.astype(np.float64) * dx * dy
        radii = np.sqrt(areas / math.pi)
    elif ndim == 3:
        volumes = counts.astype(np.float64) * dx * dy * dz
        radii = (3.0 * volumes / (4.0 * math.pi)) ** (1.0 / 3.0)
    else:
        raise ValueError(f"equivalent_radii_from_gmap supports only 2D/3D, got ndim={ndim}")

    return radii, gids.astype(np.int32)


def compute_mean_grain_size(phi, gid, params):
    """
    Compute mean equivalent grain radius for 2D/3D.
    """
    ndim = detect_ndim_from_shape(phi)
    if ndim not in (2, 3):
        raise ValueError(f"Mean grain size is intended for 2D/3D only, got ndim={ndim}")

    gmap = dominant_gid_map(phi, gid)

    if ndim == 2:
        min_cells = 4
    else:
        min_cells = 8

    radii, gids = equivalent_radii_from_gmap(
        gmap,
        params["dx"],
        params["dy"],
        params["dz"],
        min_cells=min_cells,
    )

    if radii.size == 0:
        return {
            "ngrain": 0,
            "mean_r": 0.0,
            "mean_r2": 0.0,
        }

    return {
        "ngrain": int(radii.size),
        "mean_r": float(np.mean(radii)),
        "mean_r2": float(np.mean(radii * radii)),
    }


def process_mean_grain_size(all_files, params, out_dir):
    times = []
    steps = []
    mean_r = []
    mean_r2 = []
    ngrain = []

    for snap_path in all_files:
        phi, gid, step, time = load_snapshot(snap_path)
        stat = compute_mean_grain_size(phi, gid, params)

        steps.append(step)
        times.append(time)
        mean_r.append(stat["mean_r"])
        mean_r2.append(stat["mean_r2"])
        ngrain.append(stat["ngrain"])

    steps = np.asarray(steps, dtype=np.int32)
    times = np.asarray(times, dtype=np.float64)
    mean_r = np.asarray(mean_r, dtype=np.float64)
    mean_r2 = np.asarray(mean_r2, dtype=np.float64)
    ngrain = np.asarray(ngrain, dtype=np.int32)

    np.savetxt(
        out_dir / "mean_grain_size.csv",
        np.column_stack([steps, times, ngrain, mean_r, mean_r2]),
        delimiter=",",
        header="step,time,ngrain,mean_radius,mean_radius_sq",
        comments="",
    )

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(times, mean_r, "o-")
    ax.set_xlabel("time")
    ax.set_ylabel("mean grain radius [m]")
    ax.set_title("Mean grain size vs time")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_grain_size_vs_time.png", dpi=180)
    plt.close()

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(times, mean_r2, "o-")
    ax.set_xlabel("time")
    ax.set_ylabel("mean grain radius squared [m^2]")
    ax.set_title("Mean grain size squared vs time")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_grain_size_sq_vs_time.png", dpi=180)
    plt.close()

    print("Processed mean grain size statistics.")
    

# ============================================================
# Visualization
# ============================================================
def visualize_1d(phi, gid, step, time, out_dir, stem, pss):
    profiles = extract_phi_profile_1d(phi, gid, pss=pss)
    im = phi.shape[0]

    x = np.arange(1, im + 1, dtype=np.float64)

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)

    for g, prof in profiles.items():
        ax.plot(x, prof, label=f"gid={g}")

    ax.set_xlabel("grid index")
    ax.set_ylabel("phi")
    ax.set_title(f"1D profile  step={step}  time={time:.4e}")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_plot.png", dpi=180)
    plt.close()


def visualize_2d(phi, gid, step, time, out_dir, stem):
    gmap = dominant_gid_map(phi, gid)[:, :, 0]
    bmap = boundary_map_2d(gmap)
    img = boundary_image_from_map(bmap)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img.T, cmap="gray", origin="lower", interpolation="nearest")
    ax.set_title(f"2D boundary  step={step}  time={time:.4e}")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_plot.png", dpi=180)
    plt.close()


def visualize_3d(phi, gid, step, time, out_dir, stem):
    gmap = dominant_gid_map(phi, gid)

    im, jm, km = gmap.shape
    ic = im // 2
    jc = jm // 2
    kc = km // 2

    xy = gmap[:, :, kc]
    yz = gmap[ic, :, :]
    xz = gmap[:, jc, :]

    bxy = boundary_map_2d(xy)
    byz = boundary_map_2d(yz)
    bxz = boundary_map_2d(xz)

    img_xy = boundary_image_from_map(bxy)
    img_yz = boundary_image_from_map(byz)
    img_xz = boundary_image_from_map(bxz)

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img_xy.T, cmap="gray", origin="lower", interpolation="nearest")
    ax1.set_title("XY center")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(img_yz.T, cmap="gray", origin="lower", interpolation="nearest")
    ax2.set_title("YZ center")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(img_xz.T, cmap="gray", origin="lower", interpolation="nearest")
    ax3.set_title("XZ center")
    ax3.axis("off")

    fig.suptitle(f"3D boundary slices  step={step}  time={time:.4e}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_plot.png", dpi=180)
    plt.close()


def visualize_pf(phi, gid, step, time, out_dir, stem, pss):
    ndim = detect_ndim_from_shape(phi)

    if ndim == 1:
        visualize_1d(phi, gid, step, time, out_dir, stem, pss)
    elif ndim == 2:
        visualize_2d(phi, gid, step, time, out_dir, stem)
    elif ndim == 3:
        visualize_3d(phi, gid, step, time, out_dir, stem)
    else:
        raise ValueError(f"Unsupported ndim = {ndim}")


# ============================================================
# PF benchmark
# ============================================================
def benchmark_plane(phi, gid, params, step, time, out_dir, stem):
    profiles = extract_phi_profile_1d(phi, gid, pss=params["pss"])

    if 0 not in profiles:
        print("benchmark_plane: gid=0 not found. skipped.")
        return

    im = phi.shape[0]
    dx = params["dx"]

    eta = params["xi_in"] * dx

    x = np.arange(1, im + 1, dtype=np.float64) * dx

    ic = (im + 1) // 2
    x0 = (ic - 0.5) * dx

    phi_sim = profiles[0]
    phi_ana = np.zeros(im, dtype=np.float64)

    for n in range(im):
        xx = x[n]

        if xx <= x0 - eta:
            phi_ana[n] = 1.0
        elif xx >= x0 + eta:
            phi_ana[n] = 0.0
        else:
            phi_ana[n] = 0.5 * (
                1.0 - math.sin(math.pi * (xx - x0) / (2.0 * eta))
            )

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x, phi_sim, label="simulation")
    ax.plot(x, phi_ana, "--", label="analytic")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("phi")
    ax.set_title(f"Plane benchmark  step={step}  time={time:.4e}")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_benchmark.png", dpi=180)
    plt.close()


def measure_circle_radius(phi, gid, dx, dy, target_gid=1):
    gmap = dominant_gid_map(phi, gid)[:, :, 0]
    area = np.count_nonzero(gmap == target_gid) * dx * dy
    if area <= 0.0:
        return 0.0
    return math.sqrt(area / math.pi)


def measure_sphere_radius(phi, gid, dx, dy, dz, target_gid=1):
    gmap = dominant_gid_map(phi, gid)
    volume = np.count_nonzero(gmap == target_gid) * dx * dy * dz
    if volume <= 0.0:
        return 0.0
    return (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)


def benchmark_circle(all_files, params, out_dir):
    dx = params["dx"]
    dy = params["dy"]
    sigma = params["sigma"]
    r0 = params["radius"] * dx
    emob = params["emob"]

    times = []
    rsim = []
    rana = []

    for f in all_files:
        phi, gid, step, time = load_snapshot(f)
        r = measure_circle_radius(phi, gid, dx, dy, target_gid=1)

        val = r0 * r0 - 2.0 * emob * sigma * time
        ra = math.sqrt(val) if val > 0.0 else 0.0

        times.append(time)
        rsim.append(r)
        rana.append(ra)

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(times, rsim, "o-", label="simulation")
    ax.plot(times, rana, "--", label="analytic")

    ax.set_xlabel("time")
    ax.set_ylabel("radius [m]")
    ax.set_title("Circle benchmark")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_circle.png", dpi=180)
    plt.close()


def benchmark_sphere(all_files, params, out_dir):
    dx = params["dx"]
    dy = params["dy"]
    dz = params["dz"]
    sigma = params["sigma"]
    r0 = params["radius"] * dx
    emob = params["emob"]

    times = []
    rsim = []
    rana = []

    for f in all_files:
        phi, gid, step, time = load_snapshot(f)
        r = measure_sphere_radius(phi, gid, dx, dy, dz, target_gid=1)

        val = r0 * r0 - 4.0 * emob * sigma * time
        ra = math.sqrt(val) if val > 0.0 else 0.0

        times.append(time)
        rsim.append(r)
        rana.append(ra)

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(times, rsim, "o-", label="simulation")
    ax.plot(times, rana, "--", label="analytic")

    ax.set_xlabel("time")
    ax.set_ylabel("radius [m]")
    ax.set_title("Sphere benchmark")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_sphere.png", dpi=180)
    plt.close()


def benchmark_pf(single_phi, single_gid, step, time, params, all_files, out_dir, stem):
    init_type = params["init_type"]

    if init_type == "plane":
        benchmark_plane(single_phi, single_gid, params, step, time, out_dir, stem)

    elif init_type == "circle":
        benchmark_circle(all_files, params, out_dir)

    elif init_type == "sphere":
        benchmark_sphere(all_files, params, out_dir)

    else:
        print(f"benchmark_pf: no analytic benchmark for init type '{init_type}'")


# ============================================================
# Process helpers
# ============================================================
def process_one_snapshot(snap_path, out_dir, pss):
    phi, gid, step, time = load_snapshot(snap_path)
    stem = snap_path.stem
    visualize_pf(phi, gid, step, time, out_dir, stem, pss)
    print(f"Processed plot: {snap_path.name}")


def process_all_snapshots(all_files, out_dir, pss):
    for snap_path in all_files:
        process_one_snapshot(snap_path, out_dir, pss)


def run_benchmark_if_needed(all_files, params, out_dir):
    if not all_files:
        return

    init_type = params["init_type"]

    if init_type in ("plane", "circle", "sphere"):
        last_file = all_files[-1]
        phi, gid, step, time = load_snapshot(last_file)
        stem = last_file.stem
        benchmark_pf(phi, gid, step, time, params, all_files, out_dir, stem)
        print(f"Processed benchmark for init type: {init_type}")
    else:
        print(f"No analytic benchmark for init type: {init_type}")


# ============================================================
# Main process
# ============================================================
def process_pf(input_file="input.txt", in_dir="p_out", out_dir="fig_pf", snapshot=None):
    params = read_input(input_file)

    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if snapshot is not None:
        snap_path = in_dir / snapshot
        if not snap_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snap_path}")

        process_one_snapshot(snap_path, out_dir, params["pss"])

        if params["init_type"] == "plane":
            phi, gid, step, time = load_snapshot(snap_path)
            benchmark_plane(phi, gid, params, step, time, out_dir, snap_path.stem)
            print("Processed plane benchmark (single snapshot mode).")

        print(f"Figures saved to: {out_dir}")
        return

    all_files = sorted(in_dir.glob("p_*.npz"))
    if not all_files:
        raise FileNotFoundError(f"No p_*.npz files found in {in_dir}")

    process_all_snapshots(all_files, out_dir, params["pss"])
    run_benchmark_if_needed(all_files, params, out_dir)

    if params["init_type"] == "voronoi":
        ndim = detect_ndim_from_shape(load_snapshot(all_files[0])[0])
        if ndim in (2, 3):
            process_mean_grain_size(all_files, params, out_dir)

    print(f"All figures saved to: {out_dir}")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="PF postprocess: visualization + benchmark")
    parser.add_argument("--input-file", default="input.txt", help="input file path")
    parser.add_argument("--in-dir", default="p_out", help="directory containing p_*.npz")
    parser.add_argument("--out-dir", default="fig_pf", help="directory for output figures")
    parser.add_argument("--snapshot", default=None, help="process only one snapshot if given")

    args = parser.parse_args()

    process_pf(
        input_file=args.input_file,
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        snapshot=args.snapshot,
    )


if __name__ == "__main__":
    main()
