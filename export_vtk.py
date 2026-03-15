import argparse
import configparser
from pathlib import Path

import numpy as np


EMPTY_ID = -1


def read_grid_spacing(input_path="input.txt"):
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.read(input_path, encoding="utf-8")

    dx = cfg.getfloat("grid", "dx")
    dy = cfg.getfloat("grid", "dy")
    dz = cfg.getfloat("grid", "dz")

    return dx, dy, dz


def load_snapshot(npz_path):
    data = np.load(npz_path)

    phi = data["phi"]   # shape: (im, jm, km, nmax)
    gid = data["id"]    # shape: (im, jm, km, nmax)
    step = int(data["step"])
    time = float(data["time"])

    return phi, gid, step, time


def build_voxel_fields(phi, gid):
    im, jm, km, nmax = phi.shape

    grain_id = np.zeros((im, jm, km), dtype=np.int32)
    phi_max = np.zeros((im, jm, km), dtype=np.float32)
    nactive = np.zeros((im, jm, km), dtype=np.int32)

    for i in range(im):
        for j in range(jm):
            for k in range(km):
                best_phi = -1.0
                best_gid = 0
                active_count = 0

                for s in range(nmax):
                    g = gid[i, j, k, s]
                    if g == EMPTY_ID:
                        continue

                    p = phi[i, j, k, s]
                    active_count += 1

                    if p > best_phi:
                        best_phi = p
                        best_gid = g + 1   # 0 reserved for empty

                grain_id[i, j, k] = best_gid
                phi_max[i, j, k] = 0.0 if best_phi < 0.0 else best_phi
                nactive[i, j, k] = active_count

    return grain_id, phi_max, nactive


def write_structured_points_vtk(
    vtk_path,
    grain_id,
    phi_max,
    nactive,
    dx,
    dy,
    dz,
    step,
    time,
):
    im, jm, km = grain_id.shape
    ncell = im * jm * km

    # VTK STRUCTURED_POINTS with CELL_DATA:
    # cell count = (DIMENSIONS-1 product), so dimensions must be +1
    dimx = im + 1
    dimy = jm + 1
    dimz = km + 1

    with open(vtk_path, "wb") as f:
        header = [
            "# vtk DataFile Version 3.0\n",
            f"PFM export step={step} time={time:.8e}\n",
            "BINARY\n",
            "DATASET STRUCTURED_POINTS\n",
            f"DIMENSIONS {dimx} {dimy} {dimz}\n",
            "ORIGIN 0.0 0.0 0.0\n",
            f"SPACING {dx:.16e} {dy:.16e} {dz:.16e}\n",
            f"CELL_DATA {ncell}\n",
        ]
        f.write("".join(header).encode("ascii"))

        # grain_id
        f.write(b"SCALARS grain_id int 1\n")
        f.write(b"LOOKUP_TABLE default\n")
        f.write(grain_id.astype(">i4").tobytes(order="F"))
        f.write(b"\n")

        # phi_max
        f.write(b"SCALARS phi_max float 1\n")
        f.write(b"LOOKUP_TABLE default\n")
        f.write(phi_max.astype(">f4").tobytes(order="F"))
        f.write(b"\n")

        # nactive
        f.write(b"SCALARS nactive int 1\n")
        f.write(b"LOOKUP_TABLE default\n")
        f.write(nactive.astype(">i4").tobytes(order="F"))
        f.write(b"\n")


def convert_one(npz_path, out_dir, dx, dy, dz):
    phi, gid, step, time = load_snapshot(npz_path)
    grain_id, phi_max, nactive = build_voxel_fields(phi, gid)

    vtk_name = npz_path.stem + ".vtk"
    vtk_path = out_dir / vtk_name

    write_structured_points_vtk(
        vtk_path=vtk_path,
        grain_id=grain_id,
        phi_max=phi_max,
        nactive=nactive,
        dx=dx,
        dy=dy,
        dz=dz,
        step=step,
        time=time,
    )

    print(f"Converted: {npz_path.name} -> {vtk_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Convert PFM npz output to ParaView vtk.")
    parser.add_argument("--in-dir", default="p_out", help="directory containing p_*.npz")
    parser.add_argument("--out-dir", default="pvout", help="directory for vtk output")
    parser.add_argument("--input-file", default="input.txt", help="input file for dx, dy, dz")
    parser.add_argument("--file", default=None, help="convert only one file, e.g. p_00001000.npz")

    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dx, dy, dz = read_grid_spacing(args.input_file)

    if args.file is not None:
        npz_path = in_dir / args.file
        if not npz_path.exists():
            raise FileNotFoundError(f"Not found: {npz_path}")
        convert_one(npz_path, out_dir, dx, dy, dz)
    else:
        files = sorted(in_dir.glob("p_*.npz"))
        if not files:
            raise FileNotFoundError(f"No p_*.npz files found in {in_dir}")
        for npz_path in files:
            convert_one(npz_path, out_dir, dx, dy, dz)


if __name__ == "__main__":
    main()
