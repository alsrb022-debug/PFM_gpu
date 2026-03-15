import math
import time

import numpy as np

from init_benchmark import (
    init_plane,
    init_triple,
    init_circle,
    init_sphere,
    init_neighbors,
)
from init_voronoi import init_voronoi
from input_params import prepare_simulation

try:
    from numba import cuda, float32, int16
    CUDA_OK = cuda.is_available()
except Exception:
    cuda = None
    float32 = None
    int16 = None
    CUDA_OK = False


# ============================================================
# Main
# ============================================================
def run(input_path="input.txt", out_dir="pout"):
    if not CUDA_OK:
        raise RuntimeError("CUDA is not available. Numba CUDA 환경을 먼저 확인해.")

    params = prepare_simulation(input_path, out_dir)
    phiO, phiN, idO, idN = allocate_phase_fields(params)

    initialize_fields_from_params(phiN, idN, params)
    copy_initial_state(phiO, phiN, idO, idN)

    solver_kernel = build_solver_step_kernel(params["nmax"])
    bc_kernels = build_apply_bc_kernels()

    phiO_d = cuda.to_device(phiO)
    phiN_d = cuda.device_array_like(phiO)

    idO_d = cuda.to_device(idO)
    idN_d = cuda.device_array_like(idO)

    print_simulation_info(params)
    print(f"reference dt = {params['dt']:.6e}")
    print("CUDA enabled = True")

    t0 = time.perf_counter()

    sim_time = 0.0
    dt = params["dt"]

    for step in range(params["nstep"] + 1):
        if step > 0:
            # phi / id 둘 다 ghost 갱신 필요
            apply_bc_cuda(phiO_d, params, bc_kernels)
            apply_bc_cuda(idO_d, params, bc_kernels)

            solver_step_cuda(phiO_d, phiN_d, idO_d, idN_d, params, solver_kernel)
            cuda.synchronize()

            phiO_d, phiN_d = phiN_d, phiO_d
            idO_d, idN_d = idN_d, idO_d

        if step % params["nout"] == 0:
            phi_save_host = phiO_d.copy_to_host()
            id_save_host = idO_d.copy_to_host()

            write_snapshot(
                phi_save_host,
                id_save_host,
                params["im"],
                params["jm"],
                params["km"],
                params["out_dir"],
                step,
                sim_time,
            )

            elapsed = time.perf_counter() - t0
            print(f"step = {step:8d}, time = {sim_time:.6e}, gpu_time = {elapsed:.3f} s")

        sim_time += dt


# ============================================================
# CUDA kernel factory
# phi shape = (i, j, k, slot)
# id  shape = (i, j, k, slot)
# ============================================================
def build_solver_step_kernel(nmax_fixed):
    if not CUDA_OK:
        return None

    @cuda.jit
    def solver_step_kernel(
        phiO, phiN,
        idO, idN,
        im, jm, km,
        dx, dy, dz,
        dt, pss,
        eps, omg, mob
    ):
        ii, jj, kk = cuda.grid(3)

        if ii >= im or jj >= jm or kk >= km:
            return

        i = ii + 1
        j = jj + 1
        k = kk + 1

        dx2 = dx * dx
        dy2 = dy * dy
        dz2 = dz * dz

        act_id = cuda.local.array(nmax_fixed, int16)

        phi_c_arr = cuda.local.array(nmax_fixed, float32)
        lap = cuda.local.array(nmax_fixed, float32)
        dF = cuda.local.array(nmax_fixed, float32)
        rhs = cuda.local.array(nmax_fixed, float32)
        phi_tmp = cuda.local.array(nmax_fixed, float32)

        S = 0

        for s in range(nmax_fixed):
            act_id[s] = -1
            phi_c_arr[s] = 0.0
            lap[s] = 0.0
            dF[s] = 0.0
            rhs[s] = 0.0
            phi_tmp[s] = 0.0

        # ----------------------------------------------------
        # 0) active phase ids collection from self + 6 neighbors
        # ----------------------------------------------------
        for s in range(nmax_fixed):
            pid = idO[i, j, k, s]
            if pid < 0:
                continue

            exist = False
            for t in range(S):
                if act_id[t] == pid:
                    exist = True
                    break

            if (not exist) and (S < nmax_fixed):
                act_id[S] = pid
                S += 1

        for s in range(nmax_fixed):
            pid = idO[i - 1, j, k, s]
            if pid < 0:
                continue

            exist = False
            for t in range(S):
                if act_id[t] == pid:
                    exist = True
                    break

            if (not exist) and (S < nmax_fixed):
                act_id[S] = pid
                S += 1

        for s in range(nmax_fixed):
            pid = idO[i + 1, j, k, s]
            if pid < 0:
                continue

            exist = False
            for t in range(S):
                if act_id[t] == pid:
                    exist = True
                    break

            if (not exist) and (S < nmax_fixed):
                act_id[S] = pid
                S += 1

        for s in range(nmax_fixed):
            pid = idO[i, j - 1, k, s]
            if pid < 0:
                continue

            exist = False
            for t in range(S):
                if act_id[t] == pid:
                    exist = True
                    break

            if (not exist) and (S < nmax_fixed):
                act_id[S] = pid
                S += 1

        for s in range(nmax_fixed):
            pid = idO[i, j + 1, k, s]
            if pid < 0:
                continue

            exist = False
            for t in range(S):
                if act_id[t] == pid:
                    exist = True
                    break

            if (not exist) and (S < nmax_fixed):
                act_id[S] = pid
                S += 1

        for s in range(nmax_fixed):
            pid = idO[i, j, k - 1, s]
            if pid < 0:
                continue

            exist = False
            for t in range(S):
                if act_id[t] == pid:
                    exist = True
                    break

            if (not exist) and (S < nmax_fixed):
                act_id[S] = pid
                S += 1

        for s in range(nmax_fixed):
            pid = idO[i, j, k + 1, s]
            if pid < 0:
                continue

            exist = False
            for t in range(S):
                if act_id[t] == pid:
                    exist = True
                    break

            if (not exist) and (S < nmax_fixed):
                act_id[S] = pid
                S += 1

        if S <= 0:
            for s in range(nmax_fixed):
                phiN[i, j, k, s] = 0.0
                idN[i, j, k, s] = -1
            return

        # ----------------------------------------------------
        # 1) center phi + lap[a] for active phases
        #    one slot scan finds all 7 positions
        # ----------------------------------------------------
        for a in range(S):
            pid = act_id[a]

            sc = -1
            sxm = -1
            sxp = -1
            sym = -1
            syp = -1
            szm = -1
            szp = -1

            for s in range(nmax_fixed):
                if sc < 0 and idO[i, j, k, s] == pid:
                    sc = s
                if sxm < 0 and idO[i - 1, j, k, s] == pid:
                    sxm = s
                if sxp < 0 and idO[i + 1, j, k, s] == pid:
                    sxp = s
                if sym < 0 and idO[i, j - 1, k, s] == pid:
                    sym = s
                if syp < 0 and idO[i, j + 1, k, s] == pid:
                    syp = s
                if szm < 0 and idO[i, j, k - 1, s] == pid:
                    szm = s
                if szp < 0 and idO[i, j, k + 1, s] == pid:
                    szp = s

            phi_c = 0.0
            phi_xm = 0.0
            phi_xp = 0.0
            phi_ym = 0.0
            phi_yp = 0.0
            phi_zm = 0.0
            phi_zp = 0.0

            if sc >= 0:
                phi_c = phiO[i, j, k, sc]
            if sxm >= 0:
                phi_xm = phiO[i - 1, j, k, sxm]
            if sxp >= 0:
                phi_xp = phiO[i + 1, j, k, sxp]
            if sym >= 0:
                phi_ym = phiO[i, j - 1, k, sym]
            if syp >= 0:
                phi_yp = phiO[i, j + 1, k, syp]
            if szm >= 0:
                phi_zm = phiO[i, j, k - 1, szm]
            if szp >= 0:
                phi_zp = phiO[i, j, k + 1, szp]

            phi_c_arr[a] = phi_c
            lap[a] = (
                (phi_xp - 2.0 * phi_c + phi_xm) / dx2 +
                (phi_yp - 2.0 * phi_c + phi_ym) / dy2 +
                (phi_zp - 2.0 * phi_c + phi_zm) / dz2
            )

        if S <= 1:
            for s in range(nmax_fixed):
                phiN[i, j, k, s] = phiO[i, j, k, s]
                idN[i, j, k, s] = idO[i, j, k, s]
            return

        # ----------------------------------------------------
        # 2) dF[a] using active phases only
        # ----------------------------------------------------
        for a in range(S):
            val = 0.0
            for b in range(S):
                if b == a:
                    continue
                val += 0.5 * (eps * eps) * lap[b] + omg * phi_c_arr[b]
            dF[a] = val

        # ----------------------------------------------------
        # 3) unique active pair loop (a < b)
        # ----------------------------------------------------
        for a in range(S):
            rhs[a] = 0.0

        for a in range(S):
            for b in range(a + 1, S):
                term = mob * (dF[a] - dF[b])
                rhs[a] += term
                rhs[b] -= term

        # ----------------------------------------------------
        # 4) update
        # ----------------------------------------------------
        coef = -2.0 / S

        for a in range(S):
            val = phi_c_arr[a] + dt * coef * rhs[a]
            if val < 0.0:
                val = 0.0
            phi_tmp[a] = val

        # ----------------------------------------------------
        # 5) compact writeback with stronger threshold
        # ----------------------------------------------------
        for s in range(nmax_fixed):
            phiN[i, j, k, s] = 0.0
            idN[i, j, k, s] = -1

        pss_kill = 1 * pss
        sum_survive = 0.0

        for a in range(S):
            if phi_tmp[a] <= pss_kill:
                phi_tmp[a] = 0.0
            else:
                sum_survive += phi_tmp[a]

        inv = 1.0 / sum_survive
        out_s = 0

        for a in range(S):
            val = phi_tmp[a]
            if val > 0.0:
                phiN[i, j, k, out_s] = val * inv
                idN[i, j, k, out_s] = act_id[a]
                out_s += 1

    return solver_step_kernel

# ============================================================
# CUDA PBC kernels
# works for both phi(float32) and id(int16)
# ============================================================
def build_apply_bc_kernels():
    if not CUDA_OK:
        return None, None, None

    @cuda.jit
    def apply_bc_z_kernel(arr, im, jm, km, nmax_in):
        i0, j0, s = cuda.grid(3)

        if i0 >= (im + 2) or j0 >= (jm + 2) or s >= nmax_in:
            return

        arr[i0, j0, 0,      s] = arr[i0, j0, km, s]
        arr[i0, j0, km + 1, s] = arr[i0, j0, 1,  s]

    @cuda.jit
    def apply_bc_y_kernel(arr, im, jm, km, nmax_in):
        i0, k0, s = cuda.grid(3)

        if i0 >= (im + 2) or k0 >= (km + 2) or s >= nmax_in:
            return

        arr[i0, 0,      k0, s] = arr[i0, jm, k0, s]
        arr[i0, jm + 1, k0, s] = arr[i0, 1,  k0, s]

    @cuda.jit
    def apply_bc_x_kernel(arr, im, jm, km, nmax_in):
        j0, k0, s = cuda.grid(3)

        if j0 >= (jm + 2) or k0 >= (km + 2) or s >= nmax_in:
            return

        arr[0,      j0, k0, s] = arr[im, j0, k0, s]
        arr[im + 1, j0, k0, s] = arr[1,  j0, k0, s]

    return apply_bc_x_kernel, apply_bc_y_kernel, apply_bc_z_kernel


def apply_bc_cuda(arr_d, params, bc_kernels):
    im = params["im"]
    jm = params["jm"]
    km = params["km"]
    nmax_in = params["nmax"]

    apply_bc_x_kernel, apply_bc_y_kernel, apply_bc_z_kernel = bc_kernels

    block = (4, 4, 4)

    grid_z = (
        math.ceil((im + 2) / block[0]),
        math.ceil((jm + 2) / block[1]),
        math.ceil(nmax_in / block[2]),
    )
    apply_bc_z_kernel[grid_z, block](arr_d, im, jm, km, nmax_in)

    grid_y = (
        math.ceil((im + 2) / block[0]),
        math.ceil((km + 2) / block[1]),
        math.ceil(nmax_in / block[2]),
    )
    apply_bc_y_kernel[grid_y, block](arr_d, im, jm, km, nmax_in)

    grid_x = (
        math.ceil((jm + 2) / block[0]),
        math.ceil((km + 2) / block[1]),
        math.ceil(nmax_in / block[2]),
    )
    apply_bc_x_kernel[grid_x, block](arr_d, im, jm, km, nmax_in)


# ============================================================
# one solver step (CUDA resident)
# phiO_d, idO_d -> phiN_d, idN_d
# ============================================================
def solver_step_cuda(phiO_d, phiN_d, idO_d, idN_d, params, solver_kernel):
    im = params["im"]
    jm = params["jm"]
    km = params["km"]

    dx = params["dx"]
    dy = params["dy"]
    dz = params["dz"]

    dt = params["dt"]
    pss = params["pss"]

    eps = np.float32(params["eps"])
    omg = np.float32(params["omg"])
    mob = np.float32(params["mob"])

    block = (4, 4, 4)
    grid = (
        math.ceil(im / block[0]),
        math.ceil(jm / block[1]),
        math.ceil(km / block[2]),
    )

    solver_kernel[grid, block](
        phiO_d, phiN_d,
        idO_d, idN_d,
        im, jm, km,
        dx, dy, dz,
        dt, pss,
        eps, omg, mob
    )


# ============================================================
# Prepare
# ============================================================
def allocate_phase_fields(params):
    im = params["im"]
    jm = params["jm"]
    km = params["km"]
    nmax_local = params["nmax"]

    phiO = np.zeros((im + 2, jm + 2, km + 2, nmax_local), dtype=np.float32)
    phiN = np.zeros((im + 2, jm + 2, km + 2, nmax_local), dtype=np.float32)

    idO = np.full((im + 2, jm + 2, km + 2, nmax_local), -1, dtype=np.int16)
    idN = np.full((im + 2, jm + 2, km + 2, nmax_local), -1, dtype=np.int16)

    return phiO, phiN, idO, idN


def initialize_fields_from_params(phiN, idN, params):
    initialize_fields(
        phiN, idN,
        params["im"],
        params["jm"],
        params["km"],
        params["init_type"],
        params["bc_type"],
        seed=params["seed"],
        nph=params["nph"],
        radius=params["radius"],
        cx=params["cx"],
        cy=params["cy"],
        cz=params["cz"],
    )


def copy_initial_state(phiO, phiN, idO, idN):
    phiO[:, :, :, :] = phiN[:, :, :, :]
    idO[:, :, :, :] = idN[:, :, :, :]


def print_simulation_info(params):
    print("PFM framework initialized.")
    print(f"init = {params['init_type']}")
    print(f"bc   = {params['bc_type']}")
    print(f"grid = ({params['im']}, {params['jm']}, {params['km']})")
    print(f"nmax = {params['nmax']}")
    print(f"nph  = {params['nph']}")
    print(
        f"dx, dy, dz = "
        f"{params['dx']:.6e}, {params['dy']:.6e}, {params['dz']:.6e}"
    )
    print(f"xi   = {params['xi']:.6e}")
    print(f"www  = {params['www']:.6e}")
    print(f"cep  = {params['cep']:.6e}")
    print(f"emm  = {params['emm']:.6e}")
    print(f"ndim = {params['ndim']}")


# ============================================================
# Initialization
# ============================================================
def initialize_fields(
    phiN, idN,
    im, jm, km,
    init_type, bc_type,
    seed=1234,
    nph=0,
    radius=0.0,
    cx=-1.0, cy=-1.0, cz=-1.0,
):
    if init_type == "voronoi":
        init_voronoi(phiN, idN, im, jm, km, nph, seed, bc_type)

    elif init_type == "plane":
        init_plane(phiN, idN, im, jm, km)

    elif init_type == "triple":
        init_triple(phiN, idN, im, jm, km)

    elif init_type == "circle":
        cx_use = None if cx < 0.0 else cx
        cy_use = None if cy < 0.0 else cy
        init_circle(phiN, idN, im, jm, km, radius, cx_use, cy_use)

    elif init_type == "sphere":
        cx_use = None if cx < 0.0 else cx
        cy_use = None if cy < 0.0 else cy
        cz_use = None if cz < 0.0 else cz
        init_sphere(phiN, idN, im, jm, km, radius, cx_use, cy_use, cz_use)

    elif init_type == "neighbors":
        init_neighbors(phiN, idN, im, jm, km, nph)

    else:
        raise ValueError(f"Unsupported init type: {init_type}")


def init_ids_dense_for_bench(idN, nmax):
    idN[:, :, :, :] = -1
    for s in range(nmax):
        idN[:, :, :, s] = s


# ============================================================
# Output
# ============================================================
def write_snapshot(phiN, idN, im, jm, km, out_dir, step, sim_time):
    phi_save = phiN[1:im + 1, 1:jm + 1, 1:km + 1, :].copy()
    id_save = idN[1:im + 1, 1:jm + 1, 1:km + 1, :].copy()

    np.savez_compressed(
        out_dir / f"p_{step:08d}.npz",
        phi=phi_save,
        id=id_save,
        step=np.int32(step),
        time=np.float64(sim_time),
    )


if __name__ == "__main__":
    run("input.txt", out_dir="p_out")
