import math
import numpy as np

try:
    from numba import cuda, float32, int16
    CUDA_OK = cuda.is_available()
except Exception:
    cuda = None
    float32 = None
    int16 = None
    CUDA_OK = False


def init_voronoi(phiN, idN, im, jm, km, nph, seed, bc_type):
    phiN[:, :, :, :] = 0.0
    idN[:, :, :, :] = -1

    nmax = phiN.shape[3]

    if nph <= 0:
        raise ValueError("init_voronoi requires nph > 0")
    if nmax <= 0:
        raise ValueError("init_voronoi requires nmax > 0")

    rng = np.random.default_rng(seed)

    rx = (rng.random(nph) * im + 1.0).astype(np.float32)
    ry = (rng.random(nph) * jm + 1.0).astype(np.float32)
    rz = (rng.random(nph) * km + 1.0).astype(np.float32)

    use_pbc = 1 if bc_type == "pbc" else 0

    if CUDA_OK:
        _init_voronoi_cuda(phiN, idN, im, jm, km, rx, ry, rz, use_pbc)
    else:
        _init_voronoi_cpu(phiN, idN, im, jm, km, rx, ry, rz, use_pbc)


def _init_voronoi_cpu(phiN, idN, im, jm, km, rx, ry, rz, use_pbc):
    nph = rx.shape[0]

    for i in range(1, im + 1):
        ri = float(i)
        for j in range(1, jm + 1):
            rj = float(j)
            for k in range(1, km + 1):
                rk = float(k)

                rmin = 1.0e100
                nmin = 0

                for n in range(nph):
                    dx = abs(float(rx[n]) - ri)
                    dy = abs(float(ry[n]) - rj)
                    dz = abs(float(rz[n]) - rk)

                    if use_pbc:
                        dx = min(dx, im - dx)
                        dy = min(dy, jm - dy)
                        dz = min(dz, km - dz)

                    if jm == 1:
                        dy = 0.0
                    if km == 1:
                        dz = 0.0

                    rr = dx * dx + dy * dy + dz * dz

                    if rr < rmin:
                        rmin = rr
                        nmin = n

                phiN[i, j, k, 0] = 1.0
                idN[i, j, k, 0] = nmin


def _init_voronoi_cuda(phiN, idN, im, jm, km, rx, ry, rz, use_pbc):
    phiN_d = cuda.to_device(phiN)
    idN_d = cuda.to_device(idN)

    rx_d = cuda.to_device(rx)
    ry_d = cuda.to_device(ry)
    rz_d = cuda.to_device(rz)

    block = (8, 8, 4)
    grid = (
        math.ceil(im / block[0]),
        math.ceil(jm / block[1]),
        math.ceil(km / block[2]),
    )

    voronoi_init_kernel[grid, block](
        phiN_d, idN_d,
        im, jm, km,
        rx_d, ry_d, rz_d, rx.shape[0],
        use_pbc
    )
    cuda.synchronize()

    phiN[:, :, :, :] = phiN_d.copy_to_host()
    idN[:, :, :, :] = idN_d.copy_to_host()


if CUDA_OK:
    @cuda.jit
    def voronoi_init_kernel(phiN, idN, im, jm, km, rx, ry, rz, nph, use_pbc):
        ii, jj, kk = cuda.grid(3)

        if ii >= im or jj >= jm or kk >= km:
            return

        i = ii + 1
        j = jj + 1
        k = kk + 1

        ri = float32(i)
        rj = float32(j)
        rk = float32(k)

        rmin = float32(1.0e30)
        nmin = int16(0)

        for n in range(nph):
            dx = abs(rx[n] - ri)
            dy = abs(ry[n] - rj)
            dz = abs(rz[n] - rk)

            if use_pbc == 1:
                dx = min(dx, im - dx)
                dy = min(dy, jm - dy)
                dz = min(dz, km - dz)

            if jm == 1:
                dy = 0.0
            if km == 1:
                dz = 0.0

            rr = dx * dx + dy * dy + dz * dz

            if rr < rmin:
                rmin = rr
                nmin = n

        phiN[i, j, k, 0] = 1.0
        idN[i, j, k, 0] = nmin
