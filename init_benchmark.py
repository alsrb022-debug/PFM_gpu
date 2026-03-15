import math
import numpy as np


def _check_phase_index(phiN, gid):
    nmax = phiN.shape[3]
    if gid < 0 or gid >= nmax:
        raise ValueError(f"phase index gid={gid} is out of range for nmax={nmax}")


def _clear_phi_id(phiN, idN):
    phiN[:, :, :, :] = 0.0
    idN[:, :, :, :] = -1


def init_plane(phiN, idN, im, jm, km):
    _clear_phi_id(phiN, idN)

    if phiN.shape[3] < 2:
        raise ValueError("init_plane requires nmax >= 2")

    ic = (im + 1) // 2

    for i in range(1, im + 1):
        gid = 0 if i < ic else 1
        _check_phase_index(phiN, gid)

        for j in range(1, jm + 1):
            for k in range(1, km + 1):
                phiN[i, j, k, gid] = 1.0
                idN[i, j, k, gid] = gid


def init_triple(phiN, idN, im, jm, km):
    if km != 1:
        raise ValueError("triple init is for 2D only (use km = 1)")

    _clear_phi_id(phiN, idN)

    if phiN.shape[3] < 3:
        raise ValueError("init_triple requires nmax >= 3")

    cx = 0.5 * (im + 1)
    cy = 0.5 * (jm + 1)

    for i in range(1, im + 1):
        for j in range(1, jm + 1):
            x = float(i) - cx
            y = float(j) - cy
            th = math.atan2(y, x)

            if -math.pi / 3.0 <= th < math.pi / 3.0:
                gid = 0
            elif th >= math.pi / 3.0:
                gid = 1
            else:
                gid = 2

            _check_phase_index(phiN, gid)
            phiN[i, j, 1, gid] = 1.0
            idN[i, j, 1, gid] = gid


def init_circle(phiN, idN, im, jm, km, radius, cx=None, cy=None):
    if km != 1:
        raise ValueError("circle init is for 2D only (use km = 1)")

    _clear_phi_id(phiN, idN)

    if phiN.shape[3] < 2:
        raise ValueError("init_circle requires nmax >= 2")

    if cx is None:
        cx = 0.5 * (im + 1)
    if cy is None:
        cy = 0.5 * (jm + 1)

    r2 = radius * radius

    for i in range(1, im + 1):
        for j in range(1, jm + 1):
            dx = float(i) - cx
            dy = float(j) - cy

            gid = 1 if (dx * dx + dy * dy) <= r2 else 0

            _check_phase_index(phiN, gid)
            phiN[i, j, 1, gid] = 1.0
            idN[i, j, 1, gid] = gid


def init_sphere(phiN, idN, im, jm, km, radius, cx=None, cy=None, cz=None):
    _clear_phi_id(phiN, idN)

    if phiN.shape[3] < 2:
        raise ValueError("init_sphere requires nmax >= 2")

    if cx is None:
        cx = 0.5 * (im + 1)
    if cy is None:
        cy = 0.5 * (jm + 1)
    if cz is None:
        cz = 0.5 * (km + 1)

    r2 = radius * radius

    for i in range(1, im + 1):
        for j in range(1, jm + 1):
            for k in range(1, km + 1):
                dx = float(i) - cx
                dy = float(j) - cy
                dz = float(k) - cz

                gid = 1 if (dx * dx + dy * dy + dz * dz) <= r2 else 0

                _check_phase_index(phiN, gid)
                phiN[i, j, k, gid] = 1.0
                idN[i, j, k, gid] = gid


def init_neighbors(phiN, idN, im, jm, km, nph):
    if km != 1:
        raise ValueError("neighbors init is for 2D only (use km = 1)")

    _clear_phi_id(phiN, idN)

    if nph < 2:
        raise ValueError("init_neighbors requires nph >= 2")

    if phiN.shape[3] < nph:
        raise ValueError(f"init_neighbors requires nmax >= nph, got nmax={phiN.shape[3]}, nph={nph}")

    cx = 0.5 * (im + 1)
    cy = 0.5 * (jm + 1)

    radius = 20.0
    r2 = radius * radius

    n_outer = nph - 1

    for i in range(1, im + 1):
        for j in range(1, jm + 1):
            dx = float(i) - cx
            dy = float(j) - cy
            rr = dx * dx + dy * dy

            if rr <= r2:
                gid = 0
            else:
                th = math.atan2(dy, dx)
                if th < 0.0:
                    th += 2.0 * math.pi

                sector = int(n_outer * th / (2.0 * math.pi))
                if sector >= n_outer:
                    sector = n_outer - 1

                gid = sector + 1

            _check_phase_index(phiN, gid)
            phiN[i, j, 1, gid] = 1.0
            idN[i, j, 1, gid] = gid
