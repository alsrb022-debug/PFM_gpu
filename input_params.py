import configparser
import math
from pathlib import Path


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
    data["nph"] = cfg.getint("pf", "nph")

    data["emob"] = cfg.getfloat("pf", "emob")
    data["sigma"] = cfg.getfloat("pf", "sigma")
    data["xi_in"] = cfg.getfloat("pf", "xi_in")
    data["tifac"] = cfg.getfloat("pf", "tifac")
    data["pss"] = cfg.getfloat("pf", "pss")

    data["nstep"] = cfg.getint("time", "nstep")
    data["nout"] = cfg.getint("time", "nout")

    data["init_type"] = cfg.get("init", "type", fallback="voronoi").strip().lower()
    data["seed"] = cfg.getint("init", "seed", fallback=1234)
    data["radius"] = cfg.getfloat("init", "radius", fallback=-1.0)
    data["cx"] = cfg.getfloat("init", "cx", fallback=-1.0)
    data["cy"] = cfg.getfloat("init", "cy", fallback=-1.0)
    data["cz"] = cfg.getfloat("init", "cz", fallback=-1.0)

    data["bc_type"] = cfg.get("bc", "type", fallback="pbc").strip().lower()

    validate_input(data)
    return data


def validate_input(data):
    init_type = data["init_type"]

    if init_type not in ("voronoi", "plane", "triple", "circle", "sphere", "neighbors"):
        raise ValueError(f"Unsupported init type: {init_type}")

    if data["nmax"] <= 0:
        raise ValueError("nmax must be > 0")

    if data["nph"] <= 0:
        raise ValueError("nph must be > 0")

    if data["nmax"] > data["nph"]:
        raise ValueError("current version requires nmax <= nph")

    if init_type in ("plane", "circle", "sphere") and data["nph"] < 2:
        raise ValueError(f"init_type='{init_type}' requires nph >= 2")

    if init_type == "triple" and data["nph"] < 3:
        raise ValueError("init_type='triple' requires nph >= 3")

    if init_type in ("circle", "sphere") and data["radius"] <= 0.0:
        raise ValueError(f"init_type='{init_type}' requires radius > 0")


def prepare_output_dir(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_params(p, out_dir):
    params = {}

    params["im"] = p["im"]
    params["jm"] = p["jm"]
    params["km"] = p["km"]

    params["dx"] = p["dx"]
    params["dy"] = p["dy"]
    params["dz"] = p["dz"]

    params["nmax"] = p["nmax"]
    params["nph"] = p["nph"]

    params["emob"] = p["emob"]
    params["tifac"] = p["tifac"]
    params["sigma"] = p["sigma"]
    params["xi_in"] = p["xi_in"]
    params["pss"] = p["pss"]

    params["nstep"] = p["nstep"]
    params["nout"] = p["nout"]

    params["init_type"] = p["init_type"]
    params["bc_type"] = p["bc_type"]

    params["seed"] = p["seed"]
    params["radius"] = p["radius"]
    params["cx"] = p["cx"]
    params["cy"] = p["cy"]
    params["cz"] = p["cz"]

    params["out_dir"] = out_dir

    xi, www, cep, emm, ndim = compute_pf_coefficients(params)
    params["xi"] = xi
    params["www"] = www
    params["cep"] = cep
    params["emm"] = emm
    params["ndim"] = ndim

    params["eps"] = cep
    params["omg"] = www
    params["mob"] = emm

    params["dt"] = compute_dt(
        params["dx"],
        params["cep"],
        params["emm"],
        params["ndim"],
        params["tifac"],
    )

    return params


def compute_pf_coefficients(params):
    pi = math.pi

    dx = params["dx"]
    sigma = params["sigma"]
    emob = params["emob"]
    xi_in = params["xi_in"]

    xi = xi_in * dx
    www = 2.0 * sigma / xi
    cep = 4.0 / pi * math.sqrt(xi * sigma)
    emm = emob * sigma / (cep ** 2.0)

    ndim = 0
    if params["im"] > 1:
        ndim += 1
    if params["jm"] > 1:
        ndim += 1
    if params["km"] > 1:
        ndim += 1

    return xi, www, cep, emm, ndim


def compute_dt(dx, cep, emm, ndim, tifac):
    return dx ** 2.0 / (2.0 * ndim * (cep ** 2.0 * emm)) * tifac


def prepare_simulation(input_path="input.txt", out_dir="pout"):
    p = read_input(input_path)
    out_dir = prepare_output_dir(out_dir)
    return build_params(p, out_dir)
