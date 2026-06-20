"""
sculptured_surface.py — NURBS / Parametric Sculptured Surface Mathematics
                        with PyBullet Integration for Gripper Simulation
UPDATED:
  - Triangle hole enlarged so peg goes fully inside
  - Dome surface hole_r_uv increased
  - Conjugate profile stability calculation improved
"""
import numpy as np
import pybullet as p
import os
import math
import threading
import tempfile

_MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_mesh_cache")
os.makedirs(_MESH_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# §1  B-SPLINE BASIS
# ═══════════════════════════════════════════════════════════════════════
def _bspline_basis(i, deg, t, knots):
    n_basis = len(knots) - deg - 1
    if deg == 0:
        if i == n_basis - 1 and knots[i] < knots[i + 1]:
            return 1.0 if knots[i] <= t <= knots[i + 1] else 0.0
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
    d1 = knots[i + deg] - knots[i]
    d2 = knots[i + deg + 1] - knots[i + 1]
    c1 = ((t - knots[i]) / d1 * _bspline_basis(i, deg - 1, t, knots)) if d1 > 0 else 0.0
    c2 = ((knots[i + deg + 1] - t) / d2 * _bspline_basis(i + 1, deg - 1, t, knots)) if d2 > 0 else 0.0
    return c1 + c2


def _uniform_knots(n, deg):
    m = n + deg + 1
    knots = np.zeros(m)
    n_internal = m - 2 * (deg + 1)
    for j in range(n_internal):
        knots[deg + 1 + j] = (j + 1) / (n_internal + 1)
    knots[-(deg + 1):] = 1.0
    return knots


# ═══════════════════════════════════════════════════════════════════════
# §2  NURBS SURFACE
# ═══════════════════════════════════════════════════════════════════════
class NURBSSurface:
    def __init__(self, ctrl_pts, weights, ku, kv, du=3, dv=3):
        self.P  = np.asarray(ctrl_pts, dtype=np.float64)
        self.W  = np.asarray(weights,  dtype=np.float64)
        self.ku = np.asarray(ku, dtype=np.float64)
        self.kv = np.asarray(kv, dtype=np.float64)
        self.du, self.dv = du, dv
        self.nu, self.nv = self.P.shape[:2]
        self.u_range = (float(self.ku[du]),  float(self.ku[-(du + 1)]))
        self.v_range = (float(self.kv[dv]),  float(self.kv[-(dv + 1)]))

    def evaluate(self, u, v):
        u = float(np.clip(u, *self.u_range))
        v = float(np.clip(v, *self.v_range))
        num, den = np.zeros(3), 0.0
        for i in range(self.nu):
            Ni = _bspline_basis(i, self.du, u, self.ku)
            if Ni == 0: continue
            for j in range(self.nv):
                Nj = _bspline_basis(j, self.dv, v, self.kv)
                if Nj == 0: continue
                w    = Ni * Nj * self.W[i, j]
                num += w * self.P[i, j]
                den += w
        return num / den if den > 1e-15 else np.zeros(3)

    def normal(self, u, v, h=1e-5):
        Su = (self.evaluate(u + h, v) - self.evaluate(u - h, v)) / (2 * h)
        Sv = (self.evaluate(u, v + h) - self.evaluate(u, v - h)) / (2 * h)
        n  = np.cross(Su, Sv)
        nm = np.linalg.norm(n)
        return n / nm if nm > 1e-12 else np.array([0.0, 0.0, 1.0])

    def curvature(self, u, v, h=1e-4):
        return _curvature_impl(self, u, v, h)


# ═══════════════════════════════════════════════════════════════════════
# §3  PARAMETRIC SURFACE
# ═══════════════════════════════════════════════════════════════════════
class ParametricSurface:
    def __init__(self, func, u_range=(0, 1), v_range=(0, 1)):
        self.func    = func
        self.u_range = u_range
        self.v_range = v_range

    def evaluate(self, u, v):
        return np.array(self.func(float(u), float(v)), dtype=np.float64)

    def normal(self, u, v, h=1e-5):
        Su = (self.evaluate(u + h, v) - self.evaluate(u - h, v)) / (2 * h)
        Sv = (self.evaluate(u, v + h) - self.evaluate(u, v - h)) / (2 * h)
        n  = np.cross(Su, Sv)
        nm = np.linalg.norm(n)
        return n / nm if nm > 1e-12 else np.array([0.0, 0.0, 1.0])

    def curvature(self, u, v, h=1e-4):
        return _curvature_impl(self, u, v, h)


# ═══════════════════════════════════════════════════════════════════════
# §4  DIFFERENTIAL GEOMETRY
# ═══════════════════════════════════════════════════════════════════════
def _curvature_impl(surf, u, v, h=1e-4):
    S   = surf.evaluate(u, v)
    Su  = (surf.evaluate(u + h, v) - surf.evaluate(u - h, v)) / (2 * h)
    Sv  = (surf.evaluate(u, v + h) - surf.evaluate(u, v - h)) / (2 * h)
    Suu = (surf.evaluate(u + h, v) - 2 * S + surf.evaluate(u - h, v)) / (h * h)
    Svv = (surf.evaluate(u, v + h) - 2 * S + surf.evaluate(u, v - h)) / (h * h)
    Suv = (surf.evaluate(u + h, v + h) - surf.evaluate(u + h, v - h)
           - surf.evaluate(u - h, v + h) + surf.evaluate(u - h, v - h)) / (4 * h * h)
    n_vec  = np.cross(Su, Sv)
    n_norm = np.linalg.norm(n_vec)
    zero   = {"K": 0.0, "H": 0.0, "k1": 0.0, "k2": 0.0,
               "normal": np.array([0., 0., 1.])}
    if n_norm < 1e-12:
        return zero
    n_hat = n_vec / n_norm
    E = float(Su @ Su);   F = float(Su @ Sv);   G = float(Sv @ Sv)
    L = float(Suu @ n_hat); M = float(Suv @ n_hat); N_ff = float(Svv @ n_hat)
    denom = E * G - F * F
    if abs(denom) < 1e-15:
        return zero
    K     = (L * N_ff - M * M) / denom
    H_val = (E * N_ff + G * L - 2 * F * M) / (2 * denom)
    disc  = max(0.0, H_val * H_val - K)
    k1    = H_val + math.sqrt(disc)
    k2    = H_val - math.sqrt(disc)
    return {"K": K, "H": H_val, "k1": k1, "k2": k2, "normal": n_hat}


def _classify(K, H):
    eps = 1e-6
    if abs(K) < eps and abs(H) < eps: return "PLANAR"
    if abs(K) < eps:                   return "CYLINDRICAL"
    return "ELLIPTIC" if K > eps else "HYPERBOLIC"


# ═══════════════════════════════════════════════════════════════════════
# §5  TESSELLATION & OBJ EXPORT
# ═══════════════════════════════════════════════════════════════════════
def _tri_uv_verts(center, r):
    cu, cv = center
    ang0 = math.pi / 2.0
    verts = []
    for k in range(3):
        ang = ang0 + k * 2.0 * math.pi / 3.0
        verts.append((cu + r * math.cos(ang), cv + r * math.sin(ang)))
    return verts


def _pt_in_tri(p, a, b, c):
    def _sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    b1 = _sign(p, a, b) <= 0.0
    b2 = _sign(p, b, c) <= 0.0
    b3 = _sign(p, c, a) <= 0.0
    return (b1 == b2) and (b2 == b3)


def _build_hole_checker(hole_uv, hole_r, hole_shape):
    if hole_uv is None or hole_r <= 0:
        return None
    shape = (hole_shape or "circle").lower()
    if shape in ("triangle", "tri"):
        tri = _tri_uv_verts(hole_uv, hole_r)
        return lambda u, v: _pt_in_tri((u, v), tri[0], tri[1], tri[2])
    if shape in ("square", "box"):
        return lambda u, v: max(abs(u - hole_uv[0]), abs(v - hole_uv[1])) <= hole_r
    # default: circular hole
    rr = hole_r * hole_r
    return lambda u, v: (u - hole_uv[0]) ** 2 + (v - hole_uv[1]) ** 2 <= rr


def tessellate(surface, nu=30, nv=30, hole_uv=None, hole_r=0.0, hole_shape="circle"):
    ur = getattr(surface, "u_range", (0, 1))
    vr = getattr(surface, "v_range", (0, 1))
    us = np.linspace(ur[0], ur[1], nu)
    vs = np.linspace(vr[0], vr[1], nv)
    verts, norms = [], []
    grid_idx     = np.full((nu, nv), -1, dtype=int)
    hole_hit = _build_hole_checker(hole_uv, hole_r, hole_shape)
    for i, u_val in enumerate(us):
        for j, v_val in enumerate(vs):
            if hole_hit and hole_hit(u_val, v_val):
                continue
            grid_idx[i, j] = len(verts)
            verts.append(surface.evaluate(u_val, v_val))
            norms.append(surface.normal(u_val, v_val))
    faces = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a, b = grid_idx[i, j],     grid_idx[i + 1, j]
            c, d = grid_idx[i + 1, j + 1], grid_idx[i, j + 1]
            if a >= 0 and b >= 0 and c >= 0: faces.append([a, b, c])
            if a >= 0 and c >= 0 and d >= 0: faces.append([a, c, d])
    v_arr = np.array(verts)       if verts  else np.zeros((0, 3))
    f_arr = np.array(faces, dtype=int) if faces  else np.zeros((0, 3), dtype=int)
    n_arr = np.array(norms)       if norms  else np.zeros((0, 3))
    return v_arr, f_arr, n_arr


def _export_obj(verts, faces, filepath):
    with open(filepath, "w") as f:
        f.write("# Sculptured Surface Mesh\n")
        for v in verts:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for fc in faces:
            f.write(f"f {fc[0]+1} {fc[1]+1} {fc[2]+1}\n")


# ═══════════════════════════════════════════════════════════════════════
# §6  PYBULLET VISUAL BODY CREATION
# ═══════════════════════════════════════════════════════════════════════
def create_mesh_body(verts, faces, pos, orn, rgba, mass=0, name="mesh"):
    # Each engine runs in its own thread. Use a thread-scoped OBJ path so
    # parallel workbenches never fight over the same cached mesh file.
    obj_path = None
    last_exc = None
    for mesh_dir in (_MESH_DIR, tempfile.gettempdir()):
        try:
            os.makedirs(mesh_dir, exist_ok=True)
            candidate = os.path.join(
                mesh_dir,
                f"{name}_{os.getpid()}_{threading.get_ident()}.obj",
            )
            _export_obj(verts, faces, candidate)
            obj_path = candidate
            break
        except PermissionError as exc:
            last_exc = exc
            continue
    if obj_path is None:
        raise last_exc or PermissionError("Could not create sculptured surface mesh cache file.")
    flags = p.GEOM_FORCE_CONCAVE_TRIMESH if mass == 0 else 0
    col   = p.createCollisionShape(p.GEOM_MESH, fileName=obj_path, flags=flags)
    vis   = p.createVisualShape(p.GEOM_MESH,    fileName=obj_path,
                                rgbaColor=[0, 0, 0, 0])
    body  = p.createMultiBody(mass, col, vis, pos, orn)
    p.changeDynamics(body, -1, lateralFriction=1.0, spinningFriction=0.1)
    return body


def _height_color(z, z_min, z_max, base_rgb):
    if z_max - z_min < 1e-6: t = 0.5
    else:                      t = (z - z_min) / (z_max - z_min)
    r = min(1.0, base_rgb[0] * (0.3 + 0.7 * t) + 0.3 * t * t)
    g = min(1.0, base_rgb[1] * (0.3 + 0.7 * t) + 0.3 * t * t)
    b = min(1.0, base_rgb[2] * (0.3 + 0.7 * t) + 0.3 * t * t)
    return [r, g, b, 1.0]


def create_tile_surface(surface, position, base_rgb, hole_uv=(0.5, 0.5),
                        hole_r=0.10, grid_n=12, hole_shape="circle"):
    ur  = getattr(surface, "u_range", (0, 1))
    vr  = getattr(surface, "v_range", (0, 1))
    pos = np.array(position, dtype=float)
    us  = np.linspace(ur[0], ur[1], grid_n)
    vs  = np.linspace(vr[0], vr[1], grid_n)
    samples = []
    hole_hit = _build_hole_checker(hole_uv, hole_r, hole_shape)
    for u_val in us:
        for v_val in vs:
            if hole_hit and hole_hit(u_val, v_val):
                continue
            pt = surface.evaluate(u_val, v_val)
            samples.append((u_val, v_val, pt))
    if not samples:
        return
    z_vals  = [s[2][2] for s in samples]
    z_min, z_max = min(z_vals), max(z_vals)
    u_step  = (ur[1] - ur[0]) / grid_n
    v_step  = (vr[1] - vr[0]) / grid_n
    p0 = surface.evaluate(ur[0], vr[0])
    p1 = surface.evaluate(ur[0] + u_step, vr[0])
    p2 = surface.evaluate(ur[0], vr[0] + v_step)
    tile_w  = max(0.005, np.linalg.norm(np.array(p1) - np.array(p0)) * 0.52)
    tile_d  = max(0.005, np.linalg.norm(np.array(p2) - np.array(p0)) * 0.52)
    for u_val, v_val, pt in samples:
        local_xyz = np.array(pt)
        world     = pos + local_xyz
        color     = _height_color(local_xyz[2], z_min, z_max, base_rgb)
        extra_h   = max(0.003, (local_xyz[2] - z_min) * 0.5)
        th        = 0.005 + extra_h
        col_shape = p.createCollisionShape(p.GEOM_BOX,
                                           halfExtents=[tile_w, tile_d, th])
        vis_shape = p.createVisualShape(p.GEOM_BOX,
                                        halfExtents=[tile_w, tile_d, th],
                                        rgbaColor=color,
                                        specularColor=[0.4, 0.4, 0.4])
        p.createMultiBody(0, col_shape, vis_shape,
                          [world[0], world[1], world[2] - th])


# ═══════════════════════════════════════════════════════════════════════
# §7  PRESET SURFACES
# ═══════════════════════════════════════════════════════════════════════
def make_dome(radius=0.12, height=0.10):
    """Elliptic paraboloid: K > 0 everywhere."""
    def f(u, v):
        x  = (u - 0.5) * 2 * radius
        y  = (v - 0.5) * 2 * radius
        r2 = 4 * ((u - 0.5)**2 + (v - 0.5)**2)
        return [x, y, height * max(0.0, 1.0 - r2)]
    return ParametricSurface(f)


def make_saddle(width=0.24, depth=0.24, curv=0.50):
    """Hyperbolic paraboloid: K < 0."""
    hw = width / 2
    def f(u, v):
        x  = (u - 0.5) * width
        y  = (v - 0.5) * depth
        lx = (u - 0.5) * 2
        ly = (v - 0.5) * 2
        return [x, y, curv * (lx * lx - ly * ly) * hw]
    return ParametricSurface(f)


def make_wave(width=0.24, depth=0.24, amp=0.025, freq=1.5):
    """Sinusoidal surface."""
    def f(u, v):
        return [(u - 0.5) * width,
                (v - 0.5) * depth,
                amp * math.sin(2 * math.pi * freq * u) *
                      math.cos(2 * math.pi * freq * v)]
    return ParametricSurface(f)


def make_nurbs_freeform(size=0.24):
    """5x5 NURBS freeform surface."""
    n, deg = 5, 3
    P  = np.zeros((n, n, 3))
    zz = [[0.00, 0.02, 0.04, 0.02, 0.00],
          [0.02, 0.06, 0.10, 0.06, 0.02],
          [0.04, 0.10, 0.12, 0.10, 0.04],
          [0.02, 0.06, 0.10, 0.06, 0.02],
          [0.00, 0.02, 0.04, 0.02, 0.00]]
    for i in range(n):
        for j in range(n):
            P[i, j] = [(i / (n-1) - 0.5) * size,
                       (j / (n-1) - 0.5) * size,
                       zz[i][j]]
    W  = np.ones((n, n))
    ku = _uniform_knots(n, deg)
    kv = _uniform_knots(n, deg)
    return NURBSSurface(P, W, ku, kv, deg, deg)


# ═══════════════════════════════════════════════════════════════════════
# §8  CONJUGATE SURFACE ANALYZER
# ═══════════════════════════════════════════════════════════════════════
class ConjugateAnalyzer:
    @staticmethod
    def contact_curvature(surface, u, v):
        return surface.curvature(u, v)

    @staticmethod
    def conjugate_finger_curvature(surface, u, v):
        c = surface.curvature(u, v)
        return {"k1_finger": -c["k1"], "k2_finger": -c["k2"],
                "workpiece_K": c["K"],  "workpiece_H": c["H"],
                "normal": c["normal"]}

    @staticmethod
    def grip_force(surface, u, v, base_force=200):
        c = surface.curvature(u, v)
        K, H = c["K"], abs(c["H"])
        if   K > 0: factor = max(0.6, 1.0 - H * 2.0)
        elif K < 0: factor = 1.0 + abs(K) * 5.0
        else:       factor = 1.0
        return base_force * factor

    @staticmethod
    def stability(surface, u, v):
        c    = surface.curvature(u, v)
        K, H = c["K"], c["H"]
        if   K > 0:          k_sc = 0.4
        elif abs(K) < 1e-6:  k_sc = 0.25
        else:                 k_sc = max(0.0, 0.15 - abs(K) * 0.5)
        h_sc = min(0.3, abs(H) * 0.1)
        if   K > 0 and abs(H) > 0.5: cl_sc, cl_type = 0.3, "FULL_FORM_CLOSURE"
        elif K >= 0:                   cl_sc, cl_type = 0.2, "PARTIAL_FORM_CLOSURE"
        else:                          cl_sc, cl_type = 0.1, "FORCE_CLOSURE_ONLY"
        stab = min(1.0, k_sc + h_sc + cl_sc)
        return {"stability": stab, "closure": cl_type,
                "K": K, "H": H, "k1": c["k1"], "k2": c["k2"],
                "class": _classify(K, H)}


# ═══════════════════════════════════════════════════════════════════════
# §9  BUILD COMPLETE SCULPTURED STATION
# ═══════════════════════════════════════════════════════════════════════
def build_station(surface_type, position, orn_quat, rgba,
                  hole_uv=(0.5, 0.5), hole_r=0.09, hole_shape="circle",
                  res=30, name="sculpt"):
    """
    Create sculptured surface in PyBullet.
    UPDATED: hole_r increased so triangle peg fits fully.
    """
    makers = {
        "dome":     make_dome,
        "saddle":   make_saddle,
        "wave":     make_wave,
        "freeform": make_nurbs_freeform,
    }
    surface = makers.get(surface_type, make_dome)()

    print(f"  Tessellating {surface_type} ({res}x{res})...", flush=True)
    verts, faces, norms = tessellate(surface, res, res, hole_uv, hole_r, hole_shape)

    if len(verts) == 0 or len(faces) == 0:
        print(f"  WARNING: empty mesh for {name}, using dome fallback", flush=True)
        surface = make_dome()
        verts, faces, norms = tessellate(surface, res, res, hole_uv, hole_r, hole_shape)

    body = create_mesh_body(verts, faces, position, orn_quat, rgba,
                            mass=0, name=name)
    print(f"  {surface_type}: {len(verts)} verts, {len(faces)} tris", flush=True)

    base_rgb = [rgba[0], rgba[1], rgba[2]]
    print(f"  Creating tile visualization...", flush=True)
    create_tile_surface(surface, position, base_rgb, hole_uv, hole_r,
                        grid_n=14, hole_shape=hole_shape)
    print(f"  Tiles created.", flush=True)

    # Curvature at sample point just outside hole
    u_sample = min(hole_uv[0] + hole_r + 0.03, 0.95)
    v_sample = hole_uv[1]
    curv        = surface.curvature(u_sample, v_sample)
    ins_normal  = surface.normal(hole_uv[0], hole_uv[1])
    stab        = ConjugateAnalyzer.stability(surface, u_sample, v_sample)

    # World-space hole position
    hole_local       = surface.evaluate(hole_uv[0], hole_uv[1])
    rot              = np.array(p.getMatrixFromQuaternion(orn_quat)).reshape(3, 3)
    hole_world       = np.array(position, dtype=float) + rot @ hole_local
    ins_normal_world = rot @ ins_normal

    meta = {
        "type":             "sculptured",
        "surface_type":     surface_type,
        "hole_world":       hole_world,
        "insertion_normal": ins_normal_world,
        "curvature":        curv,
        "stability":        stab,
        "surface_class":    _classify(curv["K"], curv["H"]),
        "conjugate_profile": {
            "contact_type": f"sculptured_{surface_type}",
            "closure":      stab["closure"],
            "desc":         f"Curvature-adaptive conjugate on {surface_type} "
                            f"(K={curv['K']:.4f})",
            "min_contacts": 2,
            "K":            curv["K"],
            "H":            curv["H"],
            "stability":    stab["stability"],
        },
    }
    return body, surface, meta
