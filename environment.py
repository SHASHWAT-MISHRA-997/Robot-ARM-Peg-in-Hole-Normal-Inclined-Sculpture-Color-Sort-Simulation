"""
environment.py  — FIXED
Hole shapes:
  Normal Surface  -> square peg  + SQUARE hole  (larger)
  Inclined 15/30/45 -> cylinder peg + CIRCULAR hole (larger)
  Dome/Sculpture -> triangle peg + TRIANGLE hole (larger)
  Color Sort:
    orange -> cylinder peg + CIRCULAR hole
    blue   -> triangle peg + TRIANGLE hole
    green  -> square  peg + SQUARE  hole
All holes enlarged so pegs go fully inside.
Camera positions fixed so full robot is visible.
"""
import pybullet as p
import pybullet_data
import numpy as np
import math
import sculptured_surface as ss


class Environment:
    def __init__(self, client_id):
        self.cid = client_id
        self.surfaces = {}
        self.holes = {}
        self.conveyor = None
        self._surface_body_ids = []
        self._magazine_body_ids = []
        self._conveyor_body_ids = []
        self._hidden_body_colors = {}
        self._surface_label_bodies = {}
        self._surface_original_colors = {}
        self._focus_hidden_body_colors = {}
        self._hidden_body_transforms = {}
        self._surface_hidden_transforms = {}
        self._focus_hidden_body_transforms = {}

    def _hidden_pose_for_body(self, bid):
        row = int(bid % 9)
        col = int((bid // 9) % 9)
        return [70.0 + row * 0.6, 70.0 + col * 0.6, 6.0 + (bid % 5) * 0.2], [0, 0, 0, 1]

    def _stash_body_transform(self, bid, store):
        if bid in store:
            return
        try:
            pos, orn = p.getBasePositionAndOrientation(bid)
            store[bid] = (list(pos), list(orn))
        except Exception:
            pass

    def _hide_body_transform(self, bid, store):
        self._stash_body_transform(bid, store)
        try:
            pos, orn = self._hidden_pose_for_body(bid)
            p.resetBasePositionAndOrientation(bid, pos, orn)
            p.resetBaseVelocity(bid, [0, 0, 0], [0, 0, 0])
        except Exception:
            pass

    def _restore_body_transform(self, bid, store):
        if bid not in store:
            return
        try:
            pos, orn = store[bid]
            p.resetBasePositionAndOrientation(bid, pos, orn)
            p.resetBaseVelocity(bid, [0, 0, 0], [0, 0, 0])
        except Exception:
            pass
        store.pop(bid, None)

    # ─────────────────────────────────────────────────────────────────
    def hide_non_conveyor_bodies(self, extra_body_ids=None):
        hide_ids = list(self._surface_body_ids) + list(self._magazine_body_ids)
        if extra_body_ids:
            hide_ids.extend(extra_body_ids)
        for bid in hide_ids:
            self._hide_body_transform(bid, self._hidden_body_transforms)
            try:
                vis = p.getVisualShapeData(bid)
                for si in vis:
                    rgba = list(si[7])
                    self._hidden_body_colors[(bid, si[1])] = rgba
                    p.changeVisualShape(bid, si[1], rgbaColor=[rgba[0], rgba[1], rgba[2], 0])
            except Exception:
                pass

    def show_non_conveyor_bodies(self):
        for (bid, li), rgba in self._hidden_body_colors.items():
            try:
                p.changeVisualShape(bid, li, rgbaColor=rgba)
            except Exception:
                pass
        self._hidden_body_colors.clear()
        for bid in list(self._hidden_body_transforms.keys()):
            self._restore_body_transform(bid, self._hidden_body_transforms)

    def show_only_surface(self, label, extra_body_ids=None):
        self.show_all_surfaces()
        for sl in list(self._surface_label_bodies.keys()):
            if sl != label:
                self._hide_surface(sl)
        hide_ids = list(self._magazine_body_ids) + list(self._conveyor_body_ids)
        if extra_body_ids:
            hide_ids.extend(extra_body_ids)
        self._hide_bodies(hide_ids,
                          self._focus_hidden_body_colors,
                          self._focus_hidden_body_transforms)

    def show_all_surfaces(self):
        for sl in list(self._surface_label_bodies.keys()):
            self._restore_surface_visibility(sl)
        self._surface_original_colors.clear()
        self._restore_hidden_bodies(self._focus_hidden_body_colors,
                                    self._focus_hidden_body_transforms)

    def _hide_bodies(self, body_ids, store, transform_store):
        for bid in body_ids:
            self._hide_body_transform(bid, transform_store)
            try:
                vis = p.getVisualShapeData(bid)
                for si in vis:
                    key = (bid, si[1])
                    if key not in store:
                        store[key] = list(si[7])
                    rgba = store[key]
                    p.changeVisualShape(bid, si[1], rgbaColor=[rgba[0], rgba[1], rgba[2], 0])
            except Exception:
                pass

    def _restore_hidden_bodies(self, store, transform_store):
        for (bid, li), rgba in list(store.items()):
            try:
                p.changeVisualShape(bid, li, rgbaColor=rgba)
            except Exception:
                pass
        store.clear()
        for bid in list(transform_store.keys()):
            self._restore_body_transform(bid, transform_store)

    def _hide_surface(self, label):
        for bid in self._surface_label_bodies.get(label, []):
            self._hide_body_transform(bid, self._surface_hidden_transforms)
            try:
                vis = p.getVisualShapeData(bid)
                for si in vis:
                    key = (bid, si[1])
                    if key not in self._surface_original_colors:
                        self._surface_original_colors[key] = list(si[7])
                    rgba = self._surface_original_colors[key]
                    p.changeVisualShape(bid, si[1], rgbaColor=[rgba[0], rgba[1], rgba[2], 0])
            except Exception:
                pass

    def _restore_surface_visibility(self, label):
        for bid in self._surface_label_bodies.get(label, []):
            self._restore_body_transform(bid, self._surface_hidden_transforms)
            try:
                vis = p.getVisualShapeData(bid)
                for si in vis:
                    key = (bid, si[1])
                    if key in self._surface_original_colors:
                        p.changeVisualShape(bid, si[1], rgbaColor=self._surface_original_colors[key])
                        del self._surface_original_colors[key]
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────
    def setup_scene(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        try:
            p.setPhysicsEngineParameter(
                enableContinuousCollisionDetection=1,
                contactBreakingThreshold=0.001,
            )
        except TypeError:
            # Older/newer PyBullet builds expose different CCD kwargs. Keep the
            # contact threshold and continue instead of crashing the sim thread.
            p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

    def create_workbench(self, center, width=2.5, depth=1.2, height=0.7):
        cx, cy, zf = center
        zt = zf + height
        hw, hd, th = width / 2, depth / 2, 0.05
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hw, hd, th])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hw, hd, th],
                                  rgbaColor=[0.2, 0.2, 0.22, 1])
        p.createMultiBody(0, col, vis, [cx, cy, zt - th])
        ls, lh = 0.06, (zt - th * 2 - zf) / 2
        lc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[ls, ls, lh])
        lv = p.createVisualShape(p.GEOM_BOX, halfExtents=[ls, ls, lh],
                                 rgbaColor=[0.5, 0.5, 0.55, 1])
        for sx, sy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            p.createMultiBody(0, lc, lv,
                              [cx + sx * (hw - 0.15), cy + sy * (hd - 0.15), zf + lh])
        return zt

    # ═══════════════════════════════════════════════════════════
    # PEG DIMENSIONS
    # ═══════════════════════════════════════════════════════════
    PEG_H   = 0.12
    CYL_R   = 0.024    # cylinder peg radius
    SQ_H    = 0.022    # square peg half-side
    TRI_R   = 0.027    # triangle peg circumscribed radius

    # ENLARGED hole clearances — pegs go fully inside
    HOLE_CYL_R = 0.045   # circular hole radius   (extra clearance for full insertion)
    HOLE_SQ_H  = 0.045   # square hole half-side  (extra clearance for full insertion)
    HOLE_TRI_R = 0.050   # triangle hole circum-r (extra clearance for full insertion)

    # ═══════════════════════════════════════════════════════════
    # CIRCULAR HOLE FRAME  (for cylinder pegs)
    # ═══════════════════════════════════════════════════════════
    def _make_circular_hole_frame(self, world_pos, world_orn, half_depth, color_rgb,
                                  collidable=True):
        """
        True circular hole using 8 wedge-shaped boxes arranged in a ring.
        Much more accurate circle than 4-box approximation.
        """
        bids = []
        c   = [min(1.0, v * 1.3) for v in color_rgb] + [1.0]
        rot = np.array(p.getMatrixFromQuaternion(world_orn)).reshape(3, 3)
        pos = np.array(world_pos)
        d   = half_depth
        R   = self.HOLE_CYL_R   # hole radius
        hw  = 0.130              # plate half-width
        n   = 8                  # number of segments

        for i in range(n):
            angle = (i + 0.5) * 2 * math.pi / n
            # outer strip going from R to hw
            strip_half = (hw - R) / 2.0
            cx_local   = math.cos(angle) * (R + strip_half)
            cy_local   = math.sin(angle) * (R + strip_half)
            seg_angle  = 2 * math.pi / n
            # width of segment arc
            arc_w = R * math.sin(seg_angle / 2) + 0.005

            box_half = [strip_half, arc_w, d]
            lp = np.array([cx_local, cy_local, 0.0])
            wp = pos + rot @ lp

            # rotate box to point outward
            seg_orn_euler = [0, 0, angle]
            seg_orn       = p.getQuaternionFromEuler(seg_orn_euler)
            # combine with world_orn
            combined_orn  = p.multiplyTransforms([0,0,0], world_orn,
                                                  [0,0,0], seg_orn)[1]

            vis_s = p.createVisualShape(p.GEOM_BOX, halfExtents=box_half, rgbaColor=c)
            col_s = (p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half)
                     if collidable else -1)
            bid   = p.createMultiBody(0, col_s, vis_s, wp.tolist(), list(combined_orn))
            bids.append(bid)
        return bids

    # ═══════════════════════════════════════════════════════════
    # HOLE FRAME BUILDER — shape-matched
    # ═══════════════════════════════════════════════════════════
    def _make_hole_frame(self, shape, world_pos, world_orn, half_depth, color_rgb,
                         collidable=True):
        """
        Build shape-matched hole frame.
        shape: "cylinder" -> circular hole
               "square"   -> square hole
               "triangle" -> triangle hole
        """
        if shape == "cylinder":
            return self._make_circular_hole_frame(
                world_pos, world_orn, half_depth, color_rgb, collidable=collidable)

        bids = []
        c    = [min(1.0, v * 1.3) for v in color_rgb] + [1.0]
        rot  = np.array(p.getMatrixFromQuaternion(world_orn)).reshape(3, 3)
        pos  = np.array(world_pos)
        d    = half_depth
        hw   = 0.130

        if shape == "square":
            s  = self.HOLE_SQ_H
            sw = (hw - s) / 2.0
            parts = [
                ([hw,   sw,  d], [0.0,    s + sw,  0.0]),
                ([hw,   sw,  d], [0.0,  -(s + sw), 0.0]),
                ([sw,    s,  d], [-(s + sw), 0.0,  0.0]),
                ([sw,    s,  d], [ (s + sw), 0.0,  0.0]),
            ]

        else:  # triangle
            R       = self.HOLE_TRI_R
            y_apex  =  R
            y_base  = -R / 2.0
            x_half  =  R * math.sqrt(3.0) / 2.0
            t_cap   = (hw - R) / 2.0
            t_side  = (hw - x_half) / 2.0
            h_mid   = (y_apex - y_base) / 2.0
            parts = [
                ([hw,         t_cap / 2,         d], [0.0,  y_apex + t_cap / 2, 0.0]),
                ([hw,         t_cap / 2,         d], [0.0,  y_base - t_cap / 2, 0.0]),
                ([t_side / 2, h_mid + t_cap,     d], [-(x_half + t_side / 2), (y_apex + y_base) / 2, 0.0]),
                ([t_side / 2, h_mid + t_cap,     d], [ (x_half + t_side / 2), (y_apex + y_base) / 2, 0.0]),
            ]

        for (he, lp) in parts:
            wp  = pos + rot @ np.array(lp)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=c)
            col = (p.createCollisionShape(p.GEOM_BOX, halfExtents=he)
                   if collidable else -1)
            bid = p.createMultiBody(0, col, vis, wp.tolist(), list(world_orn))
            bids.append(bid)
        return bids

    # ═══════════════════════════════════════════════════════════
    # INCLINED SURFACE
    # Normal  -> shape="square"   -> square hole
    # 15/30/45 -> shape="cylinder" -> circular hole
    # ═══════════════════════════════════════════════════════════
    def create_inclined_surface(self, pos, angle_deg, label, color_rgb, shape="cylinder"):
        angle_rad = math.radians(angle_deg)
        orn = p.getQuaternionFromEuler([angle_rad, 0, 0])
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        pos = np.array(pos)

        well_hd = 0.110       # deeper well so peg goes fully in
        fringe  = 0.02

        lbids = []
        hole_bids = self._make_hole_frame(shape,
                                          pos + rot @ np.array([0, fringe, 0]),
                                          orn, well_hd, color_rgb)
        for bid in hole_bids:
            self._surface_body_ids.append(bid)
            lbids.append(bid)

        ped_h = pos[2] - well_hd
        if ped_h > 0.01:
            bid = p.createMultiBody(0,
                p.createCollisionShape(p.GEOM_BOX,
                                       halfExtents=[0.15, 0.13, max(0.01, ped_h / 2)]),
                p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=[0.15, 0.13, max(0.01, ped_h / 2)],
                                    rgbaColor=[0.18, 0.18, 0.20, 1]),
                [pos[0], pos[1], ped_h / 2])
            self._surface_body_ids.append(bid)
            lbids.append(bid)

        self._surface_label_bodies[label] = lbids
        hole_pos = pos + rot @ np.array([0, fringe, 0])
        self.surfaces[label] = {
            "label":  label, "type": "inclined",
            "pos":    pos,   "orn":  orn,   "angle": angle_deg,
            "hole_pos": hole_pos, "shape": shape,
            "conjugate_profile": self._get_conj_profile(shape, angle_deg),
        }
        print(f"SURFACE [{label}]: shape={shape} angle={angle_deg}°", flush=True)

    @staticmethod
    def _get_conj_profile(shape, angle_deg):
        profiles = {
            "cylinder": {"contact_type": "line_contact",
                         "closure":      "form_closure",
                         "desc":         "V-groove on cylinder",
                         "min_contacts": 2},
            "square":   {"contact_type": "face_contact",
                         "closure":      "form_closure",
                         "desc":         "Flat face on square",
                         "min_contacts": 2},
            "triangle": {"contact_type": "edge_vertex_contact",
                         "closure":      "partial_form_closure",
                         "desc":         "Edge-vertex on triangle",
                         "min_contacts": 3},
        }
        pr = profiles.get(shape, profiles["cylinder"]).copy()
        pr["angle_deg"] = angle_deg
        pr["stability"] = max(0.0, 1.0 - (angle_deg / 90.0) * 0.5)
        return pr

    # ═══════════════════════════════════════════════════════════
    # SCULPTURED SURFACE — triangle peg + TRIANGLE hole (enlarged)
    # ═══════════════════════════════════════════════════════════
    def create_sculptured_surface(self, pos, label, surface_type, color_rgb,
                                  peg_shape="triangle"):
        orn      = [0, 0, 0, 1]
        mesh_pos = [pos[0], pos[1], pos[2] + 0.04]

        hole_r_uv = 0.15   # enlarged UV radius for easy insertion

        body_id, surface_obj, meta = ss.build_station(
            surface_type=surface_type, position=mesh_pos, orn_quat=orn,
            rgba=[*color_rgb, 0.95], hole_uv=(0.5, 0.5),
            hole_r=hole_r_uv, hole_shape=peg_shape, res=50,
            name=f"sculpt_{label.replace(' ','_').lower()}")
        self._surface_body_ids.append(body_id)
        lbids = [body_id]

        pos_np  = np.array(pos)
        well_hd = 0.120   # deeper — triangle peg goes fully in
        wbids   = self._make_hole_frame("triangle", pos_np, tuple(orn), well_hd, color_rgb)
        for bid in wbids:
            self._surface_body_ids.append(bid)
            lbids.append(bid)

        ped_h = pos[2] - well_hd
        if ped_h > 0.01:
            bid = p.createMultiBody(0,
                p.createCollisionShape(p.GEOM_BOX,
                                       halfExtents=[0.15, 0.15, max(0.01, ped_h / 2)]),
                p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=[0.15, 0.15, max(0.01, ped_h / 2)],
                                    rgbaColor=[0.12, 0.12, 0.15, 1]),
                [pos[0], pos[1], ped_h / 2])
            self._surface_body_ids.append(bid)
            lbids.append(bid)

        self._surface_label_bodies[label] = lbids
        self.surfaces[label] = {
            "label": label, "type": "sculptured", "surface_type": surface_type,
            "pos": pos_np, "orn": orn, "angle": 0,
            "hole_pos": meta["hole_world"], "shape": peg_shape,
            "insertion_normal": meta["insertion_normal"],
            "curvature": meta["curvature"],
            "conjugate_profile": meta["conjugate_profile"],
            "_surface_obj": surface_obj,
        }
        print(f"SCULPTURED [{label}]: peg_shape={peg_shape}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # PEG MAGAZINE
    # ═══════════════════════════════════════════════════════════
    def create_peg_magazine(self, pos):
        sx, sy, z = pos
        shapes = ["cylinder", "square", "triangle", "cylinder"]
        n      = len(shapes)

        slot_w   = 0.120
        slot_d   = 0.112
        wall_t   = 0.024
        well_dep = 0.090
        wall_abv = 0.128
        floor_h  = 0.012
        peg_h    = self.PEG_H
        pitch    = slot_w + wall_t

        rx_min = sx - slot_w / 2 - wall_t
        rx_max = sx + (n - 1) * pitch + slot_w / 2 + wall_t
        rcx    = (rx_min + rx_max) / 2
        rhx    = (rx_max - rx_min) / 2
        rhy    = slot_d / 2 + wall_t
        tot_h  = well_dep + wall_abv + floor_h
        wcz    = z - well_dep - floor_h + tot_h / 2

        wc = [0.30, 0.30, 0.38, 1]
        fc = [0.10, 0.10, 0.14, 1]
        cc = [0.60, 0.60, 0.68, 1]
        dc = [0.18, 0.18, 0.22, 1]
        lc = [0.55, 0.48, 0.00, 0.9]

        def _box(hx, hy, hz, pos3, rgba=wc):
            bid = p.createMultiBody(0,
                p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz]),
                p.createVisualShape(p.GEOM_BOX,    halfExtents=[hx, hy, hz],
                                    rgbaColor=rgba),
                pos3)
            self._magazine_body_ids.append(bid)
            return bid

        _box(rhx, rhy, floor_h / 2, [rcx, sy, z - well_dep + floor_h / 2], fc)
        fy = sy + slot_d / 2 + wall_t / 2
        by = sy - slot_d / 2 - wall_t / 2
        for wy in [fy, by]:
            _box(rhx, wall_t / 2, tot_h / 2, [rcx, wy, wcz])
        for ex in [rx_min - wall_t / 2, rx_max + wall_t / 2]:
            _box(wall_t / 2, rhy, tot_h / 2, [ex, sy, wcz])
        for i in range(1, n):
            wx = sx + i * pitch - slot_w / 2 - wall_t / 2
            _box(wall_t / 2, rhy, tot_h / 2, [wx, sy, wcz])
        dsh = 0.005
        for i in range(n):
            px  = sx + i * pitch
            b   = p.createMultiBody(0, -1,
                p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=[slot_w / 2, slot_d / 2, dsh / 2],
                                    rgbaColor=dc),
                [px, sy, z - dsh / 2])
            self._magazine_body_ids.append(b)
        for i in range(n):
            px = sx + i * pitch
            b  = p.createMultiBody(0, -1,
                p.createVisualShape(p.GEOM_CYLINDER, radius=0.030, length=0.004,
                                    rgbaColor=lc),
                [px, sy, z + 0.002])
            self._magazine_body_ids.append(b)
        cap_z = z + wall_abv + 0.004
        for wy in [fy, by]:
            _box(rhx + 0.002, wall_t / 2 + 0.001, 0.004, [rcx, wy, cap_z], cc)
        for ex in [rx_min - wall_t / 2, rx_max + wall_t / 2]:
            _box(wall_t / 2, rhy + wall_t, 0.004, [ex, sy, cap_z], cc)

        floor_top = z - well_dep + floor_h
        peg_ids, slot_pos = [], {}
        for i, sh in enumerate(shapes):
            px  = sx + i * pitch
            pcz = floor_top + peg_h / 2
            dp  = [px, sy, pcz]
            pid = self.create_peg(dp, sh)
            peg_ids.append(pid)
            slot_pos[pid] = dp

        # Guard rails so pegs stay upright and only the top half is exposed.
        # These must be collidable, otherwise the rack looks solid but won't
        # actually prevent pegs from tipping or sliding out.
        guard_inner = 0.038
        guard_t     = 0.012
        guard_h     = max(0.078, peg_h * 0.88)
        gz          = floor_top + guard_h / 2
        guard_half = [guard_inner + guard_t, guard_t / 2, guard_h / 2]
        side_half = [guard_t / 2, guard_inner, guard_h / 2]
        guard_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=guard_half)
        side_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=side_half)
        guard_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=guard_half,
                                        rgbaColor=[0.16, 0.16, 0.20, 1])
        side_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=side_half,
                                       rgbaColor=[0.16, 0.16, 0.20, 1])
        for i in range(n):
            px = sx + i * pitch
            for sy_sign in (-1, 1):
                b = p.createMultiBody(0, guard_col, guard_vis,
                                      [px, sy + sy_sign * (guard_inner + guard_t / 2), gz])
                self._magazine_body_ids.append(b)
            for sx_sign in (-1, 1):
                b = p.createMultiBody(0, side_col, side_vis,
                                      [px + sx_sign * (guard_inner + guard_t / 2), sy, gz])
                self._magazine_body_ids.append(b)

        print(f"MAGAZINE: {n} slots", flush=True)
        return peg_ids, shapes, slot_pos

    # ────────────────────────────────────────────────────────────────
    # RECOVERY STATION (drop funnel)
    # ────────────────────────────────────────────────────────────────
    def create_recovery_station(self, pos, label="RECOVERY"):
        cx, cy, z = pos
        inner = 0.055
        wall_t = 0.012
        wall_h = 0.18
        base_h = 0.006

        ids = []
        wc = [0.14, 0.14, 0.18, 1]
        bc = [0.08, 0.08, 0.10, 1]

        # Base plate
        bid = p.createMultiBody(0,
            p.createCollisionShape(p.GEOM_BOX, halfExtents=[inner + wall_t, inner + wall_t, base_h/2]),
            p.createVisualShape(p.GEOM_BOX,    halfExtents=[inner + wall_t, inner + wall_t, base_h/2],
                                rgbaColor=bc),
            [cx, cy, z - base_h / 2])
        ids.append(bid); self._magazine_body_ids.append(bid)

        # Four walls (square funnel/tube)
        for sx, sy, hx, hy in [
            (0,  1, inner + wall_t, wall_t / 2),
            (0, -1, inner + wall_t, wall_t / 2),
            (1,  0, wall_t / 2, inner),
            (-1, 0, wall_t / 2, inner),
        ]:
            bid = p.createMultiBody(0,
                p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, wall_h / 2]),
                p.createVisualShape(p.GEOM_BOX,    halfExtents=[hx, hy, wall_h / 2],
                                    rgbaColor=wc),
                [cx + sx * (inner + wall_t / 2), cy + sy * (inner + wall_t / 2), z + wall_h / 2])
            ids.append(bid); self._magazine_body_ids.append(bid)

        return {
            "label": label,
            "center": [cx, cy, z],
            "drop_pos": [cx, cy, z + wall_h * 0.85],
            "pick_pos": [cx, cy, z + wall_h * 0.35],
            "body_ids": ids,
        }

    # ═══════════════════════════════════════════════════════════
    # TRIANGLE PRISM MESH
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def _tri_mesh(radius, height):
        R, h = float(radius), float(height)
        y    = R * math.sqrt(3.0) / 2.0
        v2   = [(R, 0.0), (-R / 2, y), (-R / 2, -y)]
        verts = [(x, yy, -h / 2) for (x, yy) in v2] + \
                [(x, yy,  h / 2) for (x, yy) in v2]
        idx   = [0, 2, 1, 3, 4, 5,
                 0, 1, 4, 0, 4, 3,
                 1, 2, 5, 1, 5, 4,
                 2, 0, 3, 2, 3, 5]
        return verts, idx

    # ═══════════════════════════════════════════════════════════
    # PEG FACTORY
    # ═══════════════════════════════════════════════════════════
    def create_peg(self, pos, shape="cylinder"):
        h = self.PEG_H
        if shape == "cylinder":
            col = p.createCollisionShape(p.GEOM_CYLINDER,
                                         radius=self.CYL_R, height=h)
            vis = p.createVisualShape(p.GEOM_CYLINDER,
                                      radius=self.CYL_R, length=h,
                                      rgbaColor=[1, 0, 0.4, 1],
                                      specularColor=[1, 1, 1])
        elif shape == "square":
            col = p.createCollisionShape(p.GEOM_BOX,
                                         halfExtents=[self.SQ_H, self.SQ_H, h / 2])
            vis = p.createVisualShape(p.GEOM_BOX,
                                      halfExtents=[self.SQ_H, self.SQ_H, h / 2],
                                      rgbaColor=[1, 0.8, 0, 1],
                                      specularColor=[1, 1, 1])
        else:  # triangle
            verts, idx = self._tri_mesh(self.TRI_R, h)
            col = p.createCollisionShape(p.GEOM_MESH, vertices=verts, indices=idx)
            vis = p.createVisualShape(p.GEOM_MESH,    vertices=verts, indices=idx,
                                      rgbaColor=[0, 0.8, 1, 1],
                                      specularColor=[1, 1, 1])

        pid = p.createMultiBody(0.08, col, vis, pos)
        p.changeDynamics(pid, -1,
                         lateralFriction=8.0, spinningFriction=2.5,
                         rollingFriction=1.4, linearDamping=0.999,
                         angularDamping=0.999, restitution=0.0,
                         contactStiffness=42000, contactDamping=3200,
                         ccdSweptSphereRadius=0.005)
        return pid

    # ═══════════════════════════════════════════════════════════
    # CONVEYOR BELT
    # orange=cylinder+CIRCULAR  blue=triangle+TRIANGLE  green=square+SQUARE
    # ═══════════════════════════════════════════════════════════
    def create_conveyor_belt(self, pos, z_table):
        cx, cy, cz = pos
        belt_len = 0.62
        belt_w   = 0.15
        belt_h   = 0.015
        rail_h   = 0.060
        rail_t   = 0.009
        peg_h    = self.PEG_H

        self._conveyor_body_ids = []

        color_defs = [
            {"name": "orange", "label": "ORANGE",
             "rgba": [1.0, 0.5, 0.0, 1.0], "plate": [0.9, 0.45, 0.0, 0.8],
             "shape": "cylinder"},
            {"name": "blue",   "label": "BLUE",
             "rgba": [0.0, 0.3, 1.0, 1.0], "plate": [0.0, 0.25, 0.9, 0.8],
             "shape": "triangle"},
            {"name": "green",  "label": "GREEN",
             "rgba": [0.0, 0.8, 0.2, 1.0], "plate": [0.0, 0.7, 0.15, 0.8],
             "shape": "square"},
        ]

        # Support frame
        fh = (cz - belt_h) / 2
        if fh > 0.02:
            for sx in [-1, 1]:
                for sy2 in [-1, 1]:
                    fx  = cx + sx * (belt_len / 2 - 0.04)
                    fy  = cy + sy2 * (belt_w / 2 + rail_t)
                    bid = p.createMultiBody(0,
                        p.createCollisionShape(p.GEOM_BOX,
                                               halfExtents=[0.02, 0.02, fh]),
                        p.createVisualShape(p.GEOM_BOX,
                                            halfExtents=[0.02, 0.02, fh],
                                            rgbaColor=[0.2, 0.2, 0.25, 1]),
                        [fx, fy, fh])
                    self._conveyor_body_ids.append(bid)

        # Belt
        bid = p.createMultiBody(0,
            p.createCollisionShape(p.GEOM_BOX,
                                   halfExtents=[belt_len / 2, belt_w / 2, belt_h / 2]),
            p.createVisualShape(p.GEOM_BOX,
                                halfExtents=[belt_len / 2, belt_w / 2, belt_h / 2],
                                rgbaColor=[0.08, 0.08, 0.10, 1]),
            [cx, cy, cz])
        self._conveyor_body_ids.append(bid)

        # Rails
        for side in [-1, 1]:
            ry  = cy + side * (belt_w / 2 + rail_t / 2)
            bid = p.createMultiBody(0,
                p.createCollisionShape(p.GEOM_BOX,
                                       halfExtents=[belt_len / 2, rail_t / 2, rail_h / 2]),
                p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=[belt_len / 2, rail_t / 2, rail_h / 2],
                                    rgbaColor=[0.35, 0.35, 0.40, 1]),
                [cx, ry, cz + rail_h / 2])
            self._conveyor_body_ids.append(bid)

        # Rollers
        for i in range(8):
            rx = cx - belt_len / 2 + (i + 0.5) * (belt_len / 8)
            p.createMultiBody(0, -1,
                p.createVisualShape(p.GEOM_CYLINDER, radius=0.006, length=belt_w,
                                    rgbaColor=[0.15, 0.15, 0.18, 0.6]),
                [rx, cy, cz + belt_h / 2 + 0.001],
                p.getQuaternionFromEuler([math.pi / 2, 0, 0]))

        # Drive wheels
        for sx in [-1, 1]:
            p.createMultiBody(0, -1,
                p.createVisualShape(p.GEOM_CYLINDER, radius=0.032,
                                    length=belt_w + 0.02,
                                    rgbaColor=[0.25, 0.25, 0.30, 1]),
                [cx + sx * (belt_len / 2), cy, cz],
                p.getQuaternionFromEuler([math.pi / 2, 0, 0]))

        # Pegs on belt
        peg_offsets = [-0.12, 0.0, 0.12]
        belt_top    = cz + belt_h / 2
        peg_pids, peg_pos, peg_shapes = [], {}, {}
        zones = {}

        pickup_inner = 0.046
        pickup_wall_t = 0.010
        pickup_wall_h = 0.086
        pickup_wall_z = belt_top + pickup_wall_h / 2

        for i, cd in enumerate(color_defs):
            px  = cx + peg_offsets[i]
            py  = cy
            pcz = belt_top + peg_h / 2 + 0.003
            sh  = cd["shape"]

            if sh == "square":
                col = p.createCollisionShape(p.GEOM_BOX,
                                             halfExtents=[self.SQ_H, self.SQ_H, peg_h / 2])
                vis = p.createVisualShape(p.GEOM_BOX,
                                          halfExtents=[self.SQ_H, self.SQ_H, peg_h / 2],
                                          rgbaColor=cd["rgba"], specularColor=[1, 1, 1])
            elif sh == "triangle":
                verts, idx = self._tri_mesh(self.TRI_R, peg_h)
                col = p.createCollisionShape(p.GEOM_MESH, vertices=verts, indices=idx)
                vis = p.createVisualShape(p.GEOM_MESH,    vertices=verts, indices=idx,
                                          rgbaColor=cd["rgba"], specularColor=[1, 1, 1])
            else:  # cylinder
                col = p.createCollisionShape(p.GEOM_CYLINDER,
                                             radius=self.CYL_R, height=peg_h)
                vis = p.createVisualShape(p.GEOM_CYLINDER,
                                          radius=self.CYL_R, length=peg_h,
                                          rgbaColor=cd["rgba"], specularColor=[1, 1, 1])

            pid = p.createMultiBody(0.06, col, vis, [px, py, pcz])
            p.changeDynamics(pid, -1,
                             lateralFriction=8.0, spinningFriction=2.5,
                             rollingFriction=1.4, linearDamping=0.9995,
                             angularDamping=0.9995, restitution=0.0,
                             contactStiffness=46000, contactDamping=3600,
                             ccdSweptSphereRadius=0.008)
            peg_pids.append(pid)
            peg_pos[pid]    = [px, py, pcz]
            peg_shapes[pid] = sh

            p.createMultiBody(0, -1,
                p.createVisualShape(p.GEOM_CYLINDER, radius=0.036, length=0.004,
                                    rgbaColor=cd["plate"]),
                [px, py, belt_top + 0.001])

            front_back_half = [pickup_inner + pickup_wall_t, pickup_wall_t / 2, pickup_wall_h / 2]
            side_half = [pickup_wall_t / 2, pickup_inner, pickup_wall_h / 2]
            front_back_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=front_back_half)
            side_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=side_half)
            front_back_vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=front_back_half,
                rgbaColor=[0.18, 0.18, 0.22, 1.0],
            )
            side_vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=side_half,
                rgbaColor=[0.18, 0.18, 0.22, 1.0],
            )
            for sign in (-1, 1):
                bid = p.createMultiBody(
                    0,
                    front_back_col,
                    front_back_vis,
                    [px, py + sign * (pickup_inner + pickup_wall_t / 2), pickup_wall_z],
                )
                self._conveyor_body_ids.append(bid)
            for sign in (-1, 1):
                bid = p.createMultiBody(
                    0,
                    side_col,
                    side_vis,
                    [px + sign * (pickup_inner + pickup_wall_t / 2), py, pickup_wall_z],
                )
                self._conveyor_body_ids.append(bid)

        # Insertion holes — shape-matched, enlarged deep wells
        hole_y  = cy - belt_w / 2 - 0.18
        hole_z  = cz + 0.18
        well_hd = 0.130   # very deep — pegs go fully in

        hole_pos_map = {}
        for i, cd in enumerate(color_defs):
            hx  = cx + peg_offsets[i]
            hy  = hole_y
            hz  = hole_z
            sh  = cd["shape"]
            rgb = cd["rgba"][:3]

            # shape-matched hole
            # Conveyor sort holes are guidance visuals only. Collision walls were
            # shoving non-cylindrical pegs sideways during insertion retries.
            hbids = self._make_hole_frame(sh, np.array([hx, hy, hz]),
                                          [0, 0, 0, 1], well_hd, rgb,
                                          collidable=False)
            for bid in hbids:
                self._conveyor_body_ids.append(bid)

            # Pedestal
            ped = max(0.01, hz - well_hd)
            bid = p.createMultiBody(0,
                p.createCollisionShape(p.GEOM_BOX,
                                       halfExtents=[0.14, 0.14, ped / 2]),
                p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=[0.14, 0.14, ped / 2],
                                    rgbaColor=[0.12, 0.12, 0.15, 1]),
                [hx, hy, ped / 2])
            self._conveyor_body_ids.append(bid)

            # Colour ring on top
            p.createMultiBody(0, -1,
                p.createVisualShape(p.GEOM_CYLINDER, radius=0.048, length=0.005,
                                    rgbaColor=list(cd["rgba"])),
                [hx, hy, hz + well_hd + 0.003])

            hole_pos_map[cd["name"]] = [hx, hy, hz]
            zones[cd["name"]] = {
                "label":       cd["label"],
                "color_name":  cd["name"],
                "color":       cd["rgba"][:3],
                "hole_pos":    [hx, hy, hz],
                "approach_pos":[hx, hy, hz + 0.32],
            }

        self.conveyor = {
            "pos":           pos,
            "belt_z":        belt_top,
            "zones":         zones,
            "zone_order":    [cd["name"] for cd in color_defs],
            "peg_ids":       peg_pids,
            "peg_positions": peg_pos,
            "peg_shapes":    peg_shapes,
            "peg_colors":    {pid: cd["name"]
                              for pid, cd in zip(peg_pids, color_defs)},
            "hole_positions": hole_pos_map,
        }
        print(f"CONVEYOR: [{cx:.2f},{cy:.2f},{cz:.2f}]", flush=True)
        for pid, cd in zip(peg_pids, color_defs):
            pp = peg_pos[pid]; hp = hole_pos_map[cd["name"]]
            print(f"  {cd['label']}({cd['shape']}): peg={pp} hole={hp}", flush=True)
        return self.conveyor
