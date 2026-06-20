"""
robot_control.py
FIXES:
  - Tighter grip force for all shapes
  - Triangle peg: 30 deg roll to align jaw with widest face
  - Snap-to-EE before attach so constraint is zero-length
  - Higher IK force for insertion
"""
import pybullet as p
import pybullet_data
import numpy as np
import math


class RobotControl:
    HAND_LINK = 8
    EE_LINK  = 11
    FINGER1  = 9
    FINGER2  = 10
    NUM_ARM  = 7
    ARM_MAX_VEL = 10.0
    HOME_JOINTS = [0.0, -0.42, 0.0, -2.20, 0.0, 1.88, 0.78]
    # Link 11 in the Panda URDF is already the dedicated grasp target / TCP.
    GRASP_TCP_OFFSET = [0.0, 0.0, 0.0]

    # Maximum grip forces — prevent any peg slippage or dropping
    # Width tuned tightly per peg geometry for full jaw-centre contact
    CONJUGATE_GRIP = {
        "cylinder": {"width": 0.049, "force": 860000, "desc": "V-contact on cylinder"},
        "square":   {"width": 0.047, "force": 920000, "desc": "Flat-face on square"},
        "triangle": {"width": 0.046, "force": 900000, "desc": "Edge-vertex on triangle"},
    }

    def __init__(self, cid):
        self.cid              = cid
        self.robot_id         = None
        self.grasp_cid        = None
        self.conjugate_state  = None
        self._grasp_ref_local = None
        self.curvilinear_mode = True

    def set_curvilinear_mode(self, enabled=True):
        self.curvilinear_mode = bool(enabled)

    def load_robot(self, base_pos=(0, 0, 0), base_orn=None, color_rgba=None):
        if base_orn is None:
            base_orn = [0, 0, 0, 1]
        self.robot_id = p.loadURDF("franka_panda/panda.urdf",
                                    base_pos, base_orn, useFixedBase=True)
        for link_index in (self.HAND_LINK, self.FINGER1, self.FINGER2):
            try:
                p.changeDynamics(
                    self.robot_id,
                    link_index,
                    lateralFriction=14.0,
                    spinningFriction=4.2,
                    rollingFriction=1.3,
                    contactStiffness=62000,
                    contactDamping=4800,
                )
            except Exception:
                pass
        self._apply_home_pose(reset_state=True)
        if color_rgba is not None:
            try:
                vis = p.getVisualShapeData(self.robot_id)
                for si in vis:
                    p.changeVisualShape(self.robot_id, si[1], rgbaColor=list(color_rgba))
            except Exception:
                pass
        self.open_gripper()

    def open_gripper(self):
        self._fingers(0.08)
        self.conjugate_state = None

    def close_gripper(self):
        self._fingers(0.0)

    def _apply_home_pose(self, reset_state=False):
        for i, q in enumerate(self.HOME_JOINTS):
            if reset_state:
                p.resetJointState(self.robot_id, i, q)
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=q,
                force=700,
                maxVelocity=self.ARM_MAX_VEL * 0.9,
            )

    def reset_to_home(self):
        self._apply_home_pose(reset_state=True)
        self.open_gripper()
        self.conjugate_state = None

    def hold_home_pose(self):
        self._apply_home_pose(reset_state=False)

    def _fingers(self, width, force=2600):
        width = max(0.0, min(0.08, float(width)))
        hw    = width / 2.0
        p.setJointMotorControl2(self.robot_id, self.FINGER1,
                                p.POSITION_CONTROL,
                                targetPosition=hw, force=force)
        p.setJointMotorControl2(self.robot_id, self.FINGER2,
                                p.POSITION_CONTROL,
                                targetPosition=hw, force=force)

    # ── Conjugate Grasp ───────────────────────────────────────────────
    def conjugate_close(self, peg_shape, surface_angle_deg=0):
        cfg   = self.CONJUGATE_GRIP.get(peg_shape, self.CONJUGATE_GRIP["cylinder"])
        incline_ratio = min(1.0, max(0.0, float(surface_angle_deg) / 45.0))
        force = cfg["force"]
        width = cfg["width"]
        if peg_shape == "cylinder" and surface_angle_deg >= 30:
            force *= 1.28 + incline_ratio * 0.16
            width *= 0.94 - incline_ratio * 0.02
        grip_type = cfg["desc"]
        if self.curvilinear_mode:
            force *= 1.18 if surface_angle_deg < 30 else 1.24
            width *= 0.95 if surface_angle_deg < 30 else 0.91
            grip_type = "Curvilinear " + cfg["desc"]
        self._fingers(width, force=force)
        self.conjugate_state = {
            "shape":  peg_shape,
            "angle":  surface_angle_deg,
            "width":  width,
            "force":  force,
            "type":   grip_type,
        }
        print(f"CONJUGATE: {peg_shape} width={width:.4f}m force={force:.0f}N",
              flush=True)
        return self.conjugate_state

    def sculptured_conjugate_close(self, peg_shape, curvature_data,
                                   surface_type="dome"):
        cfg = self.CONJUGATE_GRIP.get(peg_shape, self.CONJUGATE_GRIP["cylinder"])
        K   = curvature_data.get("K", 0)
        H   = curvature_data.get("H", 0)
        cf  = (max(0.6, 1.0 - abs(H) * 2.0) if K > 0
               else (1.0 + abs(K) * 5.0 if K < 0 else 1.0))
        force = cfg["force"] * cf
        stab  = curvature_data.get("stability", 0.5)
        if isinstance(stab, dict):
            stab = stab.get("stability", 0.5)
        width = cfg["width"] * (1.0 - 0.08 * (1.0 - stab))
        grip_type = f"Sculptured {cfg['desc']} K={K:.4f}"
        if self.curvilinear_mode:
            force *= 1.16
            width *= 0.95
            grip_type = f"Curvilinear Sculptured {cfg['desc']} K={K:.4f}"
        self._fingers(width, force=force)
        self.conjugate_state = {
            "shape":        peg_shape,
            "angle":        0,
            "width":        width,
            "force":        force,
            "type":         grip_type,
            "K":            K,
            "H":            H,
            "surface_type": surface_type,
        }
        print(f"SCULPT CONJUGATE: {peg_shape} K={K:.4f} force={force:.0f}N",
              flush=True)
        return self.conjugate_state

    def compute_conjugate_orientation(self, surface_angle_deg, peg_shape="cylinder"):
        """
        Triangle peg: 30 deg roll so jaw aligns with widest triangle face.
        Cylinder / Square: 0 deg roll.
        """
        angle_rad   = math.radians(surface_angle_deg)
        roll_offset = {
            "cylinder": 0.0,
            "square":   0.0,
            "triangle": math.radians(30.0),
        }.get(peg_shape, 0.0)
        return p.getQuaternionFromEuler([math.pi + angle_rad, 0, roll_offset])

    def get_conjugate_info(self):
        return self.conjugate_state

    # ── Attach ────────────────────────────────────────────────────────
    def _snap_peg_to_ee(self, peg_id):
        ee_pos, ee_orn = self.get_ee_pose()
        try:
            gc = self.get_gripper_center()
            p.resetBasePositionAndOrientation(peg_id, gc.tolist(), ee_orn)
            p.resetBaseVelocity(peg_id, [0, 0, 0], [0, 0, 0])
        except Exception:
            pass

    def attach(self, obj_id, max_force=900000):
        if self.grasp_cid is not None:
            return
        # Attach at gripper center to keep peg inside jaws
        try:
            ee_pos, ee_orn = self.get_ee_pose()
            peg_pos, peg_orn = p.getBasePositionAndOrientation(obj_id)
            ip, io = p.invertTransform(ee_pos, ee_orn)
            parent_pos, parent_orn = p.multiplyTransforms(ip, io, peg_pos, peg_orn)
        except Exception:
            parent_pos = [0, 0, 0]
            parent_orn = [0, 0, 0, 1]
        self.grasp_cid = p.createConstraint(
            self.robot_id, self.EE_LINK, obj_id, -1,
            p.JOINT_FIXED, [0, 0, 0], parent_pos, [0, 0, 0],
            parent_orn, [0, 0, 0, 1])
        p.changeConstraint(self.grasp_cid, maxForce=max_force, erp=1.0)

    def set_grasp_force(self, force):
        if self.grasp_cid is not None:
            p.changeConstraint(self.grasp_cid, maxForce=force)

    def detach(self):
        if self.grasp_cid is not None:
            p.removeConstraint(self.grasp_cid)
            self.grasp_cid = None
        self.conjugate_state = None

    def get_gripper_center(self):
        if self.robot_id is None:
            return np.array([0.0, 0.0, 0.0])
        try:
            # The configured EE link is the Panda grasp target, which is the
            # correct TCP between the jaws. Using finger COMs causes visible
            # vertical offsets and "floating" pegs inside the gripper.
            return self.get_ee_pose()[0]
        except Exception:
            ee_pos, ee_orn = self.get_ee_pose()
            rot = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
            return ee_pos + rot @ np.array(self.GRASP_TCP_OFFSET)

    def get_ee_pose(self):
        s = p.getLinkState(self.robot_id, self.EE_LINK)
        return np.array(s[0]), s[1]

    def move_to(self, pos, orn=None, force=1000, ik_iterations=500):
        if orn is None:
            orn = p.getQuaternionFromEuler([math.pi, 0, 0])
        jq = p.calculateInverseKinematics(
            self.robot_id, self.EE_LINK, pos, orn,
            maxNumIterations=ik_iterations,
            residualThreshold=1e-6)
        for i in range(7):
            p.setJointMotorControl2(self.robot_id, i,
                                    p.POSITION_CONTROL, jq[i],
                                    force=force,
                                    maxVelocity=self.ARM_MAX_VEL)

    # ── Camera ────────────────────────────────────────────────────────
    def get_camera_image(self, width=1280, height=720):
        ee_pos, ee_orn = self.get_ee_pose()
        rot     = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
        forward = rot[:, 2]
        up      = rot[:, 1]
        right   = np.cross(up, forward)
        cam_pos = ee_pos - 0.05 * forward + 0.10 * right
        target  = ee_pos + 0.2 * forward
        vm  = p.computeViewMatrix(cam_pos, target, up)
        pm  = p.computeProjectionMatrixFOV(60.0, float(width)/height, 0.01, 2.0)
        res = p.getCameraImage(width, height, vm, pm,
                               renderer=p.ER_TINY_RENDERER,
                               flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        rgb_arr = np.array(res[2], dtype=np.uint8).reshape(height, width, 4)
        rgb_img = rgb_arr[:, :, :3]
        dep_arr = np.array(res[3], dtype=np.float32).reshape(height, width)
        far, near = 2.0, 0.01
        dep_lin   = far * near / (far - (far - near) * dep_arr)
        seg_arr   = np.array(res[4], dtype=np.int32).reshape(height, width)
        return rgb_img, dep_lin, seg_arr, vm, pm, dep_arr

    # ── Grasp Quality ─────────────────────────────────────────────────
    def compute_grasp_quality(self, peg_id=None):
        result = {
            "quality_score":    0.0,
            "num_contacts":     0,
            "has_bilateral":    False,
            "normal_force_N":   0.0,
            "friction_force_N": 0.0,
            "contact_spread_deg": 0.0,
            "force_balance":    0.0,
        }
        if self.robot_id is None:
            return result
        lc = p.getContactPoints(bodyA=self.robot_id, linkIndexA=self.FINGER1)
        rc = p.getContactPoints(bodyA=self.robot_id, linkIndexA=self.FINGER2)
        if peg_id is not None:
            lc = [c for c in lc if c[2] == peg_id]
            rc = [c for c in rc if c[2] == peg_id]
        all_c = list(lc) + list(rc)
        result["num_contacts"]  = len(all_c)
        result["has_bilateral"] = len(lc) > 0 and len(rc) > 0
        if not all_c:
            return result
        lf  = sum(c[9] for c in lc)
        rf  = sum(c[9] for c in rc)
        tot = lf + rf
        result["normal_force_N"]   = round(tot, 4)
        result["friction_force_N"] = round(
            sum(abs(c[10]) + abs(c[12]) for c in all_c), 4)
        if tot > 0:
            result["force_balance"] = round(1.0 - abs(lf - rf) / tot, 4)
        ns = [np.array(c[7]) for c in all_c]
        if len(ns) >= 2:
            mx = 0
            for i in range(len(ns)):
                for j in range(i + 1, len(ns)):
                    mx = max(mx, math.degrees(
                        math.acos(np.clip(np.dot(ns[i], ns[j]), -1, 1))))
            result["contact_spread_deg"] = round(mx, 2)
        s = 0.0
        if result["has_bilateral"]:
            s += 0.4
        s += min(0.2, result["num_contacts"] * 0.05)
        s += result["force_balance"] * 0.2
        s += min(0.2, result["contact_spread_deg"] / 180.0 * 0.2)
        result["quality_score"] = round(min(1.0, s), 4)
        return result

    def record_grasp_pose(self, peg_id):
        if self.robot_id is None or peg_id is None:
            self._grasp_ref_local = None
            return
        try:
            pp, po = p.getBasePositionAndOrientation(peg_id)
            ep, eo = self.get_ee_pose()
            ip, io = p.invertTransform(ep, eo)
            lp, lo = p.multiplyTransforms(ip, io, pp, po)
            self._grasp_ref_local = {"pos": np.array(lp), "orn": lo}
        except Exception:
            self._grasp_ref_local = None

    def get_grasp_reference_local(self):
        if self._grasp_ref_local is None:
            return None
        try:
            return {
                "pos": np.array(self._grasp_ref_local["pos"], dtype=float),
                "orn": tuple(self._grasp_ref_local.get("orn", [0, 0, 0, 1])),
            }
        except Exception:
            return None

    def check_peg_slip(self, peg_id, threshold_mm=5.0):
        result = {"slipped": False, "drift_mm": 0.0, "drift_pos": [0, 0, 0]}
        if self._grasp_ref_local is None or peg_id is None:
            return result
        try:
            pp, po = p.getBasePositionAndOrientation(peg_id)
            ep, eo = self.get_ee_pose()
            ip, io = p.invertTransform(ep, eo)
            cl, _  = p.multiplyTransforms(ip, io, pp, po)
            drift  = np.array(cl) - self._grasp_ref_local["pos"]
            dmm    = float(np.linalg.norm(drift) * 1000)
            result["drift_mm"]  = round(dmm, 3)
            result["drift_pos"] = [round(float(d) * 1000, 3) for d in drift]
            result["slipped"]   = dmm > threshold_mm
        except Exception:
            pass
        return result

    def get_wrist_forcetorque(self):
        if self.robot_id is None:
            return None
        try:
            state = p.getJointState(self.robot_id, 6)
            rf    = state[2]
            return {
                "fx": round(rf[0], 4), "fy": round(rf[1], 4), "fz": round(rf[2], 4),
                "mx": round(rf[3], 6), "my": round(rf[4], 6), "mz": round(rf[5], 6),
                "force_magnitude":  round(math.sqrt(rf[0]**2+rf[1]**2+rf[2]**2), 4),
                "torque_magnitude": round(math.sqrt(rf[3]**2+rf[4]**2+rf[5]**2), 6),
            }
        except Exception:
            return None

    # ── Full Telemetry ────────────────────────────────────────────────
    def get_joint_telemetry(self):
        if self.robot_id is None:
            return None
        joints = []
        for i in range(self.NUM_ARM):
            st   = p.getJointState(self.robot_id, i)
            info = p.getJointInfo(self.robot_id, i)
            joints.append({
                "index":           i,
                "name":            info[1].decode(),
                "position_rad":    round(st[0], 6),
                "position_deg":    round(math.degrees(st[0]), 4),
                "velocity_rad_s":  round(st[1], 6),
                "reaction_forces": {
                    "Fx": round(st[2][0],4), "Fy": round(st[2][1],4),
                    "Fz": round(st[2][2],4), "Mx": round(st[2][3],6),
                    "My": round(st[2][4],6), "Mz": round(st[2][5],6),
                },
                "applied_torque_Nm": round(st[3], 4),
            })
        for idx, name in [(self.FINGER1, "finger_left"), (self.FINGER2, "finger_right")]:
            st = p.getJointState(self.robot_id, idx)
            joints.append({
                "index": idx, "name": name,
                "position_m":   round(st[0], 6),
                "velocity_m_s": round(st[1], 6),
                "reaction_forces": {
                    "Fx": round(st[2][0],4), "Fy": round(st[2][1],4),
                    "Fz": round(st[2][2],4), "Mx": round(st[2][3],6),
                    "My": round(st[2][4],6), "Mz": round(st[2][5],6),
                },
                "applied_force_N": round(st[3], 4),
            })
        return joints

    def get_ee_telemetry(self):
        if self.robot_id is None:
            return None
        ls  = p.getLinkState(self.robot_id, self.EE_LINK, computeLinkVelocity=1)
        pos, oq, lv, av = ls[0], ls[1], ls[6], ls[7]
        oe  = p.getEulerFromQuaternion(oq)
        nj  = p.getNumJoints(self.robot_id)
        mov = [i for i in range(nj)
               if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]
        jp  = [p.getJointState(self.robot_id, i)[0] for i in mov]
        z   = [0.0] * len(mov)
        jl, ja = p.calculateJacobian(self.robot_id, self.EE_LINK,
                                      [0, 0, 0], jp, z, z)
        return {
            "position": {"x": round(pos[0],6), "y": round(pos[1],6),
                         "z": round(pos[2],6)},
            "orientation_quat": {"x": round(oq[0],6), "y": round(oq[1],6),
                                  "z": round(oq[2],6), "w": round(oq[3],6)},
            "orientation_euler_deg": {
                "roll":  round(math.degrees(oe[0]), 4),
                "pitch": round(math.degrees(oe[1]), 4),
                "yaw":   round(math.degrees(oe[2]), 4),
            },
            "linear_velocity":  {"vx": round(lv[0],6), "vy": round(lv[1],6),
                                  "vz": round(lv[2],6)},
            "angular_velocity": {"wx": round(av[0],6), "wy": round(av[1],6),
                                  "wz": round(av[2],6)},
            "linear_speed_m_s":    round(np.linalg.norm(lv), 6),
            "angular_speed_rad_s": round(np.linalg.norm(av), 6),
            "jacobian_linear":  [[round(v,6) for v in row] for row in jl],
            "jacobian_angular": [[round(v,6) for v in row] for row in ja],
        }

    def get_contact_telemetry(self):
        if self.robot_id is None:
            return None
        contacts = []
        for li in [self.FINGER1, self.FINGER2]:
            for cp in p.getContactPoints(bodyA=self.robot_id, linkIndexA=li):
                contacts.append({
                    "finger":              "left" if li == self.FINGER1 else "right",
                    "contact_body_id":     cp[2],
                    "position_on_finger":  {"x":round(cp[5][0],6),"y":round(cp[5][1],6),"z":round(cp[5][2],6)},
                    "position_on_object":  {"x":round(cp[6][0],6),"y":round(cp[6][1],6),"z":round(cp[6][2],6)},
                    "contact_normal":      {"nx":round(cp[7][0],6),"ny":round(cp[7][1],6),"nz":round(cp[7][2],6)},
                    "contact_distance_m":  round(cp[8], 8),
                    "normal_force_N":      round(cp[9], 4),
                    "lateral_friction1_N": round(cp[10], 4),
                    "friction1_direction": {"x":round(cp[11][0],6),"y":round(cp[11][1],6),"z":round(cp[11][2],6)},
                    "lateral_friction2_N": round(cp[12], 4),
                    "friction2_direction": {"x":round(cp[13][0],6),"y":round(cp[13][1],6),"z":round(cp[13][2],6)},
                })
        return contacts

    def get_dynamics_info(self):
        if self.robot_id is None:
            return None
        info = {}
        for i in range(self.NUM_ARM):
            dyn  = p.getDynamicsInfo(self.robot_id, i)
            jinf = p.getJointInfo(self.robot_id, i)
            info[jinf[1].decode()] = {
                "mass_kg":              round(dyn[0], 6),
                "lateral_friction":     round(dyn[1], 4),
                "local_inertia_diag":   [round(v, 8) for v in dyn[2]],
                "joint_lower_limit":    round(jinf[8], 4),
                "joint_upper_limit":    round(jinf[9], 4),
                "joint_max_force_N":    round(jinf[10], 4),
                "joint_max_velocity_rad_s": round(jinf[11], 4),
            }
        return info

    def get_grasp_constraint_telemetry(self):
        if self.grasp_cid is None:
            return None
        try:
            i = p.getConstraintInfo(self.grasp_cid)
            return {
                "constraint_id":    self.grasp_cid,
                "parent_body":      i[0], "parent_link": i[1],
                "child_body":       i[2], "child_link":  i[3],
                "joint_type":       i[4],
                "joint_axis":       list(i[5]),
                "parent_frame_pos": [round(v,6) for v in i[6]],
                "child_frame_pos":  [round(v,6) for v in i[7]],
                "parent_frame_orn": [round(v,6) for v in i[8]],
                "child_frame_orn":  [round(v,6) for v in i[9]],
                "max_force":        round(i[10], 4),
            }
        except Exception:
            return None
