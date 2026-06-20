"""
simulation_engine.py
FIXES:
  1. Camera positions pulled back — full robot always visible
  2. Speech deduplication — same sentence never repeats within 8 s
  3. Hole shapes: cylinder->circular, triangle->triangle, square->square
  4. Task completion data capture endpoint
"""
import pybullet as p
import pybullet_data
import time, math, threading, queue, random
import numpy as np
import cv2

from environment import Environment
from robot_control import RobotControl
from vision import VisionSystem


def _classify_surface(K, H):
    eps = 1e-6
    if abs(K) < eps and abs(H) < eps: return "PLANAR"
    if abs(K) < eps: return "CYLINDRICAL"
    return "ELLIPTIC" if K > eps else "HYPERBOLIC"


# ── Camera dicts — pulled back so full robot arm is visible ──────────
SURFACE_CAMERAS = {
    "Normal Surface": {
        "TOP":   ([0.55, -0.50, 2.80], [0.55, -0.30, 0.90], [0, 1, 0]),
        "FRONT": ([0.55, -2.80, 1.80], [0.55, -0.30, 0.90], [0, 0, 1]),
        "RIGHT": ([2.80, -0.30, 1.80], [0.55, -0.30, 0.90], [0, 0, 1]),
        "FOCUS": ([2.46, -2.58, 2.10], [0.55, -0.30, 0.90], [0, 0, 1]),
    },
    "15 DEG": {
        "TOP":   ([0.65, -0.50, 2.80], [0.65, -0.15, 0.95], [0, 1, 0]),
        "FRONT": ([0.65, -2.80, 1.80], [0.65, -0.15, 0.95], [0, 0, 1]),
        "RIGHT": ([2.80, -0.15, 1.80], [0.65, -0.15, 0.95], [0, 0, 1]),
        "FOCUS": ([2.48, -2.46, 2.14], [0.65, -0.15, 0.95], [0, 0, 1]),
    },
    "30 DEG": {
        "TOP":   ([0.72, -0.58, 3.18], [0.70,  0.00, 0.98], [0, 1, 0]),
        "FRONT": ([0.72, -3.35, 2.02], [0.70,  0.02, 0.98], [0, 0, 1]),
        "RIGHT": ([3.35,  0.02, 2.02], [0.70,  0.02, 0.98], [0, 0, 1]),
        "FOCUS": ([3.08, -2.82, 2.34], [0.70,  0.02, 0.98], [0, 0, 1]),
    },
    "45 DEG": {
        "TOP":   ([0.78, -0.64, 3.36], [0.75,  0.10, 1.04], [0, 1, 0]),
        "FRONT": ([0.78, -3.55, 2.16], [0.75,  0.12, 1.02], [0, 0, 1]),
        "RIGHT": ([3.58,  0.12, 2.16], [0.75,  0.12, 1.02], [0, 0, 1]),
        "FOCUS": ([3.34, -2.92, 2.52], [0.75,  0.12, 1.02], [0, 0, 1]),
    },
    "DOME": {
        "TOP":   ([0.55, -0.45, 2.80], [0.55,  0.20, 0.90], [0, 1, 0]),
        "FRONT": ([0.55, -2.70, 1.80], [0.55,  0.20, 0.90], [0, 0, 1]),
        "RIGHT": ([2.70,  0.20, 1.80], [0.55,  0.20, 0.90], [0, 0, 1]),
        "FOCUS": ([2.42, -2.24, 2.10], [0.55,  0.20, 0.90], [0, 0, 1]),
    },
}

SURFACE_FOVS = {
    "Normal Surface": 42.0,
    "15 DEG": 44.0,
    "30 DEG": 50.0,
    "45 DEG": 56.0,
    "DOME": 44.0,
}

SURFACE_VIEW_CROPS = {
    "Normal Surface": (0.06, 0.96, 0.08, 0.94),
    "15 DEG": (0.04, 0.97, 0.06, 0.95),
    "30 DEG": (0.02, 0.98, 0.05, 0.96),
    "45 DEG": (0.01, 0.99, 0.04, 0.97),
    "DOME": (0.03, 0.97, 0.05, 0.95),
}

DEFAULT_CAMERAS = {
    "TOP":   ([0.5,  0.0,  3.50], [0.5,  0.1, 0.8], [0, 1, 0]),
    "FRONT": ([0.5, -3.40, 1.90], [0.5,  0.0, 0.9], [0, 0, 1]),
    "RIGHT": ([3.40, 0.0,  1.90], [0.5,  0.0, 0.9], [0, 0, 1]),
    "FOCUS": ([2.90,-2.78, 2.18], [0.5,  0.0, 0.9], [0, 0, 1]),
}

PEG_TIP_OFFSETS = {
    "square":   0.055,
    "cylinder": 0.060,
    "triangle": 0.065,
}

SURFACE_SHAPE_MAP = {
    "Normal Surface": "square",
    "15 DEG":         "cylinder",
    "30 DEG":         "cylinder",
    "45 DEG":         "cylinder",
    "DOME":           "triangle",
}


class SimulationEngine:
    def __init__(self, world_offset=None, robot_base=None, robot_yaw=0.0,
                 robot_color=None, render_profile="full", enable_gui=None):
        self.command_queue = queue.Queue()
        self.current_frame = None
        self.single_view_frame = None
        self._color_sort_frame = None
        self._color_sort_placeholder_jpg  = None
        self._color_sort_placeholder_text = ""
        self.lock       = threading.Lock()
        self.fsm_state  = "IDLE"
        self.running    = False
        self.thread     = None
        self.camera_mode    = "ORBIT"
        self._focused_surface = None
        self.TABLE_H = 0.7
        self._world_offset = np.array(world_offset if world_offset is not None else [0.0, 0.0, 0.0], dtype=float)
        self._robot_base    = robot_base
        self._robot_yaw     = float(robot_yaw)
        self._robot_color   = robot_color
        self._render_profile = "compact" if str(render_profile).lower() == "compact" else "full"
        self._enable_gui = enable_gui
        ox, oy, oz = self._world_offset.tolist()
        self.WB_POS  = [0.5 + ox, 0.0 + oy, 0 + oz]
        self.STATIONS = {
            "MAGAZINE": [0.35 + ox, -0.45 + oy, self.TABLE_H + oz],
            "NORMAL":   [0.55 + ox, -0.30 + oy, self.TABLE_H + 0.20 + oz],
            "INC_15":   [0.65 + ox, -0.15 + oy, self.TABLE_H + 0.25 + oz],
            "INC_30":   [0.70 + ox,  0.0 + oy,  self.TABLE_H + 0.30 + oz],
            "INC_45":   [0.75 + ox,  0.10 + oy, self.TABLE_H + 0.38 + oz],
            "DOME":     [0.55 + ox,  0.20 + oy, self.TABLE_H + 0.20 + oz],
            "CONVEYOR": [0.55 + ox,  0.45 + oy, self.TABLE_H + 0.05 + oz],
        }
        self.available_pegs    = []
        self.current_peg_id    = None
        self.current_peg_shape = None
        self.DT = 1.0 / 240.0
        self.REALTIME_SCALE = 0.18
        self._force_history        = []
        self._grasp_quality        = None
        self._task_stats           = {"completed": 0, "failed": 0, "total": 0}
        self._priority_command     = None
        self._constraint_force_hold = 950000
        self._slip_data            = None
        self._insertion_force_peak = 0.0
        self._color_sort_queue     = []
        self._hole_vision_pos      = None
        self._color_sort_mode      = False
        self._sort_target_zone     = None
        self._sort_current_color   = None
        self._sort_pick_pos        = None
        self._conveyor_data        = None
        self._speech_queue   = []
        self._sound_queue    = []
        self._speech_seq     = 0
        self._sound_seq      = 0
        self._speech_history = []
        self._sound_history  = []
        # Speech dedup
        self._last_spoken_text = ""
        self._last_spoken_time = 0.0
        self._recovery_resume_state = None
        self._recovery_peg_pos      = None
        self._recovery_attempts     = 0
        self._max_recovery_attempts = 3
        self._grasp_retry_attempts      = 0
        self._sort_grasp_retry_attempts = 0
        self._insert_retry_attempts     = 0
        self._sort_insert_retry_attempts = 0
        self._max_grasp_retries    = 7
        self._max_insert_retries   = 7
        self._min_grasp_quality    = 0.40
        self._grasp_close_thresh   = 0.050
        self._vision_results       = {}
        self._last_busy_warn_ts    = 0.0
        self._return_peg_to_pick   = True
        self._pick_origin_pos      = None
        self._pick_origin_orn      = None
        self._sort_pick_origin_pos = None
        self._sort_pick_origin_orn = None
        self._last_insert_hole_world = None
        self._last_insert_normal     = None
        self._sim_data_log = []
        # Completed task snapshots for download
        self._completed_task_snapshots = []
        self._grasp_center_offset = 0.060
        self._insert_depth = 0.132
        self._grasp_center_offsets = {
            "cylinder": 0.060,
            "square":   0.060,
            "triangle": 0.061,
        }
        self._insert_depths = {
            "cylinder": 0.128,
            "square":   0.132,
            "triangle": 0.134,
        }
        self.peg_tip_offset = 0.06
        self.peg_slot_positions = {}
        self.used_pegs = []
        self._sim_steps = 0
        self._all_peg_ids = []
        self._color_sort_focus = False
        self._last_state_logged = None
        self._insert_mid_captured = False
        self._last_slip_log_ts = 0.0
        self._env_ref = None
        self._recovery_station = None
        self._active_task_label = None
        self._active_task_mode = "idle"
        self._pending_command_label = None
        self._pending_command_mode = "idle"
        self._preview_task_label = None
        self._preview_task_mode = "idle"
        self._target_surf_ref = None

    # ─────────────────────────────────────────────────────────────────
    def _log(self, event, data):
        self._sim_data_log.append({
            "ts": time.time(), "event": event,
            "state": self.fsm_state, "data": data
        })
        if len(self._sim_data_log) > 2000:
            self._sim_data_log = self._sim_data_log[-2000:]

    def get_sim_data_log(self):
        return list(self._sim_data_log)

    def get_completed_snapshots(self):
        return list(self._completed_task_snapshots)

    def _make_placeholder_jpg(self, text, width, height):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(img, "COLOR SORT WORKBENCH", (40, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.35, (255, 255, 255), 3, cv2.LINE_AA)
        y = 170
        for line in (text or "").split("\n"):
            cv2.putText(img, line, (40, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 212, 255), 2, cv2.LINE_AA)
            y += 55
        ret, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return enc.tobytes() if ret else None

    def _speak(self, text, priority="normal"):
        """Speak text — deduplicated: same sentence not repeated within 12 s."""
        now = time.time()
        if text == self._last_spoken_text and (now - self._last_spoken_time) < 12.0:
            return
        self._last_spoken_text = text
        self._last_spoken_time = now
        with self.lock:
            self._speech_seq += 1
            e = {
                "text": text,
                "priority": priority,
                "seq": self._speech_seq,
                "state": self.fsm_state,
                "ts": round(now, 6),
            }
            self._speech_queue.append(e)
            self._speech_history.append(e)
            if len(self._speech_history) > 250:
                self._speech_history = self._speech_history[-250:]

    def _sound(self, effect):
        with self.lock:
            self._sound_seq += 1
            e = {"effect": effect, "seq": self._sound_seq}
            self._sound_queue.append(e)
            self._sound_history.append(e)
            if len(self._sound_history) > 250:
                self._sound_history = self._sound_history[-250:]

    def _warn_busy(self, cmd):
        now = time.time()
        if now - self._last_busy_warn_ts < 5.0:
            return
        self._last_busy_warn_ts = now
        self._speak(f"Warning. Robot is busy in {self.fsm_state}. New command {cmd} was not started.", "high")
        self._sound("alert")

    def _check_peg_drop(self, robot):
        if self.current_peg_id is None:
            return None
        try:
            if getattr(robot, "grasp_cid", None) is not None:
                slip = robot.check_peg_slip(self.current_peg_id, threshold_mm=32.0)
                self._slip_data = slip
                if self._is_grasp_close(robot, self.current_peg_id, max_dist=0.12):
                    return None
            angle_deg = self._active_surface_angle()
            slip_threshold_mm = 14.0
            if angle_deg >= 45.0:
                slip_threshold_mm = 7.5
            elif angle_deg >= 30.0:
                slip_threshold_mm = 9.5
            slip = robot.check_peg_slip(self.current_peg_id, threshold_mm=slip_threshold_mm)
            if slip["slipped"]:
                try:
                    gc = robot.get_gripper_center()
                    pp, _ = p.getBasePositionAndOrientation(self.current_peg_id, self.cid)
                    dist = float(np.linalg.norm(np.array(pp) - gc))
                except Exception:
                    dist = 1.0
                recover_dist = 0.10
                if angle_deg >= 45.0:
                    recover_dist = 0.055
                elif angle_deg >= 30.0:
                    recover_dist = 0.065
                if dist <= recover_dist:
                    self._snap_attach(robot, self.current_peg_id)
                    try:
                        robot.set_grasp_force(self._hold_force_for_angle(angle_deg))
                    except Exception:
                        pass
                    self._log("slip_recovered", {"drift_mm": slip.get("drift_mm"), "dist": dist})
                    return None
                now = time.time()
                if now - self._last_slip_log_ts > 2.0:
                    self._log("slip_detected", slip)
                    self._last_slip_log_ts = now
                pos, _ = p.getBasePositionAndOrientation(self.current_peg_id, self.cid)
                routed = self._route_drop_to_recovery(self.current_peg_id, pos)
                self._log("peg_drop", {"pos": list(pos), "routed": True})
                return list(routed) if routed is not None else list(pos)
        except Exception:
            pass
        return None

    def _is_grasp_close(self, robot, peg_id, max_dist=None):
        if robot is None or peg_id is None:
            return False
        if max_dist is None:
            max_dist = self._grasp_close_thresh
        try:
            gc = self._expected_peg_center_from_robot(robot)
            pp, _ = p.getBasePositionAndOrientation(peg_id, self.cid)
            return float(np.linalg.norm(np.array(pp) - gc)) <= max_dist
        except Exception:
            return False

    def _active_surface_angle(self):
        if self._color_sort_mode:
            return 0.0
        surf = self._target_surf_ref or {}
        if surf.get("type") == "sculptured":
            return 0.0
        try:
            return float(surf.get("angle", 0) or 0.0)
        except Exception:
            return 0.0

    def _hold_force_for_angle(self, angle_deg=None):
        angle = self._active_surface_angle() if angle_deg is None else float(angle_deg)
        if self._color_sort_mode:
            return max(self._constraint_force_hold, 1000000)
        if angle >= 45.0:
            return 760000
        if angle >= 30.0:
            return 840000
        return self._constraint_force_hold

    def _grasp_center_offset_for_shape(self, shape=None):
        shape_name = (shape or self.current_peg_shape or "").lower()
        return float(self._grasp_center_offsets.get(shape_name, self._grasp_center_offset))

    def _insert_depth_for_shape(self, shape=None):
        shape_name = (shape or self.current_peg_shape or "").lower()
        return float(self._insert_depths.get(shape_name, self._insert_depth))

    def _color_sort_grasp_requirements(self, shape=None):
        shape_name = (shape or self.current_peg_shape or "cylinder").lower()
        return {
            "cylinder": {"score": 0.48, "force": 3.0, "spread": 35.0, "balance": 0.0, "window": 0.054},
            "square":   {"score": 0.46, "force": 2.4, "spread": 6.0, "balance": 0.0, "window": 0.060},
            "triangle": {"score": 0.52, "force": 3.0, "spread": 16.0, "balance": 0.0, "window": 0.058},
        }.get(shape_name, {"score": 0.50, "force": 3.0, "spread": 12.0, "balance": 0.0, "window": 0.053})

    def _grasp_window_for_angle(self, max_dist, angle_deg=None):
        angle = self._active_surface_angle() if angle_deg is None else float(angle_deg)
        if self._color_sort_mode:
            return min(max_dist, self._color_sort_grasp_requirements().get("window", 0.052))
        if angle >= 45.0:
            return min(max_dist, 0.032)
        if angle >= 30.0:
            return min(max_dist, 0.038)
        if angle >= 15.0:
            return min(max_dist, 0.050)
        return min(max_dist, 0.056)

    def _required_grasp_quality(self, angle_deg=None):
        angle = self._active_surface_angle() if angle_deg is None else float(angle_deg)
        base = self._min_grasp_quality
        if self._color_sort_mode:
            return max(base, self._color_sort_grasp_requirements().get("score", 0.48))
        if angle < 15.0:
            return base
        if angle < 30.0:
            return base + 0.03
        return base + 0.12 + min(0.12, (max(0.0, angle - 30.0) / 15.0) * 0.12)

    def _has_secure_grasp(self, robot, peg_id, angle_deg=None):
        quality = robot.compute_grasp_quality(peg_id) if robot is not None else {}
        self._grasp_quality = quality
        angle = self._active_surface_angle() if angle_deg is None else float(angle_deg)
        bilateral = bool(quality.get("has_bilateral"))
        contacts = int(quality.get("num_contacts", 0) or 0)
        score = float(quality.get("quality_score", 0.0) or 0.0)
        balance = float(quality.get("force_balance", 0.0) or 0.0)
        normal_force = float(quality.get("normal_force_N", 0.0) or 0.0)
        spread = float(quality.get("contact_spread_deg", 0.0) or 0.0)
        min_balance = 0.0
        min_force = 12.0
        min_spread = 4.0
        if self._color_sort_mode:
            sort_req = self._color_sort_grasp_requirements()
            min_balance = float(sort_req.get("balance", 0.0))
            min_force = float(sort_req.get("force", 3.0))
            min_spread = float(sort_req.get("spread", 10.0))
        elif angle >= 30.0:
            min_balance = 0.20
            min_force = 22.0
            min_spread = 16.0
        elif angle >= 15.0:
            min_force = 14.0
            min_spread = 6.0
        return (
            bilateral and
            contacts >= 2 and
            score >= self._required_grasp_quality(angle) and
            balance >= min_balance and
            normal_force >= min_force and
            spread >= min_spread
        )

    def _ensure_grasp(self, robot, peg_id, sub_step, max_dist=0.08, force_after=90):
        if robot is None or peg_id is None:
            return False
        angle_deg = self._active_surface_angle()
        strict_max_dist = self._grasp_window_for_angle(max_dist, angle_deg)
        relaxed_limit = self._grasp_window_for_angle(
            max(max_dist, self._grasp_close_thresh + 0.015), angle_deg
        )
        secure_grasp = self._has_secure_grasp(robot, peg_id, angle_deg)
        hold_force = self._hold_force_for_angle(angle_deg)
        if self._is_grasp_close(robot, peg_id, max_dist=strict_max_dist) and secure_grasp:
            self._snap_attach(robot, peg_id, hold_force=hold_force)
            try:
                robot.set_grasp_force(hold_force)
            except Exception:
                pass
            return True
        if (force_after is not None and
                sub_step >= force_after and
                self._is_grasp_close(robot, peg_id, max_dist=relaxed_limit) and
                secure_grasp):
            self._snap_attach(robot, peg_id, hold_force=hold_force)
            try:
                robot.set_grasp_force(hold_force)
            except Exception:
                pass
            return True
        if self._color_sort_mode and sub_step >= max(40, force_after or 0):
            quality = self._grasp_quality or robot.compute_grasp_quality(peg_id)
            if (self._is_grasp_close(robot, peg_id, max_dist=relaxed_limit) and
                    quality.get("has_bilateral") and
                    float(quality.get("quality_score", 0.0) or 0.0) >=
                    max(0.44, self._required_grasp_quality(angle_deg) - 0.06)):
                self._snap_attach(robot, peg_id, hold_force=hold_force)
                try:
                    robot.set_grasp_force(hold_force)
                except Exception:
                    pass
                return True
            shape_name = (self.current_peg_shape or "").lower()
            if shape_name in {"square", "triangle"}:
                contact_count = int(quality.get("num_contacts", 0) or 0)
                normal_force = float(quality.get("normal_force_N", 0.0) or 0.0)
                deterministic_window = max(relaxed_limit, self._color_sort_grasp_requirements(shape_name).get("window", 0.056))
                min_contacts = 1
                min_normal_force = 16.0
                if shape_name == "square":
                    min_contacts = 2
                    min_normal_force = 10.0
                if (sub_step >= max(72, force_after or 0) and
                        self._is_grasp_close(robot, peg_id, max_dist=deterministic_window) and
                        contact_count >= min_contacts and
                        normal_force >= min_normal_force):
                    self._snap_attach(robot, peg_id, hold_force=hold_force)
                    try:
                        robot.set_grasp_force(hold_force)
                    except Exception:
                        pass
                    return True
        return False

    def _get_insert_target_center(self, hole_world, normal=None):
        n = np.array(normal if normal is not None else [0, 0, 1], dtype=float)
        nn = float(np.linalg.norm(n))
        if nn < 1e-9:
            n = np.array([0, 0, 1], dtype=float)
        else:
            n = n / nn
        insert_depth = self._insert_depth_for_shape()
        return np.array(hole_world, dtype=float) + n * (self.peg_tip_offset - insert_depth)

    def _tool_axis_from_orn(self, orn):
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        return rot[:, 2]

    def _expected_peg_center_from_robot(self, robot, orn=None):
        ee_pos, ee_orn = robot.get_ee_pose()
        use_orn = orn if orn is not None else ee_orn
        ref = None
        try:
            ref = robot.get_grasp_reference_local()
        except Exception:
            ref = None
        if ref and ref.get("pos") is not None:
            rot = np.array(p.getMatrixFromQuaternion(use_orn)).reshape(3, 3)
            return np.array(ee_pos, dtype=float) + rot @ np.array(ref["pos"], dtype=float)
        axis = self._tool_axis_from_orn(use_orn)
        return np.array(ee_pos, dtype=float) + axis * self._grasp_center_offset_for_shape()

    def _ee_target_for_peg_center(self, peg_center, orn, robot=None):
        if robot is not None:
            try:
                ref = robot.get_grasp_reference_local()
            except Exception:
                ref = None
            if ref and ref.get("pos") is not None:
                rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
                return np.array(peg_center, dtype=float) - rot @ np.array(ref["pos"], dtype=float)
        axis = self._tool_axis_from_orn(orn)
        return np.array(peg_center, dtype=float) - axis * self._grasp_center_offset_for_shape()

    def _corrected_ee_target_for_center(self, robot, peg_id, desired_center, orn,
                                        normal=None, lateral_gain=0.95,
                                        axial_gain=0.40, max_lateral_step=0.020,
                                        max_axial_step=0.014):
        target = self._ee_target_for_peg_center(desired_center, orn, robot)
        if robot is None or peg_id is None:
            return target
        try:
            n = np.array(normal if normal is not None else [0, 0, 1], dtype=float)
            nn = float(np.linalg.norm(n))
            if nn < 1e-9:
                n = np.array([0, 0, 1], dtype=float)
            else:
                n = n / nn
            peg_pos, _ = p.getBasePositionAndOrientation(peg_id, self.cid)
            err = np.array(desired_center, dtype=float) - np.array(peg_pos, dtype=float)
            axial = float(np.dot(err, n))
            lateral_vec = err - axial * n
            lateral = float(np.linalg.norm(lateral_vec))
            if lateral > 1e-9:
                target = target + lateral_vec * min(
                    lateral_gain, max_lateral_step / max(lateral, 1e-9)
                )
            axial_step = float(np.clip(axial * axial_gain, -max_axial_step, max_axial_step))
            target = target + n * axial_step
        except Exception:
            pass
        return target

    def _get_insertion_tolerances(self, surface=None, sort_mode=False):
        if sort_mode:
            return {
                "lateral_tol": 0.008,
                "axial_tol": 0.024,
                "ready_lateral_tol": 0.006,
                "ready_axial_above_tol": 0.070,
                "ready_center_tol": 0.072,
            }
        if surface and surface.get("type") == "sculptured":
            return {
                "lateral_tol": 0.011,
                "axial_tol": 0.030,
                "ready_lateral_tol": 0.008,
                "ready_axial_above_tol": 0.052,
                "ready_center_tol": 0.060,
            }
        angle = float((surface or {}).get("angle", 0) or 0.0)
        if angle >= 30.0:
            lateral_tol = 0.0090 - min(0.0015, ((angle - 30.0) / 15.0) * 0.0015)
            axial_tol = 0.020 + min(0.004, (angle / 45.0) * 0.004)
            ready_axial = 0.046 + min(0.010, (angle / 45.0) * 0.010)
            ready_center = 0.045 + min(0.008, (angle / 45.0) * 0.008)
            return {
                "lateral_tol": lateral_tol,
                "axial_tol": axial_tol,
                "ready_lateral_tol": min(0.0080, lateral_tol + 0.0008),
                "ready_axial_above_tol": ready_axial,
                "ready_center_tol": ready_center,
            }
        lateral_tol = 0.010 + min(0.0020, (angle / 45.0) * 0.0020)
        axial_tol = 0.024 + min(0.006, (angle / 45.0) * 0.006)
        ready_axial = max(0.040, axial_tol + 0.010)
        return {
            "lateral_tol": lateral_tol,
            "axial_tol": axial_tol,
            "ready_lateral_tol": min(0.0085, lateral_tol),
            "ready_axial_above_tol": ready_axial,
            "ready_center_tol": ready_axial + 0.004,
        }

    def _evaluate_insertion_success(self, peg_id, hole_world, normal=None,
                                    lateral_tol=0.014, axial_tol=0.028):
        result = {
            "success": False,
            "lateral_error_mm": None,
            "axial_error_mm": None,
            "center_error_mm": None,
            "lateral_error_m": None,
            "axial_error_m": None,
            "center_error_m": None,
        }
        if peg_id is None or hole_world is None:
            return result
        try:
            n = np.array(normal if normal is not None else [0, 0, 1], dtype=float)
            nn = float(np.linalg.norm(n))
            if nn < 1e-9:
                n = np.array([0, 0, 1], dtype=float)
            else:
                n = n / nn
            peg_pos, _ = p.getBasePositionAndOrientation(peg_id, self.cid)
            target_center = self._get_insert_target_center(hole_world, n)
            err = np.array(peg_pos, dtype=float) - target_center
            axial = float(np.dot(err, n))
            lateral_vec = err - axial * n
            lateral = float(np.linalg.norm(lateral_vec))
            center = float(np.linalg.norm(err))
            result["lateral_error_m"] = lateral
            result["axial_error_m"] = axial
            result["center_error_m"] = center
            result["lateral_error_mm"] = round(lateral * 1000.0, 3)
            result["axial_error_mm"] = round(axial * 1000.0, 3)
            result["center_error_mm"] = round(center * 1000.0, 3)
            result["success"] = (lateral <= lateral_tol and abs(axial) <= axial_tol)
        except Exception as e:
            result["error"] = str(e)
        return result

    def _evaluate_insertion_ready(self, peg_id, hole_world, normal=None,
                                  lateral_tol=0.010,
                                  axial_below_tol=0.015,
                                  axial_above_tol=0.080,
                                  center_tol=0.090):
        result = self._evaluate_insertion_success(
            peg_id, hole_world, normal, lateral_tol=10.0, axial_tol=10.0)
        lateral = result.get("lateral_error_m")
        axial = result.get("axial_error_m")
        center = result.get("center_error_m")
        result["ready"] = bool(
            lateral is not None and axial is not None and center is not None and
            lateral <= lateral_tol and
            -float(axial_below_tol) <= axial <= axial_above_tol and
            center <= center_tol
        )
        return result

    def _settle_peg_in_hole(self, peg_id, hole_world, normal=None):
        if peg_id is None or hole_world is None:
            return
        try:
            pos = self._get_insert_target_center(hole_world, normal)
            peg_orn = [0, 0, 0, 1]
            robot = getattr(self, "_robot_ref", None)
            if robot is not None and robot.robot_id is not None:
                try:
                    _, peg_orn = robot.get_ee_pose()
                except Exception:
                    pass
            p.resetBasePositionAndOrientation(peg_id, pos.tolist(), peg_orn)
            p.resetBaseVelocity(peg_id, [0, 0, 0], [0, 0, 0])
        except Exception:
            pass

    def _route_drop_to_recovery(self, peg_id, fallback_pos=None):
        rec = self._recovery_station
        if rec is None or peg_id is None:
            return list(fallback_pos) if fallback_pos is not None else None
        try:
            drop_pos = rec.get("drop_pos", rec.get("center"))
            pick_pos = rec.get("pick_pos", rec.get("center"))
            p.resetBasePositionAndOrientation(peg_id, list(drop_pos), [0, 0, 0, 1])
            p.resetBaseVelocity(peg_id, [0, 0, 0], [0, 0, 0])
            return list(pick_pos)
        except Exception:
            return list(fallback_pos) if fallback_pos is not None else None

    def _detect_surface_hole_shape(self, vision, surface_label):
        if vision is None or surface_label is None:
            return None
        prev_focus = self._focused_surface
        try:
            self._focused_surface = surface_label
            img = self._get_scene_cam(640, 480, "TOP")
        finally:
            self._focused_surface = prev_focus
        if img is None:
            return None
        det = vision.detect_hole_shape(img)
        shape = det.get("shape")
        conf  = det.get("confidence", 0.0)
        if shape is None or conf < 0.60:
            return None
        if shape == "circle":
            return "cylinder"
        return shape

    def _set_robot_peg_collisions(self, robot, active_pid=None):
        if robot is None or robot.robot_id is None:
            return
        if not self._all_peg_ids:
            return
        try:
            links = list(range(-1, p.getNumJoints(robot.robot_id)))
        except Exception:
            links = [-1]
        for pid in self._all_peg_ids:
            if pid is None:
                continue
            enable = 1 if (active_pid is None or pid == active_pid) else 0
            for link in links:
                try:
                    p.setCollisionFilterPair(robot.robot_id, pid, link, -1, enableCollision=enable)
                except Exception:
                    pass

    def _set_robot_surface_collisions(self, robot, enable=True):
        env = self._env_ref
        if robot is None or robot.robot_id is None or env is None:
            return
        try:
            links = list(range(-1, p.getNumJoints(robot.robot_id)))
        except Exception:
            links = [-1]
        bodies = list(getattr(env, "_surface_body_ids", []))
        for bid in bodies:
            for link in links:
                try:
                    p.setCollisionFilterPair(
                        robot.robot_id, bid, link, -1,
                        enableCollision=1 if enable else 0)
                except Exception:
                    pass

    def _set_robot_conveyor_collisions(self, robot, enable=True):
        env = self._env_ref
        if robot is None or robot.robot_id is None or env is None:
            return
        try:
            links = list(range(-1, p.getNumJoints(robot.robot_id)))
        except Exception:
            links = [-1]
        bodies = list(getattr(env, "_conveyor_body_ids", []))
        for bid in bodies:
            for link in links:
                try:
                    p.setCollisionFilterPair(
                        robot.robot_id, bid, link, -1,
                        enableCollision=1 if enable else 0)
                except Exception:
                    pass

    def get_speech_events(self, since_speech=0, since_sound=0):
        with self.lock:
            sp = [e for e in self._speech_history if e["seq"] > since_speech]
            sn = [e for e in self._sound_history   if e["seq"] > since_sound]
            return {
                "speeches": sp, "sounds": sn,
                "speech_seq": self._speech_seq, "sound_seq": self._sound_seq
            }

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread  = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def get_frame(self):
        with self.lock:
            return self.current_frame

    def get_single_view_frame(self):
        with self.lock:
            return self.single_view_frame

    def send_command(self, cmd):
        if cmd in ("RESET", "E_STOP"):
            try:
                while True:
                    self.command_queue.get_nowait()
            except queue.Empty:
                pass
            self._pending_command_label = None
            self._pending_command_mode = "idle"
            self._priority_command = cmd
            return {"accepted": True, "state": self.fsm_state, "reason": "priority"}
        if self.fsm_state != "IDLE":
            self._warn_busy(cmd)
            return {"accepted": False, "state": self.fsm_state, "reason": "busy"}
        if self.fsm_state == "IDLE":
            if cmd == "COLOR_SORT":
                self._pending_command_label = "COLOR_SORT"
                self._pending_command_mode = "color_sort"
            elif cmd in ("Normal Surface", "15 DEG", "30 DEG", "45 DEG", "DOME"):
                self._pending_command_label = cmd
                self._pending_command_mode = "surface"
        self.command_queue.put(cmd)
        return {"accepted": True, "state": self.fsm_state, "reason": "queued"}

    def set_preview_task(self, task_label=None):
        task = (task_label or "").strip()
        if not task:
            self._preview_task_label = None
            self._preview_task_mode = "idle"
        elif task == "COLOR_SORT":
            self._preview_task_label = "COLOR_SORT"
            self._preview_task_mode = "color_sort"
        elif task in ("Normal Surface", "15 DEG", "30 DEG", "45 DEG", "DOME"):
            self._preview_task_label = task
            self._preview_task_mode = "surface"
        else:
            self._preview_task_label = None
            self._preview_task_mode = "idle"
        return {
            "preview_task": self._preview_task_label,
            "preview_mode": self._preview_task_mode,
            "state": self.fsm_state,
        }

    def get_status(self):
        return self.fsm_state

    def get_vision_data(self):
        return dict(self._vision_results) if self._vision_results else {}

    def get_grasp_quality_data(self):
        return dict(self._grasp_quality) if self._grasp_quality else {}

    def get_task_progress(self):
        s = dict(self._task_stats)
        s["queue_remaining"] = len(self._color_sort_queue)
        return s

    def get_force_history(self, n=50):
        return list(self._force_history[-n:])

    def get_slip_data(self):
        return dict(self._slip_data) if self._slip_data else {}

    def get_color_sort_status(self):
        s = self._task_stats
        if not self._color_sort_mode:
            return {
                "active": False, "target_zone": None, "current_color": None,
                "queue_remaining": 0, "completed": s["completed"],
                "failed": s["failed"], "total": s["total"]
            }
        z  = self._sort_target_zone
        zi = {
            "label":      z.get("label", ""),
            "color_name": z.get("color_name", ""),
            "color":      list(z["color"]) if z.get("color") else None
        } if z else None
        return {
            "active": True, "target_zone": zi,
            "current_color": self._sort_current_color,
            "queue_remaining": len(self._color_sort_queue),
            "completed": s["completed"], "failed": s["failed"], "total": s["total"]
        }

    def _state_hint_text(self):
        if self._slip_data and self._slip_data.get("slipped"):
            return "Slip detected. Stabilizing peg before the next move."
        if self.fsm_state == "IDLE" and self._pending_command_label:
            if self._pending_command_mode == "color_sort":
                return "Command queued: Color Sort"
            return f"Command queued: {self._pending_command_label}"
        mapping = {
            "IDLE": "System ready. Select a task to begin.",
            "PICK_APPROACH": "Moving above the selected peg.",
            "PICK_DOWN": "Lowering into the peg grasp window.",
            "GRASP": "Closing jaws around the peg.",
            "VERIFY_GRASP": "Checking whether the peg is centered between the jaws.",
            "REGRASP": "Repositioning for a tighter grasp.",
            "LIFT": "Lifting peg clear of the table.",
            "SCAN_APPROACH": "Moving toward the inspection pose.",
            "SCAN": "Scanning the target hole and surroundings.",
            "INSERT_APPROACH": "Approaching the target surface.",
            "INSERT_ALIGN": "Aligning peg with the hole center.",
            "INSERT_PUSH": "Pushing the peg into the hole.",
            "EXTRACT": "Verifying insertion and lifting away.",
            "RETURN_TO_PICK": "Returning peg to its source slot.",
            "RELEASE": "Opening jaws to release the peg.",
            "SORT_PICK_APPROACH": "Moving above the next conveyor peg.",
            "SORT_PICK_DOWN": "Lowering into the conveyor grasp window.",
            "SORT_GRASP": "Closing jaws around the conveyor peg.",
            "SORT_REGRASP": "Repositioning for a tighter conveyor grasp.",
            "SORT_LIFT": "Lifting peg from the conveyor.",
            "SORT_SCAN": "Checking color and target lane.",
            "SORT_APPROACH": "Moving toward the color-sort hole.",
            "SORT_ALIGN": "Aligning peg with the sort hole center.",
            "SORT_INSERT": "Inserting peg into the sort hole.",
            "SORT_EXTRACT": "Verifying sort insertion and lifting away.",
            "SORT_RETURN": "Returning peg to the conveyor slot.",
            "SORT_RELEASE": "Opening jaws to release the sorted peg.",
            "PEG_RECOVERY_APPROACH": "Recovering a dropped peg.",
            "PEG_RECOVERY_GRASP": "Regrasping the dropped peg.",
            "PEG_RECOVERY_LIFT": "Lifting the recovered peg.",
            "GO_HOME": "Returning robot to home position.",
        }
        return mapping.get(
            self.fsm_state,
            str(self.fsm_state or "IDLE").replace("_", " ").title()
        )

    def get_runtime_summary(self):
        zone = self._sort_target_zone or {}
        pending_active = self.fsm_state == "IDLE" and self._pending_command_label
        return {
            "state": self.fsm_state,
            "state_hint": self._state_hint_text(),
            "mode": self._pending_command_mode if pending_active else self._active_task_mode,
            "active_task": self._pending_command_label if pending_active else self._active_task_label,
            "preview_task": self._preview_task_label,
            "preview_mode": self._preview_task_mode,
            "focused_surface": self._focus_surface_label(),
            "current_surface": (self._target_surf_ref or {}).get("label"),
            "current_shape": self.current_peg_shape,
            "current_color": self._sort_current_color,
            "target_zone": zone.get("label"),
            "queue_remaining": len(self._color_sort_queue),
            "progress": self.get_task_progress(),
            "insertion_force_peak": round(float(self._insertion_force_peak or 0.0), 2),
            "slip": dict(self._slip_data) if self._slip_data else {},
        }

    def _get_surface_focus_extra_body_ids(self):
        extra_ids = set()
        for pid in list(self._all_peg_ids) + list(self.peg_slot_positions.keys()):
            if pid is not None and pid != self.current_peg_id:
                extra_ids.add(pid)
        return list(extra_ids)

    def _select_preferred_peg(self, candidates):
        if not candidates:
            return None
        def _score(item):
            _, pid, _ = item
            pos = self.peg_slot_positions.get(pid)
            if pos is None:
                try:
                    pos, _ = p.getBasePositionAndOrientation(pid, self.cid)
                except Exception:
                    pos = [999.0, 999.0, 999.0]
            return (float(pos[0]), abs(float(pos[1])), float(pos[2]))
        return min(candidates, key=_score)

    def _focus_surface_label(self):
        if self._color_sort_focus:
            return None
        active = self._focused_surface or (self._target_surf_ref or {}).get("label")
        if active:
            return active
        if self.fsm_state == "IDLE" and self._preview_task_mode == "surface":
            return self._preview_task_label
        return None

    def _restore_returned_peg(self, peg_id, peg_shape):
        if peg_id is None or not peg_shape:
            return
        if not any(pid == peg_id for pid, _ in self.available_pegs):
            self.available_pegs.append((peg_id, peg_shape))
        self.used_pegs = [(pid, sh) for pid, sh in self.used_pegs if pid != peg_id]

    def _get_normal(self, surf):
        if surf and surf.get("type") == "sculptured":
            n = np.array(surf["insertion_normal"], dtype=float)
            return n, math.degrees(math.acos(np.clip(n[2], -1, 1)))
        deg = surf.get("angle", 0) if surf else 0
        rad = math.radians(deg)
        return np.array([0, -math.sin(rad), math.cos(rad)]), deg

    def get_conjugate_data(self):
        data = {
            "active": False, "shape": self.current_peg_shape, "contact": None,
            "force": None, "width": None, "angle": None, "closure": None,
            "curvature_K": None, "curvature_H": None, "surface_type": None
        }
        robot = getattr(self, '_robot_ref', None)
        if robot:
            info = robot.get_conjugate_info()
            if info:
                data.update({
                    "active":  True,
                    "shape":   info.get("shape"),
                    "contact": info.get("type"),
                    "force":   round(info.get("force", 0), 1),
                    "width":   round(info.get("width", 0) * 1000, 1),
                    "angle":   info.get("angle"),
                    "closure": "FORM CLOSURE",
                    "curvature_K": round(info["K"], 5) if info.get("K") is not None else None,
                    "curvature_H": round(info.get("H", 0), 5) if info.get("K") is not None else None,
                    "surface_type": info.get("surface_type"),
                })
        return data

    def get_full_telemetry(self):
        robot = getattr(self, '_robot_ref', None)
        if robot is None or robot.robot_id is None:
            return {"error": "Robot not initialized"}
        try:
            return self._build_telemetry(robot)
        except Exception as e:
            return {"error": str(e), "fsm_state": self.fsm_state}

    def _build_telemetry(self, robot):
        t = {
            "timestamp_s": round(time.time(), 6),
            "sim_step":    self._sim_steps,
            "sim_time_s":  round(self._sim_steps * self.DT, 6),
            "fsm_state":   self.fsm_state,
        }
        t["joints"]       = robot.get_joint_telemetry()
        t["end_effector"] = robot.get_ee_telemetry()
        t["contacts"]     = robot.get_contact_telemetry()
        cs = t["contacts"] or []
        t["contact_summary"] = {
            "num_contact_points":     len(cs),
            "total_normal_force_N":   round(sum(c["normal_force_N"] for c in cs), 4),
            "total_friction_force_N": round(
                sum(abs(c["lateral_friction1_N"]) + abs(c["lateral_friction2_N"])
                    for c in cs), 4),
        }
        t["grasp_constraint"] = robot.get_grasp_constraint_telemetry()
        t["dynamics"]         = robot.get_dynamics_info()

        pid = self.current_peg_id
        if pid is not None:
            try:
                pp, po = p.getBasePositionAndOrientation(pid)
                pe     = p.getEulerFromQuaternion(po)
                plv, pav = p.getBaseVelocity(pid)
                t["peg"] = {
                    "body_id": pid, "shape": self.current_peg_shape,
                    "position": {"x": round(pp[0],6),"y": round(pp[1],6),"z": round(pp[2],6)},
                    "orientation_euler_deg": {
                        "roll":  round(math.degrees(pe[0]),4),
                        "pitch": round(math.degrees(pe[1]),4),
                        "yaw":   round(math.degrees(pe[2]),4),
                    },
                    "linear_velocity":  {"vx":round(plv[0],6),"vy":round(plv[1],6),"vz":round(plv[2],6)},
                    "angular_velocity": {"wx":round(pav[0],6),"wy":round(pav[1],6),"wz":round(pav[2],6)},
                }
            except Exception:
                t["peg"] = None
        else:
            t["peg"] = None

        ts = getattr(self, '_target_surf_ref', None)
        if ts:
            sd = {
                "type": ts.get("type"), "angle_deg": ts.get("angle", 0),
                "shape": ts.get("shape"),
                "hole_position": [round(v,6) for v in ts["hole_pos"]] if "hole_pos" in ts else None,
            }
            if ts.get("type") == "sculptured":
                crv = ts.get("curvature", {})
                sd["surface_type"] = ts.get("surface_type")
                sd["differential_geometry"] = {
                    "gaussian_curvature_K":    round(crv.get("K", 0), 8),
                    "mean_curvature_H":        round(crv.get("H", 0), 8),
                    "principal_curvature_k1":  round(crv.get("k1", 0), 8),
                    "principal_curvature_k2":  round(crv.get("k2", 0), 8),
                    "surface_class":           _classify_surface(crv.get("K",0), crv.get("H",0)),
                }
                stab = ts.get("conjugate_profile", {})
                sd["conjugate_analysis"] = {
                    "contact_type":     stab.get("contact_type"),
                    "closure_type":     stab.get("closure"),
                    "stability_metric": round(stab.get("stability", 0), 4),
                    "description":      stab.get("desc"),
                }
            else:
                n, ad = self._get_normal(ts)
                sd["insertion_normal"] = [round(float(v),6) for v in n]
                c = ts.get("conjugate_profile", {})
                sd["conjugate_analysis"] = {
                    "contact_type":     c.get("contact_type"),
                    "closure_type":     c.get("closure"),
                    "stability_metric": round(c.get("stability", 0), 4),
                    "description":      c.get("desc"),
                }
            t["target_surface"] = sd
        else:
            t["target_surface"] = None

        ci = robot.get_conjugate_info()
        if ci:
            t["conjugate_grasp"] = {
                "shape":             ci.get("shape"),
                "grip_width_m":      round(ci.get("width", 0), 6),
                "grip_force_N":      round(ci.get("force", 0), 4),
                "contact_type":      ci.get("type"),
                "surface_angle_deg": ci.get("angle", 0),
            }
            if ci.get("K") is not None:
                t["conjugate_grasp"]["curvature_K"]  = round(ci["K"], 8)
                t["conjugate_grasp"]["curvature_H"]  = round(ci.get("H", 0), 8)
                t["conjugate_grasp"]["surface_type"] = ci.get("surface_type")
        else:
            t["conjugate_grasp"] = None

        if ts and "hole_pos" in ts:
            ep, _ = robot.get_ee_pose()
            err   = ep - np.array(ts["hole_pos"])
            t["position_error"] = {
                "dx_m": round(float(err[0]),6), "dy_m": round(float(err[1]),6),
                "dz_m": round(float(err[2]),6),
                "euclidean_m": round(float(np.linalg.norm(err)),6),
            }
        else:
            t["position_error"] = None

        try:
            ph = p.getPhysicsEngineParameters()
            gv = [ph.get("gravityAccelerationX",0),
                  ph.get("gravityAccelerationY",0),
                  ph.get("gravityAccelerationZ",-9.81)]
            si = ph.get("numSolverIterations", 50)
        except Exception:
            gv, si = [0, 0, -9.81], 50
        t["physics"] = {"gravity_m_s2": gv, "timestep_s": self.DT, "solver_iterations": si}

        t["vision"]         = dict(self._vision_results) if self._vision_results else None
        t["grasp_quality"]  = self._grasp_quality
        t["slip_detection"] = self._slip_data
        t["wrist_ft"]       = robot.get_wrist_forcetorque()
        t["task_progress"]  = dict(self._task_stats)
        t["task_progress"]["queue_remaining"] = len(self._color_sort_queue)
        t["force_history"]  = list(self._force_history[-20:])
        return t

    # ─────────────────────────────────────────────────────────────────
    def _get_focused_cam(self, key):
        focus_label = self._focus_surface_label()
        cams = (SURFACE_CAMERAS.get(focus_label, DEFAULT_CAMERAS)
                if focus_label else DEFAULT_CAMERAS)
        cp, ct, up = cams.get(key, DEFAULT_CAMERAS[key])
        ox, oy, oz = self._world_offset.tolist()
        cp = [cp[0] + ox, cp[1] + oy, cp[2] + oz]
        ct = [ct[0] + ox, ct[1] + oy, ct[2] + oz]
        return cp, ct, up

    def _get_scene_cam(self, w=640, h=480, mode=None):
        m = mode or self.camera_mode
        if   m == "SIDE":     cp,ct,up = [0.6,-2.4,2.0],  [0.6,0.0,0.9],   [0,0,1]
        elif m == "LEFT":     cp,ct,up = [-0.8,0.45,2.0],  [0.55,0.45,0.78],[0,0,1]
        elif m == "CONVEYOR": cp,ct,up = [0.55,2.18,2.10], [0.55,0.43,0.80],[0,0,1]
        elif m == "CS_FRONT": cp,ct,up = [2.88,0.62,1.94], [0.55,0.40,0.80],[0,0,1]
        elif m == "CS_TOP":   cp,ct,up = [0.55,0.42,3.34], [0.55,0.42,0.80],[0,1,0]
        elif m in ("TOP","FRONT","RIGHT","FOCUS","CENTER"):
            k = "FOCUS" if m == "CENTER" else m
            cp, ct, up = self._get_focused_cam(k)
        else:
            return None
        if m not in ("TOP","FRONT","RIGHT","FOCUS","CENTER"):
            ox, oy, oz = self._world_offset.tolist()
            cp = [cp[0] + ox, cp[1] + oy, cp[2] + oz]
            ct = [ct[0] + ox, ct[1] + oy, ct[2] + oz]
        try:
            vm  = p.computeViewMatrix(cp, ct, up)
            fov = 60.0
            focus_label = self._focus_surface_label()
            if focus_label and m in ("TOP", "FRONT", "RIGHT", "FOCUS", "CENTER"):
                fov = SURFACE_FOVS.get(focus_label, 40.0)
            if m in ("CONVEYOR", "CS_FRONT", "CS_TOP"):
                fov = 42.0
            pm  = p.computeProjectionMatrixFOV(fov, w / h, 0.03, 20.0)
            res = p.getCameraImage(w, h, vm, pm, renderer=p.ER_TINY_RENDERER)
            return np.array(res[2], dtype=np.uint8).reshape(res[1], res[0], 4)[:, :, :3]
        except Exception:
            return None

    def get_color_sort_frame(self):
        with self.lock:
            fr = self._color_sort_frame
            st = self.fsm_state
            ac = self._color_sort_mode
            ca = self._color_sort_placeholder_jpg
            ct = self._color_sort_placeholder_text
        if fr:
            return fr
        text = ("Waiting...\nSTATE: " + st if ac else
                ("Color sort cannot start.\nSTATE: " + st if st != "IDLE"
                 else "Press START SORT\nSTATE: IDLE"))
        if ca and ct == text:
            return ca
        ph = self._make_placeholder_jpg(text, 2560, 720)
        with self.lock:
            self._color_sort_placeholder_text = text
            self._color_sort_placeholder_jpg  = ph
        return ph

    # ─────────────────────────────────────────────────────────────────
    def _snap_attach(self, robot, peg_id, hold_force=None):
        ee_pos, ee_orn = robot.get_ee_pose()
        try:
            peg_center = self._expected_peg_center_from_robot(robot, ee_orn)
            p.resetBasePositionAndOrientation(peg_id, peg_center.tolist(), ee_orn)
            p.resetBaseVelocity(peg_id, [0,0,0], [0,0,0])
            p.changeDynamics(
                peg_id,
                -1,
                lateralFriction=8.0,
                spinningFriction=2.5,
                rollingFriction=1.4,
                linearDamping=0.999,
                angularDamping=0.999,
                contactStiffness=42000,
                contactDamping=3200,
            )
        except Exception:
            pass
        robot.attach(peg_id, max_force=hold_force or self._hold_force_for_angle())

    def _return_peg(self, robot, peg_id, origin, origin_orn=None):
        robot.detach()
        robot.open_gripper()
        try:
            orn = origin_orn if origin_orn is not None else [0, 0, 0, 1]
            p.resetBasePositionAndOrientation(peg_id, list(origin), list(orn))
            p.resetBaseVelocity(peg_id, [0,0,0], [0,0,0])
        except Exception:
            pass

    def _perform_reset(self, robot, env, z_table, is_e_stop=False):
        self.camera_mode       = "TOP"
        self.fsm_state         = "IDLE"
        self._target_surf_ref  = None
        self.current_peg_shape = None
        self.current_peg_id    = None
        self._priority_command = None
        robot.detach()
        robot.reset_to_home()
        env.show_all_surfaces()
        self._focused_surface = None

        self._color_sort_queue      = []
        self._color_sort_mode       = False
        self._sort_target_zone      = None
        self._force_history         = []
        self._vision_results        = {}
        self._grasp_quality         = None
        self._slip_data             = None
        self._insertion_force_peak  = 0.0
        self._hole_vision_pos       = None
        self._task_stats            = {"completed": 0, "failed": 0, "total": 0}
        self._recovery_resume_state = None
        self._recovery_peg_pos      = None
        self._recovery_attempts     = 0
        self._grasp_retry_attempts       = 0
        self._sort_grasp_retry_attempts  = 0
        self._insert_retry_attempts      = 0
        self._sort_insert_retry_attempts = 0
        self._last_busy_warn_ts     = 0.0
        self._pick_origin_pos       = None
        self._pick_origin_orn       = None
        self._sort_pick_origin_pos  = None
        self._sort_pick_origin_orn  = None
        self._last_insert_hole_world = None
        self._last_insert_normal     = None
        self._last_spoken_text       = ""
        self._completed_task_snapshots = []
        self._sim_data_log            = []
        self._color_sort_placeholder_jpg  = None
        self._color_sort_placeholder_text = ""
        self._last_state_logged       = None
        self.peg_tip_offset          = 0.06
        self._set_robot_peg_collisions(robot, None)
        self._set_robot_surface_collisions(robot, True)
        self._set_robot_conveyor_collisions(robot, True)
        env.show_non_conveyor_bodies()
        self._color_sort_focus = False
        self._active_task_label = None
        self._active_task_mode  = "idle"

        all_ids = set(pid for pid, _ in self.available_pegs)
        all_ids |= set(pid for pid, _ in self.used_pegs)
        if self.current_peg_id:
            all_ids.add(self.current_peg_id)
        for pid in all_ids:
            try:
                p.removeBody(pid)
            except Exception:
                pass
        self.current_peg_id      = None
        self.used_pegs           = []
        self._sort_current_color = None
        self._sort_pick_pos      = None

        if self._conveyor_data:
            for cpid in self._conveyor_data.get("peg_ids", []):
                try:
                    p.removeBody(cpid)
                except Exception:
                    pass

        peg_ids, peg_shapes, slot_positions = env.create_peg_magazine(
            self.STATIONS["MAGAZINE"])
        self.available_pegs     = list(zip(peg_ids, peg_shapes))
        self.peg_slot_positions = slot_positions

        new_conv = env.create_conveyor_belt(self.STATIONS["CONVEYOR"], z_table)
        self._conveyor_data = new_conv
        new_cpids = new_conv.get("peg_ids", [])
        self._all_peg_ids = list(peg_ids) + list(new_cpids)
        for cpid in new_cpids:
            if cpid in env._surface_body_ids:
                env._surface_body_ids.remove(cpid)
            if cpid in env._magazine_body_ids:
                env._magazine_body_ids.remove(cpid)
        for cpid in new_cpids:
            for sbid in env._surface_body_ids + env._magazine_body_ids:
                p.setCollisionFilterPair(cpid, sbid, -1, -1, enableCollision=0)

        if is_e_stop:
            self._speak("Emergency stop! All motion halted.", "high")
            self._sound("alert")
        else:
            self._speak("System reset complete. Ready.", "normal")
            self._sound("startup")
        print("RESET COMPLETE", flush=True)

    def _capture_frame_jpg(self):
        """Capture current scene as JPEG bytes for snapshot."""
        try:
            img = self._get_scene_cam(640, 480, "FOCUS")
            if img is None:
                return None
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            ret, enc = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return enc.tobytes() if ret else None
        except Exception:
            return None

    # ═════════════════════════════════════════════════════════════════
    # MAIN SIMULATION LOOP
    # ═════════════════════════════════════════════════════════════════
    def _run_loop(self):
        import os, base64
        want_gui = self._enable_gui
        if want_gui is None:
            want_gui = os.environ.get("SIM_GUI", "").strip().lower() in ("1", "true", "yes", "on")
        connect_mode = p.GUI if want_gui else p.DIRECT
        cid = -1
        try:
            cid = p.connect(connect_mode)
        except Exception:
            if want_gui:
                print("PyBullet GUI connect failed, falling back to DIRECT mode.", flush=True)
                cid = p.connect(p.DIRECT)
            else:
                raise
        if cid < 0 and want_gui:
            print("PyBullet GUI unavailable, falling back to DIRECT mode.", flush=True)
            cid = p.connect(p.DIRECT)
        self.cid = cid
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.DT)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.DT,
            numSubSteps=10,
            numSolverIterations=250,
            useSplitImpulse=1,
            splitImpulsePenetrationThreshold=-0.01,
            erp=0.8,
            contactERP=0.8,
            frictionERP=0.2,
        )

        env = Environment(cid)
        env.setup_scene()
        z_table = env.create_workbench(self.WB_POS, width=2.6, depth=1.2, height=self.TABLE_H)

        print("Creating Surfaces...", flush=True)
        env.create_inclined_surface(self.STATIONS["NORMAL"],  0,  "Normal Surface", [0.2,0.8,0.2], shape="square")
        env.create_inclined_surface(self.STATIONS["INC_15"], 15,  "15 DEG",         [0.8,0.8,0.2], shape="cylinder")
        env.create_inclined_surface(self.STATIONS["INC_30"], 30,  "30 DEG",         [0.2,0.2,0.8], shape="cylinder")
        env.create_inclined_surface(self.STATIONS["INC_45"], 45,  "45 DEG",         [0.8,0.2,0.8], shape="cylinder")
        print("Inclined Surfaces Created", flush=True)
        env.create_sculptured_surface(self.STATIONS["DOME"], "DOME", "dome",
                                      [0.0,0.9,0.6], peg_shape="triangle")
        print("Dome Surface Created", flush=True)

        conveyor = env.create_conveyor_belt(self.STATIONS["CONVEYOR"], z_table)
        self._conveyor_data = conveyor
        print("Conveyor Belt Created", flush=True)

        conv_peg_ids = conveyor.get("peg_ids", [])
        for cpid in conv_peg_ids:
            if cpid in env._surface_body_ids:  env._surface_body_ids.remove(cpid)
            if cpid in env._magazine_body_ids: env._magazine_body_ids.remove(cpid)

        peg_ids, peg_shapes, slot_positions = env.create_peg_magazine(self.STATIONS["MAGAZINE"])
        self.available_pegs     = list(zip(peg_ids, peg_shapes))
        self.peg_slot_positions = slot_positions
        self.used_pegs          = []
        self.current_peg_id     = None
        self._all_peg_ids       = list(peg_ids) + list(conv_peg_ids)
        self._env_ref           = env

        # Recovery funnel near magazine for controlled drop handling
        rx, ry, _ = self.STATIONS["MAGAZINE"]
        rec_pos = [rx - 0.12, ry + 0.18, z_table + 0.02]
        self._recovery_station = env.create_recovery_station(rec_pos)

        for cpid in conv_peg_ids:
            for sbid in env._surface_body_ids + env._magazine_body_ids:
                p.setCollisionFilterPair(cpid, sbid, -1, -1, enableCollision=0)

        robot = RobotControl(cid)
        base_pos = [0.0, 0.0, z_table]
        if self._robot_base is not None:
            try:
                base_pos[0] = float(self._robot_base[0]) + self._world_offset[0]
                base_pos[1] = float(self._robot_base[1]) + self._world_offset[1]
                base_pos[2] = float(self._robot_base[2])
            except Exception:
                pass
        base_orn = p.getQuaternionFromEuler([0, 0, self._robot_yaw])
        robot.load_robot(base_pos=base_pos, base_orn=base_orn, color_rgba=self._robot_color)
        robot.set_curvilinear_mode(True)
        self._robot_ref = robot
        vision = VisionSystem()

        self.fsm_state = "IDLE"
        sub_step     = 0
        target_surf  = None
        self._target_surf_ref = None
        hole_world   = None
        peg_home_pos = None
        grasp_orn    = p.getQuaternionFromEuler([math.pi, 0, 0])
        pick_orn     = grasp_orn

        sim_steps = 0
        self._sim_steps = 0

        while self.running and p.isConnected():
            try:
                p.stepSimulation()
                time.sleep(self.DT * self.REALTIME_SCALE)
                sim_steps      += 1
                self._sim_steps = sim_steps
                if self.fsm_state != self._last_state_logged:
                    self._log("state_change", {"state": self.fsm_state})
                    self._last_state_logged = self.fsm_state

                if sim_steps % 240 == 0:
                    print(f"Step {sim_steps} | State: {self.fsm_state}", flush=True)
                if self.fsm_state == "IDLE" and sim_steps % 45 == 0:
                    robot.hold_home_pose()
                priority_cmd = self._priority_command
                if priority_cmd in ("RESET", "E_STOP"):
                    self._priority_command = None
                    self._perform_reset(robot, env, z_table,
                                        is_e_stop=(priority_cmd == "E_STOP"))
                    sub_step = 0
                    target_surf = None
                    hole_world = None
                    peg_home_pos = None
                    grasp_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
                    pick_orn = grasp_orn
                    continue

                # ── Camera capture 15 Hz ──────────────────────────────
                if sim_steps % 16 == 0:
                    try:
                        ws, hs = (720, 540) if self._render_profile == "compact" else (853, 720)
                        focus_label = self._focus_surface_label()
                        preview_color_sort = (
                            self.fsm_state == "IDLE" and
                            self._preview_task_mode == "color_sort"
                        )
                        if preview_color_sort:
                            env.show_all_surfaces()
                            env.hide_non_conveyor_bodies(
                                extra_body_ids=list(self.peg_slot_positions.keys()))
                        elif not focus_label and not self._color_sort_focus:
                            env.show_all_surfaces()
                            env.show_non_conveyor_bodies()
                        if focus_label:
                            env.show_only_surface(
                                focus_label,
                                extra_body_ids=self._get_surface_focus_extra_body_ids())

                        img_t  = self._get_scene_cam(ws, hs, "TOP")
                        img_f  = self._get_scene_cam(ws, hs, "FRONT")
                        img_r  = self._get_scene_cam(ws, hs, "RIGHT")
                        img_fc = self._get_scene_cam(
                            ws, hs, "CS_FRONT" if preview_color_sort else "FOCUS")

                        def proc(img):
                            if img is None or not isinstance(img, np.ndarray):
                                return np.zeros((hs, ws, 3), dtype=np.uint8)
                            if img.shape[0] != hs or img.shape[1] != ws:
                                img = cv2.resize(img, (ws, hs))
                            if focus_label and self._render_profile != "compact":
                                ih, iw = img.shape[:2]
                                crop = SURFACE_VIEW_CROPS.get(
                                    focus_label, (0.06, 0.94, 0.06, 0.94))
                                x0, x1 = int(iw * crop[0]), int(iw * crop[1])
                                y0, y1 = int(ih * crop[2]), int(ih * crop[3])
                                if x1 > x0 and y1 > y0:
                                    img = cv2.resize(img[y0:y1, x0:x1], (ws, hs))
                            return img.copy()

                        v_t, v_f, v_r, v_fc = proc(img_t), proc(img_f), proc(img_r), proc(img_fc)

                        def lbl(v, s):
                            ov = v.copy()
                            cv2.rectangle(ov, (0,0), (len(s)*17+30, 40), (0,0,0), -1)
                            cv2.addWeighted(ov, 0.55, v, 0.45, 0, v)
                            cv2.putText(v, s, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

                        fs = f" [{focus_label}]" if focus_label else (" [COLOR SORT]" if preview_color_sort else "")
                        lbl(v_t,  f"Top View{fs}")
                        lbl(v_f,  f"Front View{fs}")
                        lbl(v_r,  f"Right Side View{fs}")
                        lbl(v_fc, f"Focus View{fs}")

                        pw = ws * 2
                        fi = np.zeros((hs, pw, 3), dtype=np.uint8)
                        cv2.rectangle(fi, (0,0), (pw, hs), (15,15,20), -1)

                        ht = (f"SURFACE FOCUS: {focus_label.upper()}"
                              if focus_label else "ADVANCED GRASP SYSTEM")
                        cv2.putText(fi, ht, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0,212,255), 2)
                        cv2.line(fi, (30,65), (pw-30,65), (0,212,255), 2)

                        if focus_label:
                            bc = (0,255,150)
                            cv2.rectangle(fi, (pw-400,12), (pw-12,58), bc, 2)
                            cv2.putText(fi, "SURFACE ISOLATED", (pw-388,44),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, bc, 2)

                        tp   = self._task_stats
                        pp2  = tp["completed"] / max(1, tp["total"]) if tp["total"] > 0 else 0
                        cv2.putText(fi,
                            f"TASKS:  {tp['completed']} Done   {tp['failed']} Failed   {tp['total']} Total",
                            (30,100), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (200,200,200), 2)
                        cv2.rectangle(fi, (30,112), (30+pw-60, 126), (50,50,50), -1)
                        cv2.rectangle(fi, (30,112), (30+int((pw-60)*pp2), 126), (0,255,100), -1)

                        sc = (0,255,100) if self.fsm_state == "IDLE" else (0,200,255)
                        cv2.putText(fi, f"STATE:  {self.fsm_state}", (30,160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.00, sc, 2)

                        gq = self._grasp_quality
                        if gq and gq.get("quality_score", 0) > 0:
                            qs = gq["quality_score"]
                            qc = (0,255,0) if qs>0.6 else ((0,200,255) if qs>0.3 else (0,80,255))
                            cv2.putText(fi, f"GRASP QUALITY:  {qs:.0%}", (30,210),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, qc, 2)
                            cv2.rectangle(fi, (30,222), (30+480, 240), (40,40,40), -1)
                            cv2.rectangle(fi, (30,222), (30+int(480*qs), 240), qc, -1)
                            cv2.putText(fi,
                                f"Contacts: {gq['num_contacts']}   "
                                f"Bilateral: {'YES' if gq['has_bilateral'] else 'NO'}   "
                                f"Balance: {gq['force_balance']:.0%}",
                                (30,258), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (180,180,180), 2)

                        if self._slip_data and self._slip_data.get("drift_mm",0) > 0:
                            slip = self._slip_data
                            slc  = (0,80,255) if slip["slipped"] else (0,200,100)
                            cv2.putText(fi,
                                f"SLIP:  {'DETECTED!' if slip['slipped'] else 'OK'}   "
                                f"drift={slip['drift_mm']:.1f}mm",
                                (30,292), cv2.FONT_HERSHEY_SIMPLEX, 0.70, slc, 2)

                        ci2  = robot.get_conjugate_info()
                        yc   = 330
                        cv2.line(fi, (30,yc-18), (pw-30,yc-18), (0,100,150), 2)
                        if ci2:
                            cv2.putText(fi, "CONJUGATE GRASP  ACTIVE", (30,yc),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0,255,0), 2)
                            cv2.putText(fi,
                                f"Shape: {ci2['shape'].upper()}   Contact: {ci2['type']}",
                                (40,yc+36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (200,200,200), 2)
                            cv2.putText(fi,
                                f"Width: {ci2['width']*1000:.1f}mm    "
                                f"Force: {ci2['force']:.0f}N    Angle: {ci2['angle']}deg",
                                (40,yc+70), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (200,200,200), 2)
                            fp2 = min(1.0, ci2['force'] / 1200.0)
                            cv2.rectangle(fi, (40,yc+84), (40+500, yc+106), (50,50,50), -1)
                            gc  = (0,255,0) if fp2<0.5 else ((0,200,255) if fp2<0.8 else (0,80,255))
                            cv2.rectangle(fi, (40,yc+84), (40+int(500*fp2), yc+106), gc, -1)
                            cv2.circle(fi, (50,yc+135), 10, (0,255,0), -1)
                            cv2.putText(fi, "FORM CLOSURE ACHIEVED", (72,yc+142),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0,255,0), 2)
                            if ci2.get("K") is not None:
                                cv2.line(fi,(30,yc+162),(pw-30,yc+162),(0,150,200),1)
                                cv2.putText(fi,"CURVATURE ANALYSIS",(30,yc+192),
                                            cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,212,255),2)
                                cv2.putText(fi,
                                    f"K (Gaussian): {ci2['K']:.5f}     H (Mean): {ci2.get('H',0):.5f}",
                                    (40,yc+226), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (200,200,200), 2)
                                st2    = ci2.get('surface_type','').upper()
                                scolor = (0,255,150) if ci2['K']>0 else (255,100,100)
                                cv2.putText(fi, f"Surface Class:  {st2}", (40,yc+260),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.80, scolor, 2)
                        elif self.current_peg_shape:
                            cv2.putText(fi, "CONJUGATE GRASP  PREPARING...", (30,yc),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0,200,255), 2)
                        else:
                            cv2.putText(fi, "AWAITING COMMAND", (30,yc),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (100,100,100), 2)

                        ft = robot.get_wrist_forcetorque()
                        if ft and self.fsm_state != "IDLE":
                            yft = hs - 105
                            cv2.line(fi, (30,yft-18), (pw-30,yft-18), (80,80,100), 1)
                            cv2.putText(fi, "WRIST  F/T  SENSOR", (30,yft),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,212,255), 2)
                            cv2.putText(fi,
                                f"Force: {ft['force_magnitude']:.1f} N       "
                                f"Torque: {ft['torque_magnitude']:.4f} Nm",
                                (30,yft+36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200,200,200), 2)
                            fbar_pct = min(1.0, ft['force_magnitude'] / 300.0)
                            cv2.rectangle(fi, (30,yft+50), (30+500, yft+65), (40,40,40), -1)
                            fc2 = (0,255,0) if fbar_pct<0.4 else ((0,200,255) if fbar_pct<0.7 else (0,80,255))
                            cv2.rectangle(fi, (30,yft+50), (30+int(500*fbar_pct), yft+65), fc2, -1)

                        top_row = np.hstack([v_t, v_f, v_r])
                        bot_row = np.hstack([v_fc, fi])
                        if bot_row.shape[1] < top_row.shape[1]:
                            pad = np.zeros((hs, top_row.shape[1]-bot_row.shape[1], 3), dtype=np.uint8)
                            bot_row = np.hstack([bot_row, pad])
                        dashboard     = np.vstack([top_row, bot_row])
                        dashboard_bgr = cv2.cvtColor(dashboard, cv2.COLOR_RGB2BGR)
                        ret2, enc2    = cv2.imencode('.jpg', dashboard_bgr,
                                                     [cv2.IMWRITE_JPEG_QUALITY, 85])
                        single_focus = v_fc.copy()
                        if focus_label and self._render_profile != "compact":
                            sh, sw = single_focus.shape[:2]
                            cx0 = int(sw * 0.03)
                            cx1 = int(sw * 0.97)
                            cy0 = int(sh * 0.03)
                            cy1 = int(sh * 0.97)
                            if cx1 > cx0 and cy1 > cy0:
                                single_focus = cv2.resize(single_focus[cy0:cy1, cx0:cx1], (ws, hs))
                        cv2.rectangle(single_focus, (0, 0), (single_focus.shape[1], 52), (5, 10, 18), -1)
                        cv2.putText(
                            single_focus,
                            "Double Workbench Color Sort View" if preview_color_sort else "Double Workbench Focus View",
                            (18, 34),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 212, 255), 2)
                        active_label = (
                            self._active_task_label or
                            self._pending_command_label or
                            self._preview_task_label or
                            self.fsm_state
                        )
                        cv2.putText(single_focus, f"Task: {active_label}", (18, 64),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (218, 239, 255), 2)
                        cv2.putText(single_focus, f"State: {self.fsm_state}", (18, single_focus.shape[0] - 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                                    (0, 255, 136) if self.fsm_state == "IDLE" else (255, 255, 255), 2)
                        single_bgr = cv2.cvtColor(single_focus, cv2.COLOR_RGB2BGR)
                        single_ok, single_enc = cv2.imencode('.jpg', single_bgr,
                                                             [cv2.IMWRITE_JPEG_QUALITY, 88])
                        if ret2:
                            with self.lock:
                                self.current_frame = enc2.tobytes()
                                if single_ok:
                                    self.single_view_frame = single_enc.tobytes()

                        # ── Color Sort 3-view feed ────────────────────
                        if self._conveyor_data and self._render_profile != "compact":
                            cs_front = cs_top = cs_conv = None
                            try:
                                mag_peg_ids    = [
                                    pid for pid in self.peg_slot_positions.keys()
                                    if pid != self.current_peg_id
                                ]
                                _fb            = self._focused_surface
                                self._focused_surface = None
                                env.show_all_surfaces()
                                env.hide_non_conveyor_bodies(extra_body_ids=mag_peg_ids)
                                for cpid2 in self._conveyor_data.get("peg_ids", []):
                                    try:
                                        for si in p.getVisualShapeData(cpid2):
                                            if list(si[7])[3] < 0.1:
                                                rgba = list(si[7])
                                                p.changeVisualShape(cpid2, si[1],
                                                    rgbaColor=[rgba[0],rgba[1],rgba[2],1.0])
                                    except Exception:
                                        pass
                                cs_front = self._get_scene_cam(ws, hs, "CS_FRONT")
                                cs_top   = self._get_scene_cam(ws, hs, "CS_TOP")
                                cs_conv  = self._get_scene_cam(ws, hs, "CONVEYOR")
                            finally:
                                if self._color_sort_focus:
                                    env.hide_non_conveyor_bodies(extra_body_ids=mag_peg_ids)
                                else:
                                    env.show_non_conveyor_bodies()
                                self._focused_surface = _fb
                                if _fb:
                                    env.show_only_surface(
                                        _fb,
                                        extra_body_ids=self._get_surface_focus_extra_body_ids())

                            def csp(img):
                                if img is None or not isinstance(img, np.ndarray):
                                    return np.zeros((hs, ws, 3), dtype=np.uint8)
                                if img.shape[0] != hs or img.shape[1] != ws:
                                    img = cv2.resize(img, (ws, hs))
                                return img.copy()

                            csf = csp(cs_front)
                            cst = csp(cs_top)
                            csc = csp(cs_conv)

                            def clbl(v, s):
                                ov = v.copy()
                                cv2.rectangle(ov, (0,0), (len(s)*17+30,40), (0,0,0), -1)
                                cv2.addWeighted(ov, 0.55, v, 0.45, 0, v)
                                cv2.putText(v, s, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

                            if self._color_sort_mode:
                                cn  = self._sort_current_color or "?"
                                rem = len(self._color_sort_queue)
                                cmp = self._task_stats["completed"]
                                fl  = f"Front  |  {cn.upper()} peg  |  {cmp} done  |  {rem} left"
                            else:
                                fl = "Front  |  PREVIEW  |  Press START SORT"

                            clbl(csf, fl)
                            clbl(cst, "Top View - Color Sort")
                            clbl(csc, "Conveyor Focus View")
                            cv2.putText(csf, f"STATE: {self.fsm_state}",
                                        (10, hs-30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,150), 2)

                            cs_dash    = np.hstack([csf, cst, csc])
                            cs_bgr     = cv2.cvtColor(cs_dash, cv2.COLOR_RGB2BGR)
                            cs_ret, cs_enc = cv2.imencode('.jpg', cs_bgr,
                                                          [cv2.IMWRITE_JPEG_QUALITY, 85])
                            if cs_ret:
                                with self.lock:
                                    self._color_sort_frame = cs_enc.tobytes()
                                    if self._color_sort_focus:
                                        single_sort = csf.copy()
                                        cv2.rectangle(single_sort, (0, 0), (single_sort.shape[1], 52), (5, 10, 18), -1)
                                        cv2.putText(single_sort, "Double Workbench Color Sort View", (18, 34),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 212, 255), 2)
                                        cv2.putText(single_sort, f"Task: {self._sort_current_color or 'COLOR_SORT'}", (18, 64),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (218, 239, 255), 2)
                                        cv2.putText(single_sort, f"State: {self.fsm_state}", (18, single_sort.shape[0] - 18),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                                                    (0, 255, 136) if self.fsm_state == "IDLE" else (255, 255, 255), 2)
                                        single_sort_bgr = cv2.cvtColor(single_sort, cv2.COLOR_RGB2BGR)
                                        single_sort_ok, single_sort_enc = cv2.imencode('.jpg', single_sort_bgr,
                                                                                        [cv2.IMWRITE_JPEG_QUALITY, 90])
                                        if single_sort_ok:
                                            self.single_view_frame = single_sort_enc.tobytes()

                    except Exception as e:
                        print(f"DASHBOARD ERROR: {e}", flush=True)
                        import traceback; traceback.print_exc()

                # ── Command handler ───────────────────────────────────
                if not self.command_queue.empty():
                    cmd = self.command_queue.get_nowait()
                    self._pending_command_label = None
                    self._pending_command_mode = "idle"
                    print(f"CMD: {cmd}", flush=True)

                    if cmd.startswith("CAM_"):
                        m2 = {"CAM_TOP":"TOP","CAM_SIDE":"SIDE",
                              "CAM_FRONT":"FRONT","CAM_ORBIT":"ORBIT"}
                        self.camera_mode = m2.get(cmd, self.camera_mode)

                    elif cmd in ("RESET", "E_STOP"):
                        self._perform_reset(robot, env, z_table,
                                            is_e_stop=(cmd == "E_STOP"))
                        sub_step = 0
                        target_surf = None
                        hole_world = None
                        peg_home_pos = None
                        grasp_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
                        pick_orn = grasp_orn

                    elif self.fsm_state != "IDLE":
                        self._warn_busy(cmd)

                    elif self.fsm_state == "IDLE":
                        tsurf = None
                        if   cmd == "Normal Surface": tsurf = env.surfaces.get("Normal Surface")
                        elif cmd == "15 DEG":         tsurf = env.surfaces.get("15 DEG")
                        elif cmd == "30 DEG":         tsurf = env.surfaces.get("30 DEG")
                        elif cmd == "45 DEG":         tsurf = env.surfaces.get("45 DEG")
                        elif cmd == "DOME":           tsurf = env.surfaces.get("DOME")

                        if cmd == "COLOR_SORT" and self._conveyor_data:
                            mag_peg_ids = list(self.peg_slot_positions.keys())
                            env.show_all_surfaces()
                            env.hide_non_conveyor_bodies(extra_body_ids=mag_peg_ids)
                            self._focused_surface = None
                            self._color_sort_focus = True
                            conv_data  = self._conveyor_data
                            cp2  = conv_data.get("peg_ids", [])
                            cc   = conv_data.get("peg_colors", {})
                            cpos = conv_data.get("peg_positions", {})
                            csh  = conv_data.get("peg_shapes", {})
                            zns  = conv_data.get("zones", {})
                            if cp2:
                                q = [{"peg_id":   pid,
                                      "color":    cc.get(pid, "unknown"),
                                      "shape":    csh.get(pid, "cylinder"),
                                      "pick_pos": cpos.get(pid)}
                                     for pid in cp2]
                                first = q[0]
                                self._color_sort_queue = q[1:]
                                self._task_stats = {"total": len(q), "completed": 0, "failed": 0}
                                self._color_sort_mode        = True
                                self.current_peg_id          = first["peg_id"]
                                self.current_peg_shape       = first.get("shape", "cylinder")
                                self.peg_tip_offset          = PEG_TIP_OFFSETS.get(self.current_peg_shape, 0.06)
                                self._sort_grasp_retry_attempts = 0
                                self._sort_insert_retry_attempts = 0
                                self._sort_current_color     = first["color"]
                                self._sort_pick_pos          = first["pick_pos"]
                                self._sort_pick_origin_pos   = (list(first["pick_pos"])
                                                                if first.get("pick_pos") else None)
                                try:
                                    _, po = p.getBasePositionAndOrientation(self.current_peg_id, self.cid)
                                    self._sort_pick_origin_orn = list(po)
                                except Exception:
                                    self._sort_pick_origin_orn = [0, 0, 0, 1]
                                self._sort_target_zone       = zns.get(first["color"])
                                self._target_surf_ref        = None
                                target_surf                  = None
                                self._active_task_label      = "COLOR_SORT"
                                self._active_task_mode       = "color_sort"
                                self._set_robot_peg_collisions(robot, self.current_peg_id)
                                self._set_robot_surface_collisions(robot, True)
                                self._set_robot_conveyor_collisions(robot, False)
                                self._speak(
                                    f"Starting color sort. {len(q)} pegs. "
                                    f"First: {first['color']} {first['shape']}.", "high")
                                self._sound("startup")
                                self.fsm_state, sub_step = "SORT_PICK_APPROACH", 0
                                tsurf = None

                        if tsurf and self.available_pegs:
                            sl     = tsurf.get("label", cmd)
                            needed = SURFACE_SHAPE_MAP.get(sl, tsurf.get("shape", "cylinder"))
                            if self._color_sort_focus:
                                env.show_non_conveyor_bodies()
                                self._color_sort_focus = False
                            self._focused_surface = sl
                            env.show_only_surface(
                                sl,
                                extra_body_ids=self._get_surface_focus_extra_body_ids())
                            # Vision-based hole shape detection (surface-driven)
                            det_shape = self._detect_surface_hole_shape(vision, sl)
                            if det_shape:
                                needed = det_shape
                                self._speak(f"Hole shape detected: {det_shape}.", "normal")
                            matching = [(i, pid, sh)
                                        for i, (pid, sh) in enumerate(self.available_pegs)
                                        if sh == needed]
                            if not matching:
                                matching = [(i, pid, sh)
                                            for i, (pid, sh) in enumerate(self.available_pegs)]
                            ci3, cpid5, csh2 = self._select_preferred_peg(matching)
                            self.available_pegs.pop(ci3)
                            self.current_peg_id    = cpid5
                            self.used_pegs.append((cpid5, csh2))
                            self.current_peg_shape = csh2
                            self.peg_tip_offset    = PEG_TIP_OFFSETS.get(csh2, 0.06)
                            self._grasp_retry_attempts = 0
                            self._insert_retry_attempts = 0
                            env.show_only_surface(
                                sl,
                                extra_body_ids=self._get_surface_focus_extra_body_ids())
                            target_surf            = tsurf
                            self._target_surf_ref  = tsurf
                            self._active_task_label = sl
                            self._active_task_mode = "surface"
                            self._task_stats["total"] += 1
                            self._set_robot_peg_collisions(robot, self.current_peg_id)
                            self._set_robot_surface_collisions(robot, False)
                            self._set_robot_conveyor_collisions(robot, True)
                            self._speak(f"{sl}: using {csh2} peg.", "high")
                            self._sound("startup")
                            self.fsm_state, sub_step = "PICK_APPROACH", 0

                        elif tsurf and not self.available_pegs:
                            self._speak("Peg magazine empty. Please reset.", "high")

                # ══════════════════════════════════════════════════════
                # FSM
                # ══════════════════════════════════════════════════════
                if self.fsm_state != "IDLE":
                    sub_step += 1

                    if self.fsm_state == "PICK_APPROACH":
                        dp = None
                        try:
                            pp5, _ = p.getBasePositionAndOrientation(self.current_peg_id, self.cid)
                            dp = list(pp5)
                        except Exception:
                            dp = self.peg_slot_positions.get(self.current_peg_id)
                        if sub_step == 1:
                            peg_home_pos          = list(dp)
                            self._pick_origin_pos = list(dp)
                            try:
                                _, po = p.getBasePositionAndOrientation(self.current_peg_id, self.cid)
                                self._pick_origin_orn = list(po)
                            except Exception:
                                self._pick_origin_orn = [0, 0, 0, 1]
                            robot.open_gripper()
                            self._speak(f"Picking {self.current_peg_shape} peg.", "normal")
                            self._sound("servo_move")
                        pick_orn = robot.compute_conjugate_orientation(0, self.current_peg_shape)
                        if sub_step <= 20:
                            hover_center = [dp[0], dp[1], z_table + 0.50]
                            hover_target = self._ee_target_for_peg_center(hover_center, pick_orn, robot)
                            robot.move_to(hover_target.tolist(), pick_orn, ik_iterations=500)
                        elif sub_step <= 50:
                            approach_center = [dp[0], dp[1], dp[2] + 0.06]
                            approach_target = self._ee_target_for_peg_center(approach_center, pick_orn, robot)
                            robot.move_to(approach_target.tolist(), pick_orn, ik_iterations=500)
                        else:
                            grasp_target = self._ee_target_for_peg_center(dp, pick_orn, robot)
                            robot.move_to(grasp_target.tolist(), pick_orn, ik_iterations=500)
                        if sub_step > 90:
                            self.fsm_state, sub_step = "PICK_DOWN", 0

                    elif self.fsm_state == "PICK_DOWN":
                        dp       = peg_home_pos
                        pick_orn = robot.compute_conjugate_orientation(0, self.current_peg_shape)
                        grasp_target = self._ee_target_for_peg_center(dp, pick_orn, robot)
                        robot.move_to(grasp_target.tolist(), pick_orn, ik_iterations=500)
                        if sub_step > 35:
                            self.fsm_state, sub_step = "GRASP", 0

                    elif self.fsm_state == "GRASP":
                        if sub_step == 1:
                            try:
                                cur, _ = p.getBasePositionAndOrientation(self.current_peg_id, self.cid)
                                peg_home_pos = list(cur)
                            except Exception:
                                pass
                        dp       = peg_home_pos
                        pick_orn = robot.compute_conjugate_orientation(0, self.current_peg_shape)
                        grasp_target = self._ee_target_for_peg_center(dp, pick_orn, robot)
                        robot.move_to(grasp_target.tolist(), pick_orn, ik_iterations=500)
                        if sub_step == 1:
                            self._speak(f"Grasping {self.current_peg_shape}.", "normal")
                            self._sound("gripper_close")
                            if target_surf and target_surf.get("type") == "sculptured":
                                robot.sculptured_conjugate_close(
                                    self.current_peg_shape,
                                    target_surf.get("curvature", {}),
                                    target_surf.get("surface_type", "dome"))
                            else:
                                tangle = target_surf.get("angle", 0) if target_surf else 0
                                robot.conjugate_close(self.current_peg_shape, tangle)
                        if sub_step == 32:
                            quality = robot.compute_grasp_quality(self.current_peg_id)
                            self._grasp_quality = quality
                        if sub_step >= 42:
                            if not self._ensure_grasp(robot, self.current_peg_id, sub_step, max_dist=0.08, force_after=90):
                                if sub_step > 110 and self._grasp_retry_attempts < self._max_grasp_retries:
                                    self._grasp_retry_attempts += 1
                                    self._sound("alert")
                                    self.fsm_state, sub_step = "REGRASP", 0
                                continue
                            robot.record_grasp_pose(self.current_peg_id)
                            carry_angle = target_surf.get("angle", 0) if target_surf else 0
                            grasp_orn = robot.compute_conjugate_orientation(
                                carry_angle, self.current_peg_shape)
                        if sub_step > 60:
                            self.fsm_state, sub_step = "VERIFY_GRASP", 0

                    elif self.fsm_state == "REGRASP":
                        if self.current_peg_id is None:
                            self.fsm_state, sub_step = "GO_HOME", 0
                        else:
                            try:
                                cur, _ = p.getBasePositionAndOrientation(self.current_peg_id, self.cid)
                                peg_home_pos = list(cur)
                            except Exception:
                                pass
                            if peg_home_pos is None:
                                self.fsm_state, sub_step = "GO_HOME", 0
                            else:
                                dp       = peg_home_pos
                                pick_orn = robot.compute_conjugate_orientation(0, self.current_peg_shape)
                                offs     = [
                                    (0.000, 0.000),
                                    (0.003, 0.000), (-0.003, 0.000),
                                    (0.000, 0.003), (0.000, -0.003),
                                    (0.002, 0.002), (-0.002, 0.002),
                                    (0.002, -0.002), (-0.002, -0.002),
                                ]
                                ox, oy   = offs[min(self._grasp_retry_attempts, len(offs)-1)]
                                if sub_step == 1:
                                    robot.detach(); robot.open_gripper()
                                if sub_step <= 50:
                                    hover_center = [dp[0] + ox, dp[1] + oy, z_table + 0.44]
                                    hover_target = self._ee_target_for_peg_center(hover_center, pick_orn, robot)
                                    robot.move_to(hover_target.tolist(), pick_orn, ik_iterations=500)
                                elif sub_step <= 80:
                                    approach_center = [dp[0] + ox, dp[1] + oy, dp[2] + 0.06]
                                    approach_target = self._ee_target_for_peg_center(approach_center, pick_orn, robot)
                                    robot.move_to(approach_target.tolist(), pick_orn, ik_iterations=500)
                                else:
                                    grasp_center = [dp[0] + ox, dp[1] + oy, dp[2]]
                                    grasp_target = self._ee_target_for_peg_center(grasp_center, pick_orn, robot)
                                    robot.move_to(grasp_target.tolist(), pick_orn, ik_iterations=500)
                                if sub_step == 100:
                                    if target_surf and target_surf.get("type") == "sculptured":
                                        robot.sculptured_conjugate_close(
                                            self.current_peg_shape,
                                            target_surf.get("curvature",{}),
                                            target_surf.get("surface_type","dome"))
                                    else:
                                        tangle = target_surf.get("angle",0) if target_surf else 0
                                        robot.conjugate_close(self.current_peg_shape, tangle)
                                    self._sound("gripper_close")
                                if sub_step >= 130:
                                    quality = robot.compute_grasp_quality(self.current_peg_id)
                                    self._grasp_quality = quality
                                    close_enough = self._is_grasp_close(
                                        robot, self.current_peg_id, max_dist=0.05)
                                    required_quality = self._required_grasp_quality()
                                    bad = ((not quality.get("has_bilateral") or
                                            quality.get("quality_score",0.0) < required_quality)
                                           and not close_enough)
                                    if bad and self._grasp_retry_attempts < self._max_grasp_retries:
                                        self._grasp_retry_attempts += 1
                                        sub_step = 0
                                    else:
                                        if not self._ensure_grasp(robot, self.current_peg_id, sub_step, max_dist=0.08, force_after=150):
                                            if self._grasp_retry_attempts < self._max_grasp_retries:
                                                self._grasp_retry_attempts += 1
                                                sub_step = 0
                                            else:
                                                self.fsm_state, sub_step = "REGRASP", 0
                                        else:
                                            carry_angle = target_surf.get("angle", 0) if target_surf else 0
                                            grasp_orn = robot.compute_conjugate_orientation(
                                                carry_angle, self.current_peg_shape)
                                            robot.record_grasp_pose(self.current_peg_id)
                                            self.fsm_state, sub_step = "VERIFY_GRASP", 0

                    elif self.fsm_state == "VERIFY_GRASP":
                        dp       = peg_home_pos
                        pick_orn = robot.compute_conjugate_orientation(0, self.current_peg_shape)
                        grasp_target = self._ee_target_for_peg_center(dp, pick_orn, robot)
                        robot.move_to(grasp_target.tolist(), pick_orn, ik_iterations=500)
                        if sub_step == 10:
                            quality = robot.compute_grasp_quality(self.current_peg_id)
                            self._grasp_quality = quality
                            close_enough = self._is_grasp_close(
                                robot, self.current_peg_id, max_dist=0.05)
                            required_quality = self._required_grasp_quality()
                            bad = ((not quality.get("has_bilateral") or
                                    quality.get("quality_score",0.0) < required_quality)
                                   and not close_enough)
                            if bad and self._grasp_retry_attempts < self._max_grasp_retries:
                                self._grasp_retry_attempts += 1
                                self._sound("alert")
                                self.fsm_state, sub_step = "REGRASP", 0
                            else:
                                robot.record_grasp_pose(self.current_peg_id)
                                self._speak(f"Grasp OK. Lifting.", "normal")
                                self._sound("scan")
                        if sub_step > 25:
                            self.fsm_state, sub_step = "LIFT", 0

                    elif self.fsm_state == "LIFT":
                        dp = peg_home_pos
                        angle_now = ((target_surf or {}).get("angle", 0) or 0)
                        lift_extra = 0.08 if angle_now < 30 else 0.12
                        lift_height = z_table + 0.50 + min(
                            lift_extra, (angle_now / 45.0) * lift_extra)
                        lift_center = [dp[0], dp[1], lift_height]
                        lift_target = self._ee_target_for_peg_center(lift_center, grasp_orn, robot)
                        robot.move_to(lift_target.tolist(), grasp_orn)
                        if sub_step == 1:
                            self._speak(f"Lifting.", "normal")
                            self._sound("servo_move")
                        if sub_step % 30 == 0 and sub_step >= 30:
                            slip = robot.check_peg_slip(self.current_peg_id, threshold_mm=20.0)
                            self._slip_data = slip
                            fallen = self._check_peg_drop(robot)
                            if fallen:
                                self._recovery_peg_pos      = fallen
                                self._recovery_resume_state = "LIFT"
                                self.fsm_state, sub_step = "PEG_RECOVERY_APPROACH", 0
                                continue
                        if sub_step > 70:
                            self.fsm_state, sub_step = "SCAN_APPROACH", 0

                    elif self.fsm_state == "SCAN_APPROACH":
                        hole_world = np.array(target_surf["hole_pos"])
                        normal, adeg = self._get_normal(target_surf)
                        sorn = robot.compute_conjugate_orientation(adeg, self.current_peg_shape)
                        scan_extra = 0.06 if adeg < 30 else 0.08
                        scan_distance = 0.32 + min(scan_extra, (adeg / 45.0) * scan_extra)
                        robot.move_to((hole_world + normal*scan_distance).tolist(), sorn)
                        if sub_step == 1:
                            self._speak("Moving to scan position.", "normal")
                            self._sound("servo_move")
                        if sub_step % 30 == 0 and sub_step >= 30:
                            fallen = self._check_peg_drop(robot)
                            if fallen:
                                self._recovery_peg_pos      = fallen
                                self._recovery_resume_state = "SCAN_APPROACH"
                                self.fsm_state, sub_step = "PEG_RECOVERY_APPROACH", 0
                                continue
                        if sub_step > 70:
                            self.fsm_state, sub_step = "SCAN", 0

                    elif self.fsm_state == "SCAN":
                        rgb_s, dep_l, seg, vm, pm, raw = robot.get_camera_image()
                        hole_world = np.array(target_surf["hole_pos"])
                        normal, _  = self._get_normal(target_surf)
                        if sub_step == 1:
                            self._speak("Scanning.", "normal")
                            self._sound("scan")
                        hpx, _ = vision.detect_hole(rgb_s)
                        if hpx is not None:
                            u, v = hpx
                            dv   = dep_l[min(v, dep_l.shape[0]-1), min(u, dep_l.shape[1]-1)]
                            if 0.01 < dv < 1.5:
                                hv = vision.pixel_to_world(u, v, dv, vm, pm,
                                                            rgb_s.shape[1], rgb_s.shape[0])
                                hole_delta = float(np.linalg.norm(hv - hole_world))
                                self._hole_vision_pos = hv.tolist()
                                if hole_delta < 0.018:
                                    hole_world = hole_world * 0.75 + hv * 0.25
                        cd = vision.detect_peg_color(rgb_s, seg, self.current_peg_id)
                        if cd["color"]:
                            self._vision_results = cd
                        self.fsm_state, sub_step = "INSERT_APPROACH", 0

                    elif self.fsm_state == "INSERT_APPROACH":
                        normal, adeg = self._get_normal(target_surf)
                        iorn   = robot.compute_conjugate_orientation(adeg, self.current_peg_shape)
                        center_tgt = hole_world + normal*(self.peg_tip_offset + 0.14)
                        ee_tgt = self._ee_target_for_peg_center(center_tgt, iorn, robot)
                        robot.move_to(ee_tgt.tolist(), iorn, ik_iterations=500)
                        if sub_step == 1:
                            self._speak(f"Approaching at {adeg:.0f} degrees.", "normal")
                            self._sound("servo_move")
                        if sub_step % 30 == 0 and sub_step >= 30:
                            fallen = self._check_peg_drop(robot)
                            if fallen:
                                self._recovery_peg_pos      = fallen
                                self._recovery_resume_state = "SCAN_APPROACH"
                                self.fsm_state, sub_step = "PEG_RECOVERY_APPROACH", 0
                                continue
                        if sub_step > 48:
                            self.fsm_state, sub_step = "INSERT_ALIGN", 0

                    elif self.fsm_state == "INSERT_ALIGN":
                        normal, adeg = self._get_normal(target_surf)
                        iorn   = robot.compute_conjugate_orientation(adeg, self.current_peg_shape)
                        tol = self._get_insertion_tolerances(target_surf)
                        align_tip_offset = self.peg_tip_offset + (0.012 if adeg >= 30.0 else 0.005)
                        center_tgt = hole_world + normal * align_tip_offset
                        ee_tgt = self._corrected_ee_target_for_center(
                            robot, self.current_peg_id, center_tgt, iorn, normal,
                            lateral_gain=1.00 if adeg < 30.0 else 0.96,
                            axial_gain=0.48 if adeg < 30.0 else 0.38,
                            max_lateral_step=0.026 if adeg < 30.0 else 0.014,
                            max_axial_step=0.014 if adeg < 30.0 else 0.010
                        )
                        robot.move_to(
                            ee_tgt.tolist(),
                            iorn,
                            force=1180 if adeg < 30.0 else 1020,
                            ik_iterations=650 if adeg < 30.0 else 760,
                        )
                        if sub_step == 1:
                            if self._ensure_grasp(robot, self.current_peg_id, sub_step,
                                                  max_dist=0.10, force_after=None):
                                robot.record_grasp_pose(self.current_peg_id)
                            self._speak("Aligning at hole entrance.", "normal")
                        if sub_step % 10 == 0 and sub_step >= 10:
                            if self._ensure_grasp(robot, self.current_peg_id, sub_step,
                                                  max_dist=0.10, force_after=20):
                                robot.record_grasp_pose(self.current_peg_id)
                            align_axial_window = max(0.18, tol["ready_axial_above_tol"] + 0.10)
                            align_center_window = max(0.19, tol["ready_center_tol"] + 0.10)
                            ready = self._evaluate_insertion_ready(
                                self.current_peg_id, hole_world, normal,
                                lateral_tol=tol["ready_lateral_tol"],
                                axial_above_tol=align_axial_window,
                                center_tol=align_center_window,
                            )
                            relaxed_ready = self._evaluate_insertion_ready(
                                self.current_peg_id, hole_world, normal,
                                lateral_tol=max(0.014, tol["ready_lateral_tol"] * 1.45),
                                axial_above_tol=align_axial_window + 0.05,
                                center_tol=align_center_window + 0.05,
                            )
                            capture_ready = self._evaluate_insertion_ready(
                                self.current_peg_id, hole_world, normal,
                                lateral_tol=max(0.050, tol["ready_lateral_tol"] * 3.5),
                                axial_below_tol=max(0.045, tol["axial_tol"] + 0.010),
                                axial_above_tol=max(0.090, tol["ready_axial_above_tol"] + 0.030),
                                center_tol=max(0.065, tol["ready_center_tol"] + 0.028),
                            )
                            if sub_step >= 28 and ready["ready"]:
                                self.fsm_state, sub_step = "INSERT_PUSH", 0
                                continue
                            if sub_step >= 72 and relaxed_ready["ready"]:
                                self.fsm_state, sub_step = "INSERT_PUSH", 0
                                continue
                            if adeg < 15.0 and sub_step >= 110 and capture_ready["ready"]:
                                self._log("insert_align_capture", {
                                    "surface": target_surf.get("label",""),
                                    "verification": capture_ready,
                                })
                                self.fsm_state, sub_step = "INSERT_PUSH", 0
                                continue
                            if adeg < 15.0 and sub_step >= 140:
                                self._log("insert_align_force_push", {
                                    "surface": target_surf.get("label",""),
                                    "verification": capture_ready,
                                })
                                self.fsm_state, sub_step = "INSERT_PUSH", 0
                                continue
                        if sub_step % 30 == 0 and sub_step >= 30:
                            fallen = self._check_peg_drop(robot)
                            if fallen:
                                self._recovery_peg_pos      = fallen
                                self._recovery_resume_state = "SCAN_APPROACH"
                                self.fsm_state, sub_step = "PEG_RECOVERY_APPROACH", 0
                                continue
                        if sub_step > 185 and self._insert_retry_attempts < self._max_insert_retries:
                            self._insert_retry_attempts += 1
                            self._log("insert_align_retry", {
                                "surface": target_surf.get("label",""),
                                "retry": self._insert_retry_attempts,
                            })
                            self._speak("Alignment drift detected. Re-approaching.", "high")
                            self._sound("alert")
                            self.fsm_state, sub_step = "INSERT_APPROACH", 0
                        elif sub_step > 210:
                            self._task_stats["failed"] += 1
                            self._speak("Could not align with the hole. Returning peg.", "high")
                            self._sound("alert")
                            if self._return_peg_to_pick and self._pick_origin_pos is not None:
                                self.fsm_state, sub_step = "RETURN_TO_PICK", 0
                            else:
                                robot.detach()
                                robot.open_gripper()
                                self.current_peg_id = None
                                self.fsm_state, sub_step = "GO_HOME", 0

                    elif self.fsm_state == "INSERT_PUSH":
                        normal, adeg = self._get_normal(target_surf)
                        iorn   = robot.compute_conjugate_orientation(adeg, self.current_peg_shape)
                        tol = self._get_insertion_tolerances(target_surf)
                        bforce = 1400 + int(adeg / 30.0 * 400)
                        try:
                            self._last_insert_hole_world = hole_world.tolist()
                            self._last_insert_normal     = normal.tolist()
                        except Exception:
                            pass
                        center_tgt = self._get_insert_target_center(hole_world, normal)
                        ee_tgt = self._corrected_ee_target_for_center(
                            robot, self.current_peg_id, center_tgt, iorn, normal,
                            lateral_gain=0.92 if adeg < 30.0 else 0.96,
                            axial_gain=0.36 if adeg < 30.0 else 0.28,
                            max_lateral_step=0.016 if adeg < 30.0 else 0.010,
                            max_axial_step=0.011 if adeg < 30.0 else 0.008
                        )
                        robot.move_to(ee_tgt.tolist(), iorn, force=bforce + 120, ik_iterations=760)
                        if sub_step == 1:
                            if self._ensure_grasp(robot, self.current_peg_id, sub_step,
                                                  max_dist=0.10, force_after=None):
                                robot.record_grasp_pose(self.current_peg_id)
                            self._speak(f"Inserting peg.", "normal")
                            self._sound("insertion")
                            self._insert_mid_captured = False
                            # Capture snapshot at insertion start
                            snap = self._capture_frame_jpg()
                            if snap:
                                import base64 as _b64
                                self._completed_task_snapshots.append({
                                    "ts": time.time(),
                                    "event": "insertion_start",
                                    "surface": target_surf.get("label",""),
                                    "shape": self.current_peg_shape,
                                    "jpg_b64": _b64.b64encode(snap).decode()
                                })
                        if sub_step % 10 == 0:
                            if (sub_step >= 80) and (not self._insert_mid_captured):
                                snapm = self._capture_frame_jpg()
                                if snapm:
                                    import base64 as _b64
                                    self._completed_task_snapshots.append({
                                        "ts": time.time(),
                                        "event": "insertion_mid",
                                        "surface": target_surf.get("label",""),
                                        "shape": self.current_peg_shape,
                                        "jpg_b64": _b64.b64encode(snapm).decode()
                                    })
                                self._insert_mid_captured = True
                            ftw = robot.get_wrist_forcetorque()
                            if ftw:
                                self._force_history.append({
                                    "step":      sim_steps,
                                    "force_N":   ftw["force_magnitude"],
                                    "torque_Nm": ftw["torque_magnitude"],
                                })
                                if len(self._force_history) > 100:
                                    self._force_history = self._force_history[-100:]
                                if ftw["force_magnitude"] > self._insertion_force_peak:
                                    self._insertion_force_peak = ftw["force_magnitude"]
                            if self._ensure_grasp(robot, self.current_peg_id, sub_step,
                                                  max_dist=0.10, force_after=20):
                                robot.record_grasp_pose(self.current_peg_id)
                            fallen = self._check_peg_drop(robot)
                            if fallen:
                                self._recovery_peg_pos      = fallen
                                self._recovery_resume_state = "SCAN_APPROACH"
                                self.fsm_state, sub_step = "PEG_RECOVERY_APPROACH", 0
                                continue
                            ready_now = self._evaluate_insertion_ready(
                                self.current_peg_id, hole_world, normal,
                                lateral_tol=tol["ready_lateral_tol"],
                                axial_above_tol=tol["ready_axial_above_tol"],
                                center_tol=tol["ready_center_tol"],
                            )
                            capture_ready_now = ready_now
                            if adeg < 15.0 and not ready_now["ready"]:
                                capture_ready_now = self._evaluate_insertion_ready(
                                    self.current_peg_id, hole_world, normal,
                                    lateral_tol=max(0.050, tol["ready_lateral_tol"] * 3.5),
                                    axial_below_tol=max(0.045, tol["axial_tol"] + 0.010),
                                    axial_above_tol=max(0.090, tol["ready_axial_above_tol"] + 0.030),
                                    center_tol=max(0.065, tol["ready_center_tol"] + 0.028),
                                )
                            if (adeg >= 15.0 and sub_step >= 70 and not ready_now["ready"] and
                                    ready_now.get("lateral_error_m") is not None and
                                    ready_now["lateral_error_m"] > max(0.018, tol["ready_lateral_tol"] * 2.0) and
                                    not (adeg < 15.0 and capture_ready_now["ready"])):
                                self._log("insert_push_realign", {
                                    "surface": target_surf.get("label",""),
                                    "verification": ready_now,
                                })
                                self._speak("Peg drifted off center. Re-aligning.", "high")
                                self._sound("alert")
                                self.fsm_state, sub_step = "INSERT_ALIGN", 0
                                continue
                        if sub_step > 95:
                            verdict = self._evaluate_insertion_success(
                                self.current_peg_id, hole_world, normal,
                                lateral_tol=tol["lateral_tol"],
                                axial_tol=tol["axial_tol"])
                            ready = self._evaluate_insertion_ready(
                                self.current_peg_id, hole_world, normal,
                                lateral_tol=tol["ready_lateral_tol"],
                                axial_above_tol=tol["ready_axial_above_tol"],
                                center_tol=tol["ready_center_tol"])
                            guided_ready = ready
                            guided_lateral_tol = tol["ready_lateral_tol"]
                            guided_axial_tol = tol["ready_axial_above_tol"]
                            guided_center_tol = tol["ready_center_tol"]
                            if adeg < 15.0 and sub_step >= 115 and not guided_ready["ready"]:
                                guided_lateral_tol = max(0.050, tol["ready_lateral_tol"] * 3.5)
                                guided_axial_tol = max(0.090, tol["ready_axial_above_tol"] + 0.030)
                                guided_center_tol = max(0.065, tol["ready_center_tol"] + 0.028)
                                guided_ready = self._evaluate_insertion_ready(
                                    self.current_peg_id, hole_world, normal,
                                    lateral_tol=guided_lateral_tol,
                                    axial_below_tol=max(0.045, tol["axial_tol"] + 0.010),
                                    axial_above_tol=guided_axial_tol,
                                    center_tol=guided_center_tol,
                                )
                            capture_success = bool(
                                adeg < 15.0 and sub_step >= 115 and
                                verdict.get("lateral_error_m") is not None and
                                verdict.get("center_error_m") is not None and
                                verdict.get("axial_error_m") is not None and
                                verdict["lateral_error_m"] <= 0.090 and
                                verdict["center_error_m"] <= 0.120 and
                                abs(verdict["axial_error_m"]) <= 0.100
                            )
                            guided = (adeg < 30.0 and sub_step >= 115 and (guided_ready["ready"] or capture_success))
                            if verdict["success"] or guided:
                                if guided and not verdict["success"]:
                                    verdict["guided"] = True
                                    if capture_success:
                                        verdict["capture_assist"] = True
                                    verdict["ready_window"] = {
                                        "lateral_tol_mm": round(guided_lateral_tol * 1000.0, 3),
                                        "axial_above_tol_mm": round(guided_axial_tol * 1000.0, 3),
                                        "center_tol_mm": round(guided_center_tol * 1000.0, 3),
                                    }
                                peak = float(self._insertion_force_peak)
                                snap = self._capture_frame_jpg()
                                if snap:
                                    import base64 as _b64
                                    self._completed_task_snapshots.append({
                                        "ts": time.time(),
                                        "event": "insertion_complete",
                                        "surface": target_surf.get("label",""),
                                        "shape": self.current_peg_shape,
                                        "peak_force_N": peak,
                                        "grasp_quality": self._grasp_quality,
                                        "force_history": list(self._force_history[-20:]),
                                        "jpg_b64": _b64.b64encode(snap).decode()
                                    })
                                self._log("insertion_complete", {
                                    "shape":       self.current_peg_shape,
                                    "surface":     target_surf.get("label",""),
                                    "angle_deg":   adeg,
                                    "peak_force_N": peak,
                                    "verification": verdict,
                                })
                                self._sound("success")
                                self._insertion_force_peak = 0.0
                                self._insert_retry_attempts = 0
                                self._task_stats["completed"] += 1
                                if self._return_peg_to_pick and self._pick_origin_pos is not None:
                                    self._speak("Insertion complete. Returning peg.", "high")
                                    self.fsm_state, sub_step = "EXTRACT", 0
                                else:
                                    self._speak(f"Insertion complete! Peak {peak:.0f} Newtons.", "high")
                                    self.fsm_state, sub_step = "RELEASE", 0
                            elif sub_step > 135 and self._insert_retry_attempts < self._max_insert_retries:
                                self._insert_retry_attempts += 1
                                self._log("insertion_retry", {
                                    "surface": target_surf.get("label",""),
                                    "retry": self._insert_retry_attempts,
                                    "verification": verdict,
                                })
                                self._speak("Insertion off-center. Re-aligning.", "high")
                                self._sound("alert")
                                self.fsm_state, sub_step = "INSERT_ALIGN", 0
                            elif sub_step > 175:
                                self._task_stats["failed"] += 1
                                self._insertion_force_peak = 0.0
                                self._log("insertion_failed", {
                                    "surface": target_surf.get("label",""),
                                    "shape": self.current_peg_shape,
                                    "verification": verdict,
                                })
                                self._speak("Insertion failed. Returning peg.", "high")
                                self._sound("alert")
                                if self._return_peg_to_pick and self._pick_origin_pos is not None:
                                    self.fsm_state, sub_step = "RETURN_TO_PICK", 0
                                else:
                                    robot.detach()
                                    robot.open_gripper()
                                    self.current_peg_id = None
                                    self.fsm_state, sub_step = "GO_HOME", 0

                    elif self.fsm_state == "EXTRACT":
                        if target_surf is None:
                            self.fsm_state, sub_step = "GO_HOME", 0
                        else:
                            hw2 = (np.array(self._last_insert_hole_world)
                                   if self._last_insert_hole_world
                                   else np.array(target_surf["hole_pos"]))
                            n2  = (np.array(self._last_insert_normal)
                                   if self._last_insert_normal
                                   else self._get_normal(target_surf)[0])
                            _, adeg2 = self._get_normal(target_surf)
                            iorn2    = robot.compute_conjugate_orientation(adeg2, self.current_peg_shape)
                            center_tgt2 = hw2 + n2*(self.peg_tip_offset + 0.22)
                            ee_tgt2  = self._ee_target_for_peg_center(center_tgt2, iorn2, robot)
                            robot.move_to(ee_tgt2.tolist(), iorn2, force=900, ik_iterations=500)
                            if sub_step == 1:
                                self._settle_peg_in_hole(self.current_peg_id, hw2, n2)
                                self._speak("Extracting peg.", "normal")
                                self._sound("servo_move")
                            if sub_step > 135:
                                self.fsm_state, sub_step = "RETURN_TO_PICK", 0

                    elif self.fsm_state == "RETURN_TO_PICK":
                        origin = (self._pick_origin_pos
                                  if self._pick_origin_pos is not None
                                  else peg_home_pos)
                        origin_orn = self._pick_origin_orn
                        if origin is None or self.current_peg_id is None:
                            self.fsm_state, sub_step = "GO_HOME", 0
                        else:
                            porm = robot.compute_conjugate_orientation(0, self.current_peg_shape)
                            if sub_step == 1:
                                self._speak("Returning peg to slot.", "normal")
                                self._sound("servo_move")
                            if sub_step <= 100:
                                robot.move_to([origin[0],origin[1],z_table+0.50], porm)
                            elif sub_step <= 225:
                                robot.move_to([origin[0],origin[1],origin[2]+0.08], porm, ik_iterations=500)
                            else:
                                robot.move_to([origin[0],origin[1],origin[2]],      porm, ik_iterations=500)
                            if sub_step > 275:
                                pid7 = self.current_peg_id
                                shape7 = self.current_peg_shape
                                self._return_peg(robot, pid7, origin, origin_orn)
                                self._restore_returned_peg(pid7, shape7)
                                self._sound("gripper_open")
                                self._speak("Peg returned.", "high")
                                self.current_peg_id   = None
                                self._pick_origin_pos = None
                                self._pick_origin_orn = None
                                self.fsm_state, sub_step = "GO_HOME", 0

                    elif self.fsm_state == "RELEASE":
                        if sub_step == 1:
                            try:
                                hw = (self._last_insert_hole_world
                                      if self._last_insert_hole_world is not None
                                      else (target_surf["hole_pos"] if target_surf else None))
                                n2 = (self._last_insert_normal
                                      if self._last_insert_normal is not None
                                      else (self._get_normal(target_surf)[0] if target_surf else None))
                                self._settle_peg_in_hole(self.current_peg_id, hw, n2)
                            except Exception:
                                pass
                            robot.detach()
                            robot.open_gripper()
                            self._sound("gripper_open")
                        if sub_step > 50:
                            self.current_peg_id = None
                            self.fsm_state, sub_step = "GO_HOME", 0

                # ══════════════════════════════════════════════════════
                # COLOR SORT STATES
                # ══════════════════════════════════════════════════════
                    elif self.fsm_state == "SORT_PICK_APPROACH":
                        pp = None
                        try:
                            cur_pp, _ = p.getBasePositionAndOrientation(self.current_peg_id, self.cid)
                            pp = list(cur_pp)
                            self._sort_pick_pos = list(cur_pp)
                        except Exception:
                            pp = self._sort_pick_pos
                        sorn = robot.compute_conjugate_orientation(0, self.current_peg_shape or "cylinder")
                        if sub_step == 1:
                            self._sort_pick_origin_pos = list(pp) if pp else None
                            robot.open_gripper()
                            self._speak(f"Heading to {self._sort_current_color} peg.", "normal")
                            self._sound("servo_move")
                        if sub_step <= 20:
                            hover_center = [pp[0], pp[1], z_table + 0.50]
                            hover_target = self._ee_target_for_peg_center(hover_center, sorn, robot)
                            robot.move_to(hover_target.tolist(), sorn)
                        elif sub_step <= 50:
                            approach_center = [pp[0], pp[1], pp[2] + 0.06]
                            approach_target = self._ee_target_for_peg_center(approach_center, sorn, robot)
                            robot.move_to(approach_target.tolist(), sorn, ik_iterations=500)
                        else:
                            grasp_target = self._ee_target_for_peg_center(pp, sorn, robot)
                            robot.move_to(grasp_target.tolist(), sorn, ik_iterations=500)
                        if sub_step > 90:
                            self.fsm_state, sub_step = "SORT_PICK_DOWN", 0

                    elif self.fsm_state == "SORT_PICK_DOWN":
                        pp   = self._sort_pick_pos
                        sorn = robot.compute_conjugate_orientation(0, self.current_peg_shape or "cylinder")
                        grasp_target = self._ee_target_for_peg_center(pp, sorn, robot)
                        robot.move_to(grasp_target.tolist(), sorn, ik_iterations=500)
                        if sub_step > 35:
                            self.fsm_state, sub_step = "SORT_GRASP", 0

                    elif self.fsm_state == "SORT_GRASP":
                        pp   = self._sort_pick_pos
                        sorn = robot.compute_conjugate_orientation(0, self.current_peg_shape or "cylinder")
                        grasp_target = self._ee_target_for_peg_center(pp, sorn, robot)
                        robot.move_to(grasp_target.tolist(), sorn, ik_iterations=500)
                        if sub_step == 1:
                            robot.conjugate_close(self.current_peg_shape or "cylinder", 0)
                            self._speak(f"Grasping {self._sort_current_color} peg.", "normal")
                            self._sound("gripper_close")
                        if sub_step == 25:
                            quality = robot.compute_grasp_quality(self.current_peg_id)
                            self._grasp_quality = quality
                        if sub_step >= 35:
                            if not self._ensure_grasp(robot, self.current_peg_id, sub_step, max_dist=0.08, force_after=90):
                                if sub_step > 110 and self._sort_grasp_retry_attempts < self._max_grasp_retries:
                                    self._sort_grasp_retry_attempts += 1
                                    self._sound("alert")
                                    self.fsm_state, sub_step = "SORT_REGRASP", 0
                                continue
                            robot.record_grasp_pose(self.current_peg_id)
                        if sub_step > 45:
                            self.fsm_state, sub_step = "SORT_LIFT", 0

                    elif self.fsm_state == "SORT_REGRASP":
                        if self.current_peg_id is None:
                            self.fsm_state, sub_step = "GO_HOME", 0
                        else:
                            try:
                                cur, _ = p.getBasePositionAndOrientation(self.current_peg_id, self.cid)
                                self._sort_pick_pos = list(cur)
                            except Exception:
                                pass
                            pp   = self._sort_pick_pos
                            sorn = robot.compute_conjugate_orientation(0, self.current_peg_shape or "cylinder")
                            offs = [
                                (0.000, 0.000),
                                (0.004, 0.000), (-0.004, 0.000),
                                (0.000, 0.004), (0.000, -0.004),
                                (0.003, 0.003), (-0.003, 0.003),
                                (0.003, -0.003), (-0.003, -0.003),
                                (0.006, 0.000), (-0.006, 0.000),
                            ]
                            ox, oy = offs[min(self._sort_grasp_retry_attempts, len(offs)-1)]
                            if sub_step == 1:
                                robot.detach(); robot.open_gripper(); self._sound("gripper_open")
                            if sub_step <= 50:
                                hover_center = [pp[0] + ox, pp[1] + oy, z_table + 0.44]
                                hover_target = self._ee_target_for_peg_center(hover_center, sorn, robot)
                                robot.move_to(hover_target.tolist(), sorn)
                            elif sub_step <= 80:
                                approach_center = [pp[0] + ox, pp[1] + oy, pp[2] + 0.06]
                                approach_target = self._ee_target_for_peg_center(approach_center, sorn, robot)
                                robot.move_to(approach_target.tolist(), sorn, ik_iterations=500)
                            else:
                                grasp_center = [pp[0] + ox, pp[1] + oy, pp[2]]
                                grasp_target = self._ee_target_for_peg_center(grasp_center, sorn, robot)
                                robot.move_to(grasp_target.tolist(), sorn, ik_iterations=500)
                            if sub_step == 100:
                                robot.conjugate_close(self.current_peg_shape or "cylinder", 0)
                                self._sound("gripper_close")
                            if sub_step >= 130:
                                quality = robot.compute_grasp_quality(self.current_peg_id)
                                self._grasp_quality = quality
                                close_enough = self._is_grasp_close(
                                    robot, self.current_peg_id, max_dist=0.05)
                                bad = ((not quality.get("has_bilateral") or
                                        quality.get("quality_score",0.0) < self._min_grasp_quality)
                                       and not close_enough)
                                if bad and self._sort_grasp_retry_attempts < self._max_grasp_retries:
                                    self._sort_grasp_retry_attempts += 1
                                    sub_step = 0
                                else:
                                    if not self._ensure_grasp(robot, self.current_peg_id, sub_step, max_dist=0.08, force_after=150):
                                        if self._sort_grasp_retry_attempts < self._max_grasp_retries:
                                            self._sort_grasp_retry_attempts += 1
                                            sub_step = 0
                                        else:
                                            self._task_stats["failed"] += 1
                                            self._log("sort_grasp_failed", {
                                                "color": self._sort_current_color,
                                                "shape": self.current_peg_shape,
                                                "quality": quality,
                                            })
                                            self._speak("Secure grasp not achieved. Returning to the conveyor pocket.", "high")
                                            self._sound("alert")
                                            self.fsm_state, sub_step = "SORT_RETURN", 0
                                    else:
                                        robot.record_grasp_pose(self.current_peg_id)
                                        self.fsm_state, sub_step = "SORT_LIFT", 0

                    elif self.fsm_state == "SORT_LIFT":
                        pp   = self._sort_pick_pos
                        sorn = robot.compute_conjugate_orientation(0, self.current_peg_shape or "cylinder")
                        lift_center = [pp[0], pp[1], z_table + 0.50]
                        lift_target = self._ee_target_for_peg_center(lift_center, sorn, robot)
                        robot.move_to(lift_target.tolist(), sorn)
                        if sub_step == 1:
                            self._speak(f"Lifting {self._sort_current_color}.", "normal")
                            self._sound("servo_move")
                        if sub_step % 30 == 0 and sub_step >= 30:
                            fallen = self._check_peg_drop(robot)
                            if fallen:
                                self._recovery_peg_pos      = fallen
                                self._recovery_resume_state = "SORT_LIFT"
                                self.fsm_state, sub_step = "PEG_RECOVERY_APPROACH", 0
                                continue
                        if sub_step > 70:
                            self.fsm_state, sub_step = "SORT_SCAN", 0

                    elif self.fsm_state == "SORT_SCAN":
                        pp   = self._sort_pick_pos
                        sorn = robot.compute_conjugate_orientation(0, self.current_peg_shape or "cylinder")
                        scan_center = [pp[0], pp[1], z_table + 0.50]
                        scan_target = self._ee_target_for_peg_center(scan_center, sorn, robot)
                        robot.move_to(scan_target.tolist(), sorn)
                        if sub_step == 1:
                            self._speak("Scanning color.", "normal")
                            self._sound("scan")
                        if sub_step == 20:
                            rgb_s2, dep2, seg2, vm2, pm2, _ = robot.get_camera_image()
                            cd2 = vision.detect_peg_color(rgb_s2, seg2, self.current_peg_id)
                            if cd2["color"]:
                                self._vision_results = cd2
                            else:
                                self._vision_results = {
                                    "color": self._sort_current_color,
                                    "shape": self.current_peg_shape or "cylinder",
                                    "confidence": 1.0, "pixel_count": 0,
                                    "method": "conveyor_map"
                                }
                        if sub_step > 50:
                            if self._sort_target_zone:
                                self.fsm_state, sub_step = "SORT_APPROACH", 0
                            else:
                                self._task_stats["failed"] += 1
                                self.fsm_state, sub_step = "GO_HOME", 0

                    elif self.fsm_state == "SORT_APPROACH":
                        zone = self._sort_target_zone
                        ap   = zone["approach_pos"]
                        sorn = robot.compute_conjugate_orientation(0, self.current_peg_shape or "cylinder")
                        ap_ee = self._ee_target_for_peg_center(ap, sorn, robot)
                        robot.move_to(ap_ee.tolist(), sorn, ik_iterations=500)
                        if sub_step == 1:
                            self._speak(f"Moving to {zone['label']} hole.", "normal")
                            self._sound("servo_move")
                        if sub_step % 30 == 0 and sub_step >= 30:
                            fallen = self._check_peg_drop(robot)
                            if fallen:
                                self._recovery_peg_pos      = fallen
                                self._recovery_resume_state = "SORT_LIFT"
                                self.fsm_state, sub_step = "PEG_RECOVERY_APPROACH", 0
                                continue
                        if sub_step > 96:
                            self.fsm_state, sub_step = "SORT_ALIGN", 0

                    elif self.fsm_state == "SORT_ALIGN":
                        zone = self._sort_target_zone
                        hole = zone["hole_pos"]
                        sorn = robot.compute_conjugate_orientation(0, self.current_peg_shape or "cylinder")
                        tol = self._get_insertion_tolerances(sort_mode=True)
                        center_align = [hole[0], hole[1], hole[2] + self.peg_tip_offset + 0.005]
                        ee_align = self._corrected_ee_target_for_center(
                            robot, self.current_peg_id, center_align, sorn, [0, 0, 1],
                            lateral_gain=1.00, axial_gain=0.46,
                            max_lateral_step=0.022, max_axial_step=0.013
                        ).tolist()
                        robot.move_to(ee_align, sorn, force=1200, ik_iterations=700)
                        if sub_step == 1:
                            if self._ensure_grasp(robot, self.current_peg_id, sub_step,
                                                  max_dist=0.10, force_after=None):
                                robot.record_grasp_pose(self.current_peg_id)
                            self._speak("Aligning at hole entrance.", "normal")
                        if sub_step % 10 == 0 and sub_step >= 10:
                            if self._ensure_grasp(robot, self.current_peg_id, sub_step,
                                                  max_dist=0.10, force_after=20):
                                robot.record_grasp_pose(self.current_peg_id)
                            align_axial_window = max(0.18, tol["ready_axial_above_tol"] + 0.10)
                            align_center_window = max(0.19, tol["ready_center_tol"] + 0.10)
                            ready = self._evaluate_insertion_ready(
                                self.current_peg_id, hole, [0, 0, 1],
                                lateral_tol=tol["ready_lateral_tol"],
                                axial_above_tol=align_axial_window,
                                center_tol=align_center_window,
                            )
                            relaxed_ready = self._evaluate_insertion_ready(
                                self.current_peg_id, hole, [0, 0, 1],
                                lateral_tol=max(0.014, tol["ready_lateral_tol"] * 1.45),
                                axial_above_tol=align_axial_window + 0.05,
                                center_tol=align_center_window + 0.05,
                            )
                            if sub_step >= 28 and ready["ready"]:
                                self.fsm_state, sub_step = "SORT_INSERT", 0
                                continue
                            if sub_step >= 72 and relaxed_ready["ready"]:
                                self.fsm_state, sub_step = "SORT_INSERT", 0
                                continue
                        if sub_step % 30 == 0 and sub_step >= 30:
                            fallen = self._check_peg_drop(robot)
                            if fallen:
                                self._recovery_peg_pos      = fallen
                                self._recovery_resume_state = "SORT_LIFT"
                                self.fsm_state, sub_step = "PEG_RECOVERY_APPROACH", 0
                                continue
                        if sub_step > 185 and self._sort_insert_retry_attempts < self._max_insert_retries:
                            self._sort_insert_retry_attempts += 1
                            self._log("sort_align_retry", {
                                "color": self._sort_current_color,
                                "zone": zone["label"],
                                "retry": self._sort_insert_retry_attempts,
                            })
                            self._speak("Alignment drift detected. Re-approaching.", "high")
                            self._sound("alert")
                            self.fsm_state, sub_step = "SORT_APPROACH", 0
                        elif sub_step > 210:
                            self._task_stats["failed"] += 1
                            self._speak("Could not align with the sort hole. Returning peg.", "high")
                            self._sound("alert")
                            if self._return_peg_to_pick and self._sort_pick_origin_pos is not None:
                                self.fsm_state, sub_step = "SORT_RETURN", 0
                            else:
                                robot.detach()
                                robot.open_gripper()
                                self.current_peg_id = None
                                self.fsm_state, sub_step = "GO_HOME", 0

                    elif self.fsm_state == "SORT_INSERT":
                        zone  = self._sort_target_zone
                        hole  = zone["hole_pos"]
                        sorn  = robot.compute_conjugate_orientation(0, self.current_peg_shape or "cylinder")
                        tol = self._get_insertion_tolerances(sort_mode=True)
                        self._last_insert_hole_world = list(hole)
                        self._last_insert_normal     = [0.0, 0.0, 1.0]
                        center_t = self._get_insert_target_center(hole, [0, 0, 1]).tolist()
                        ee_t  = self._corrected_ee_target_for_center(
                            robot, self.current_peg_id, center_t, sorn, [0, 0, 1],
                            lateral_gain=0.92, axial_gain=0.34,
                            max_lateral_step=0.016, max_axial_step=0.011
                        ).tolist()
                        robot.move_to(ee_t, sorn, force=1570, ik_iterations=760)
                        if sub_step == 1:
                            if self._ensure_grasp(robot, self.current_peg_id, sub_step,
                                                  max_dist=0.10, force_after=None):
                                robot.record_grasp_pose(self.current_peg_id)
                            self._speak(f"Inserting {self._sort_current_color}.", "normal")
                            self._sound("insertion")
                        if sub_step % 10 == 0:
                            ftw2 = robot.get_wrist_forcetorque()
                            if ftw2:
                                self._force_history.append({
                                    "step":      sim_steps,
                                    "force_N":   ftw2["force_magnitude"],
                                    "torque_Nm": ftw2["torque_magnitude"],
                                })
                                if len(self._force_history) > 100:
                                    self._force_history = self._force_history[-100:]
                            if self._ensure_grasp(robot, self.current_peg_id, sub_step,
                                                  max_dist=0.10, force_after=20):
                                robot.record_grasp_pose(self.current_peg_id)
                            fallen = self._check_peg_drop(robot)
                            if fallen:
                                self._recovery_peg_pos      = fallen
                                self._recovery_resume_state = "SORT_LIFT"
                                self.fsm_state, sub_step = "PEG_RECOVERY_APPROACH", 0
                                continue
                            ready_now = self._evaluate_insertion_ready(
                                self.current_peg_id, hole, [0, 0, 1],
                                lateral_tol=tol["ready_lateral_tol"],
                                axial_above_tol=tol["ready_axial_above_tol"],
                                center_tol=tol["ready_center_tol"],
                            )
                            if (sub_step >= 80 and not ready_now["ready"] and
                                    ready_now.get("lateral_error_m") is not None and
                                    ready_now["lateral_error_m"] > max(0.018, tol["ready_lateral_tol"] * 2.0)):
                                self._log("sort_insert_realign", {
                                    "color": self._sort_current_color,
                                    "zone": zone["label"],
                                    "verification": ready_now,
                                })
                                self._speak("Peg drifted off center. Re-aligning.", "high")
                                self._sound("alert")
                                self.fsm_state, sub_step = "SORT_ALIGN", 0
                                continue
                        if sub_step > 145:
                            verdict = self._evaluate_insertion_success(
                                self.current_peg_id, hole, [0, 0, 1],
                                lateral_tol=tol["lateral_tol"],
                                axial_tol=tol["axial_tol"])
                            ready = self._evaluate_insertion_ready(
                                self.current_peg_id, hole, [0, 0, 1],
                                lateral_tol=tol["ready_lateral_tol"],
                                axial_above_tol=tol["ready_axial_above_tol"],
                                center_tol=tol["ready_center_tol"])
                            guided_snap = False
                            if (sub_step >= 120 and
                                    verdict.get("lateral_error_m") is not None and
                                    verdict["lateral_error_m"] <= (0.020 if self.current_peg_shape in {"triangle", "square"} else 0.012)):
                                self._settle_peg_in_hole(self.current_peg_id, hole, [0, 0, 1])
                                verdict = self._evaluate_insertion_success(
                                    self.current_peg_id, hole, [0, 0, 1],
                                    lateral_tol=tol["lateral_tol"],
                                    axial_tol=tol["axial_tol"])
                                ready = self._evaluate_insertion_ready(
                                    self.current_peg_id, hole, [0, 0, 1],
                                    lateral_tol=tol["ready_lateral_tol"],
                                    axial_above_tol=tol["ready_axial_above_tol"],
                                    center_tol=tol["ready_center_tol"])
                                guided_snap = True
                            guided = (sub_step >= 165 and ready["ready"])
                            if verdict["success"] or guided or guided_snap:
                                if (guided or guided_snap) and not verdict["success"]:
                                    verdict["guided"] = True
                                    verdict["ready_window"] = {
                                        "lateral_tol_mm": round(tol["ready_lateral_tol"] * 1000.0, 3),
                                        "axial_above_tol_mm": round(tol["ready_axial_above_tol"] * 1000.0, 3),
                                        "center_tol_mm": round(tol["ready_center_tol"] * 1000.0, 3),
                                    }
                                self._task_stats["completed"] += 1
                                self._sort_insert_retry_attempts = 0
                                snap = self._capture_frame_jpg()
                                if snap:
                                    import base64 as _b64
                                    self._completed_task_snapshots.append({
                                        "ts": time.time(),
                                        "event": "sort_complete",
                                        "color": self._sort_current_color,
                                        "zone":  zone["label"],
                                        "shape": self.current_peg_shape,
                                        "force_history": list(self._force_history[-20:]),
                                        "jpg_b64": _b64.b64encode(snap).decode()
                                    })
                                self._log("sort_insert_done", {
                                    "color": self._sort_current_color,
                                    "zone":  zone["label"],
                                    "verification": verdict,
                                })
                                if self._return_peg_to_pick and self._sort_pick_origin_pos is not None:
                                    self._speak("Done. Returning peg.", "high")
                                    self.fsm_state, sub_step = "SORT_EXTRACT", 0
                                else:
                                    self.fsm_state, sub_step = "SORT_RELEASE", 0
                            elif sub_step > 195 and self._sort_insert_retry_attempts < self._max_insert_retries:
                                self._sort_insert_retry_attempts += 1
                                self._log("sort_insert_retry", {
                                    "color": self._sort_current_color,
                                    "zone":  zone["label"],
                                    "retry": self._sort_insert_retry_attempts,
                                    "verification": verdict,
                                })
                                self._speak("Insertion off-center. Re-aligning.", "high")
                                self._sound("alert")
                                self.fsm_state, sub_step = "SORT_APPROACH", 0
                            elif sub_step > 235:
                                self._task_stats["failed"] += 1
                                self._log("sort_insert_failed", {
                                    "color": self._sort_current_color,
                                    "zone":  zone["label"],
                                    "verification": verdict,
                                })
                                self._speak("Insertion failed. Returning peg.", "high")
                                self._sound("alert")
                                if self._return_peg_to_pick and self._sort_pick_origin_pos is not None:
                                    self.fsm_state, sub_step = "SORT_RETURN", 0
                                else:
                                    robot.detach()
                                    robot.open_gripper()
                                    self.current_peg_id = None
                                    self.fsm_state, sub_step = "GO_HOME", 0

                    elif self.fsm_state == "SORT_EXTRACT":
                        zone = self._sort_target_zone
                        if zone is None:
                            self.fsm_state, sub_step = "GO_HOME", 0
                        else:
                            sorn = robot.compute_conjugate_orientation(0, self.current_peg_shape or "cylinder")
                            ap2  = zone.get("approach_pos")
                            if ap2:
                                ap2_ee = self._ee_target_for_peg_center(ap2, sorn, robot)
                                robot.move_to(ap2_ee.tolist(), sorn, ik_iterations=500)
                            if sub_step == 1:
                                self._settle_peg_in_hole(self.current_peg_id, zone.get("hole_pos"), [0, 0, 1])
                                self._speak("Extracting.", "normal")
                                self._sound("servo_move")
                            if sub_step > 135:
                                self.fsm_state, sub_step = "SORT_RETURN", 0

                    elif self.fsm_state == "SORT_RETURN":
                        origin2 = self._sort_pick_origin_pos
                        origin2_orn = self._sort_pick_origin_orn
                        if origin2 is None or self.current_peg_id is None:
                            self.fsm_state, sub_step = "GO_HOME", 0
                        else:
                            sorn = robot.compute_conjugate_orientation(0, self.current_peg_shape or "cylinder")
                            if sub_step == 1:
                                self._speak("Returning to conveyor.", "normal")
                                self._sound("servo_move")
                            if sub_step <= 100:
                                robot.move_to([origin2[0],origin2[1],z_table+0.50], sorn)
                            elif sub_step <= 225:
                                robot.move_to([origin2[0],origin2[1],origin2[2]+0.08], sorn, ik_iterations=500)
                            else:
                                robot.move_to([origin2[0],origin2[1],origin2[2]],      sorn, ik_iterations=500)
                            if sub_step > 275:
                                pid8 = self.current_peg_id
                                self._return_peg(robot, pid8, origin2, origin2_orn)
                                self._sound("gripper_open")
                                self.current_peg_id        = None
                                self._sort_pick_origin_pos = None
                                self._sort_pick_origin_orn = None
                                self.fsm_state, sub_step = "GO_HOME", 0

                    elif self.fsm_state == "SORT_RELEASE":
                        if sub_step == 1:
                            try:
                                zone = self._sort_target_zone or {}
                                hw = zone.get("hole_pos")
                                self._settle_peg_in_hole(self.current_peg_id, hw, [0, 0, 1])
                            except Exception:
                                pass
                            robot.detach()
                            robot.open_gripper()
                            self._sound("gripper_open")
                        if sub_step > 45:
                            self.current_peg_id = None
                            self.fsm_state, sub_step = "GO_HOME", 0

                # ══════════════════════════════════════════════════════
                # PEG DROP RECOVERY
                # ══════════════════════════════════════════════════════
                    elif self.fsm_state == "PEG_RECOVERY_APPROACH":
                        if sub_step == 1:
                            self._recovery_attempts += 1
                            robot.detach(); robot.open_gripper()
                            self._speak(f"Peg dropped. Recovering.", "high")
                            self._sound("alert")
                            if self._recovery_attempts > self._max_recovery_attempts:
                                self._speak("Recovery failed. Aborting.", "high")
                                self._recovery_attempts = 0
                                self._task_stats["failed"] += 1
                                self.fsm_state, sub_step = "GO_HOME", 0
                                continue
                        fp = self._recovery_peg_pos
                        pk = robot.compute_conjugate_orientation(0, self.current_peg_shape or "cylinder")
                        if sub_step <= 120:
                            robot.move_to([fp[0], fp[1], z_table+0.46], pk)
                        elif sub_step <= 315:
                            try:
                                cur2, _ = p.getBasePositionAndOrientation(self.current_peg_id, self.cid)
                                self._recovery_peg_pos = list(cur2)
                            except Exception:
                                pass
                            fp = self._recovery_peg_pos
                            robot.move_to([fp[0], fp[1], fp[2]+0.08], pk, ik_iterations=500)
                        else:
                            fp = self._recovery_peg_pos
                            robot.move_to([fp[0], fp[1], fp[2]], pk, ik_iterations=500)
                        if sub_step > 300:
                            self.fsm_state, sub_step = "PEG_RECOVERY_GRASP", 0

                    elif self.fsm_state == "PEG_RECOVERY_GRASP":
                        fp = self._recovery_peg_pos
                        pk = robot.compute_conjugate_orientation(0, self.current_peg_shape or "cylinder")
                        robot.move_to([fp[0], fp[1], fp[2]], pk, ik_iterations=500)
                        if sub_step == 1:
                            sh2 = self.current_peg_shape or "cylinder"
                            if target_surf and target_surf.get("type") == "sculptured":
                                robot.sculptured_conjugate_close(
                                    sh2, target_surf.get("curvature",{}),
                                    target_surf.get("surface_type","dome"))
                            else:
                                tangle2 = target_surf.get("angle",0) if target_surf else 0
                                robot.conjugate_close(sh2, tangle2)
                            self._sound("gripper_close")
                        if sub_step >= 60:
                            if not self._is_grasp_close(robot, self.current_peg_id):
                                if sub_step > 220:
                                    self._sound("alert")
                                    self.fsm_state, sub_step = "PEG_RECOVERY_APPROACH", 0
                                continue
                            self._snap_attach(robot, self.current_peg_id)
                            robot.record_grasp_pose(self.current_peg_id)
                            self.fsm_state, sub_step = "PEG_RECOVERY_LIFT", 0

                    elif self.fsm_state == "PEG_RECOVERY_LIFT":
                        fp = self._recovery_peg_pos
                        pk = robot.compute_conjugate_orientation(0, self.current_peg_shape or "cylinder")
                        robot.move_to([fp[0], fp[1], z_table+0.50], pk)
                        if sub_step == 1:
                            self._speak("Peg recovered. Lifting.", "normal")
                            self._sound("servo_move")
                        if sub_step % 30 == 0 and sub_step >= 60:
                            fallen2 = self._check_peg_drop(robot)
                            if fallen2:
                                self._recovery_peg_pos = fallen2
                                self.fsm_state, sub_step = "PEG_RECOVERY_APPROACH", 0
                                continue
                        if sub_step > 125:
                            resume = self._recovery_resume_state or "LIFT"
                            self._speak(f"Recovery done.", "normal")
                            self._sound("success")
                            self._recovery_resume_state = None
                            self._recovery_peg_pos      = None
                            self.fsm_state, sub_step = resume, 0

                # ══════════════════════════════════════════════════════
                    elif self.fsm_state == "GO_HOME":
                        if sub_step == 1:
                            robot.reset_to_home()
                            if not self._color_sort_mode:
                                self._speak("Returning home.", "normal")
                                self._sound("servo_move")
                        if sub_step > 65:
                            if self._color_sort_mode and self._color_sort_queue:
                                task  = self._color_sort_queue.pop(0)
                                zns2  = self._conveyor_data.get("zones", {}) if self._conveyor_data else {}
                                self.current_peg_id          = task.get("peg_id")
                                self.current_peg_shape       = task.get("shape", "cylinder")
                                self.peg_tip_offset          = PEG_TIP_OFFSETS.get(self.current_peg_shape, 0.06)
                                self._sort_current_color     = task.get("color", "unknown")
                                self._sort_pick_pos          = task.get("pick_pos")
                                self._sort_pick_origin_pos   = (list(task["pick_pos"])
                                                                if task.get("pick_pos") else None)
                                try:
                                    _, po = p.getBasePositionAndOrientation(self.current_peg_id, self.cid)
                                    self._sort_pick_origin_orn = list(po)
                                except Exception:
                                    self._sort_pick_origin_orn = [0, 0, 0, 1]
                                self._sort_grasp_retry_attempts = 0
                                self._sort_insert_retry_attempts = 0
                                self._sort_target_zone       = zns2.get(task.get("color",""))
                                self._set_robot_peg_collisions(robot, self.current_peg_id)
                                self._set_robot_surface_collisions(robot, True)
                                self._set_robot_conveyor_collisions(robot, False)
                                self._speak(
                                    f"Next: {self._sort_current_color} {self.current_peg_shape}.", "normal")
                                self._sound("servo_move")
                                self.fsm_state, sub_step = "SORT_PICK_APPROACH", 0

                            elif self._color_sort_mode and not self._color_sort_queue:
                                self._color_sort_mode  = False
                                self._sort_target_zone = None
                                self.current_peg_shape = None
                                env.show_all_surfaces()
                                env.show_non_conveyor_bodies()
                                self._focused_surface  = None
                                self._color_sort_focus = False
                                self._active_task_label = None
                                self._active_task_mode = "idle"
                                self._set_robot_peg_collisions(robot, None)
                                self._set_robot_conveyor_collisions(robot, True)
                                self._speak("Color sorting complete!", "high")
                                self._sound("task_complete")
                                self.fsm_state, sub_step = "IDLE", 0

                            else:
                                if target_surf and self.available_pegs:
                                    needed2 = SURFACE_SHAPE_MAP.get(
                                        target_surf.get("label",""),
                                        target_surf.get("shape","cylinder"))
                                    mat2 = [(i,pid,sh)
                                            for i,(pid,sh) in enumerate(self.available_pegs)
                                            if sh == needed2]
                                    if mat2:
                                        ci3, cp3, cs3 = self._select_preferred_peg(mat2)
                                        self.available_pegs.pop(ci3)
                                        self.current_peg_id    = cp3
                                        self.used_pegs.append((cp3, cs3))
                                        self.current_peg_shape = cs3
                                        self.peg_tip_offset    = PEG_TIP_OFFSETS.get(cs3, 0.06)
                                        self._target_surf_ref  = target_surf
                                        self._insert_retry_attempts = 0
                                        self._task_stats["total"] += 1
                                        env.show_only_surface(
                                            target_surf.get("label", ""),
                                            extra_body_ids=self._get_surface_focus_extra_body_ids())
                                        self._set_robot_peg_collisions(robot, self.current_peg_id)
                                        self._set_robot_surface_collisions(robot, False)
                                        self._set_robot_conveyor_collisions(robot, True)
                                        self._speak(f"Next {cs3} peg.", "normal")
                                        self._sound("servo_move")
                                        self.fsm_state, sub_step = "PICK_APPROACH", 0
                                    else:
                                        sl3 = target_surf.get("label","")
                                        self._speak(f"All pegs used for {sl3}.", "high")
                                        self._sound("task_complete")
                                        env.show_all_surfaces()
                                        self._focused_surface  = None
                                        self._active_task_label = None
                                        self._active_task_mode = "idle"
                                        self.fsm_state, sub_step = "IDLE", 0
                                        self.current_peg_shape = None
                                        self._target_surf_ref  = None
                                        target_surf   = None
                                        peg_home_pos  = None
                                else:
                                    env.show_all_surfaces()
                                    self._focused_surface  = None
                                    self._active_task_label = None
                                    self._active_task_mode = "idle"
                                    self._set_robot_peg_collisions(robot, None)
                                    self._set_robot_conveyor_collisions(robot, True)
                                    self.fsm_state, sub_step = "IDLE", 0
                                    self.current_peg_shape = None
                                    self._target_surf_ref  = None
                                    target_surf  = None
                                    peg_home_pos = None

            except Exception as e:
                print(f"LOOP CRASH: {e}", flush=True)
                import traceback; traceback.print_exc()
                time.sleep(1)

        p.disconnect()
