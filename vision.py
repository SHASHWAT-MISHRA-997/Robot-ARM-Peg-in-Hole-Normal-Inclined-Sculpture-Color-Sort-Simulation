"""
vision.py — Industrial Computer Vision Pipeline
FIXED:
  - BGR→RGB bug removed (PyBullet returns RGB directly)
  - Multi-range HSV for all 6 colors
  - HSV match priority over RGB distance
  - Orange/Blue/Green expanded ranges for conveyor pegs
"""
import cv2
import numpy as np
import math


class PegCNN:
    CLASSES    = ["cylinder", "square", "triangle"]
    N_CLASSES  = 3
    INPUT_SIZE = 32

    def __init__(self):
        self._class_centroids = {
            "cylinder": np.array([255.0,   0.0, 102.0]),
            "square":   np.array([255.0, 204.0,   0.0]),
            "triangle": np.array([  0.0, 204.0, 255.0]),
        }

    def preprocess(self, crop_rgb):
        if crop_rgb is None or crop_rgb.size == 0:
            return None
        return cv2.resize(crop_rgb, (self.INPUT_SIZE, self.INPUT_SIZE),
                          interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

    def extract_features(self, norm):
        if norm is None:
            return np.zeros(60)
        img  = (norm * 255).astype(np.uint8)
        feat = []
        for ch in range(3):
            h = cv2.calcHist([img], [ch], None, [16], [0, 256]).flatten()
            feat.extend((h / (h.sum() + 1e-8)).tolist())
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag  = np.sqrt(sx**2 + sy**2)
        ori  = np.arctan2(sy, sx + 1e-8)
        ms   = mag.sum() + 1e-8
        for i in range(8):
            lo = -math.pi + i * 2 * math.pi / 8
            hi = lo + 2 * math.pi / 8
            feat.append(float(mag[(ori >= lo) & (ori < hi)].sum()) / ms)
        mo  = cv2.moments(gray)
        hu  = cv2.HuMoments(mo).flatten()
        hul = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        feat.extend(hul[:4].tolist())
        return np.array(feat, dtype=np.float32)

    def classify(self, crop_rgb, mean_rgb=None):
        pre  = self.preprocess(crop_rgb)
        self.extract_features(pre)
        rgb  = (np.array(mean_rgb, dtype=float) if mean_rgb is not None
                else (np.mean(crop_rgb.reshape(-1, 3), axis=0)
                      if crop_rgb is not None and crop_rgb.size > 0
                      else np.array([128, 128, 128])))
        dist = np.array([float(np.linalg.norm(rgb - self._class_centroids[c]))
                         for c in self.CLASSES])
        lg   = -dist / 50.0
        exp  = np.exp(lg - lg.max())
        prob = exp / (exp.sum() + 1e-8)
        bi   = int(np.argmax(prob))
        return {
            "class_name":    self.CLASSES[bi],
            "confidence":    float(prob[bi]),
            "probabilities": {self.CLASSES[i]: float(prob[i]) for i in range(self.N_CLASSES)},
            "method": "CNN+Color",
        }


class VisionSystem:
    """
    COLOR_PROFILES — true RGB values from PyBullet rgbaColor.

    Magazine pegs (RGB):
      cylinder → pink    [1,0,0.4]   → (255,0,102)
      square   → yellow  [1,0.8,0]   → (255,204,0)
      triangle → cyan    [0,0.8,1]   → (0,204,255)

    Conveyor pegs (RGB):
      orange   → cylinder [1.0,0.5,0.0] → (255,128,0)
      blue     → triangle [0.0,0.3,1.0] → (0,76,255)
      green    → square   [0.0,0.8,0.2] → (0,204,51)

    HSV: H[0-180], S[0-255], V[0-255]
    """
    COLOR_PROFILES = {
        "pink": {
            "rgb": np.array([255, 0, 102]),
            "shape": "cylinder",
            "hsv_ranges": [
                (np.array([155, 80,  80]), np.array([180, 255, 255])),
                (np.array([  0, 80,  80]), np.array([  5, 255, 255])),
            ],
        },
        "yellow": {
            "rgb": np.array([255, 204, 0]),
            "shape": "square",
            "hsv_ranges": [
                (np.array([18, 100, 80]), np.array([38, 255, 255])),
            ],
        },
        "cyan": {
            "rgb": np.array([0, 204, 255]),
            "shape": "triangle",
            "hsv_ranges": [
                (np.array([85, 80, 80]), np.array([105, 255, 255])),
            ],
        },
        "orange": {
            "rgb": np.array([255, 128, 0]),
            "shape": "cylinder",
            "hsv_ranges": [
                (np.array([5, 120, 80]), np.array([22, 255, 255])),
            ],
        },
        "blue": {
            "rgb": np.array([0, 76, 255]),
            "shape": "triangle",
            "hsv_ranges": [
                (np.array([100, 100, 60]), np.array([135, 255, 255])),
            ],
        },
        "green": {
            "rgb": np.array([0, 204, 51]),
            "shape": "square",
            "hsv_ranges": [
                (np.array([45, 100, 60]), np.array([85, 255, 255])),
            ],
        },
    }

    COLOR_DISPLAY = {
        "pink": "red",
    }

    COLOR_BGR = {
        "pink":   (102,   0, 255),
        "red":    (102,   0, 255),
        "yellow": (  0, 204, 255),
        "cyan":   (255, 204,   0),
        "orange": (  0, 128, 255),
        "blue":   (255,  76,   0),
        "green":  ( 51, 204,   0),
    }

    def __init__(self):
        self.cnn = PegCNN()
        self._detection_history = []

    # ── HOLE DETECTION ───────────────────────────────────────────────
    @staticmethod
    def detect_hole(image):
        if image is None or image.size == 0:
            return None, None
        gray     = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        ba = cv2.adaptiveThreshold(enhanced, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 12)
        _, bf = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.bitwise_or(ba, bf)
        kc = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        ko = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kc)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  ko)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        area_img = gray.shape[0] * gray.shape[1]
        best, best_cnt, best_score = None, None, 0
        for cnt in cnts:
            a = cv2.contourArea(cnt)
            if a < 400 or a > area_img * 0.6:
                continue
            per = cv2.arcLength(cnt, True)
            if per < 1:
                continue
            circ   = 4 * math.pi * a / (per * per)
            approx = cv2.approxPolyDP(cnt, 0.04 * per, True)
            score  = a * circ
            if 3 <= len(approx) <= 8 and score > best_score:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    best       = (int(round(M["m10"] / M["m00"])),
                                  int(round(M["m01"] / M["m00"])))
                    best_cnt   = approx
                    best_score = score
        return best, best_cnt

    def detect_hole_shape(self, image):
        """
        Detect hole shape in the scene image.
        Returns dict: {shape, confidence, contour, center, circularity, vertices}.
        shape in {"triangle","square","circle",None}
        """
        center, cnt = self.detect_hole(image)
        if center is None or cnt is None or len(cnt) < 3:
            return {"shape": None, "confidence": 0.0,
                    "contour": None, "center": None,
                    "circularity": 0.0, "vertices": 0}
        area = cv2.contourArea(cnt)
        per  = cv2.arcLength(cnt, True)
        if per <= 1e-6:
            return {"shape": None, "confidence": 0.0,
                    "contour": cnt, "center": center,
                    "circularity": 0.0, "vertices": 0}
        approx = cv2.approxPolyDP(cnt, 0.035 * per, True)
        vertices = len(approx)
        circ = 4 * math.pi * area / (per * per) if per > 0 else 0.0
        shape = None
        conf  = 0.0
        if vertices == 3:
            shape, conf = "triangle", 0.85
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ar = float(w) / max(h, 1)
            if 0.75 <= ar <= 1.25:
                shape, conf = "square", 0.80
        elif circ >= 0.70:
            shape, conf = "circle", min(0.90, 0.55 + circ * 0.45)
        return {
            "shape": shape, "confidence": round(conf, 3),
            "contour": cnt, "center": center,
            "circularity": round(circ, 4), "vertices": vertices
        }

    # ── DEPTH UNPROJECTION ───────────────────────────────────────────
    @staticmethod
    def pixel_to_world(u, v, depth_linear, view_matrix, proj_matrix, width, height):
        x_ndc  = 2.0 * u / width  - 1.0
        y_ndc  = 1.0 - 2.0 * v / height
        pm     = np.array(proj_matrix).reshape(4, 4).T
        zc     = -float(depth_linear)
        xc     = x_ndc * (-zc) / pm[0, 0]
        yc     = y_ndc * (-zc) / pm[1, 1]
        p_cam  = np.array([xc, yc, zc, 1.0])
        vm     = np.array(view_matrix).reshape(4, 4).T
        return (np.linalg.inv(vm) @ p_cam)[:3]

    # ── PEG COLOR DETECTION (segmentation-based) ────────────────────
    def detect_peg_color(self, rgb_image, seg_mask, peg_body_id):
        """
        FIXED:
          - PyBullet getCameraImage returns RGB — no BGR flip needed
          - HSV match is checked first, RGB distance as fallback
        """
        result = {
            "color": None, "shape": None, "mean_rgb": None,
            "confidence": 0.0, "pixel_count": 0,
            "cnn_result": None, "shape_analysis": None,
            "bbox": None, "method": "none",
        }
        if rgb_image is None or seg_mask is None:
            return result

        body_mask   = (seg_mask & ((1 << 24) - 1)) == peg_body_id
        pixel_count = int(np.sum(body_mask))
        result["pixel_count"] = pixel_count
        if pixel_count < 10:
            return result

        # PyBullet returns RGB — use directly
        peg_pixels = rgb_image[body_mask]
        mean_rgb   = np.mean(peg_pixels, axis=0)
        result["mean_rgb"] = [round(float(v), 1) for v in mean_rgb]

        ys, xs     = np.where(body_mask)
        x1 = max(0, int(xs.min()) - 4)
        y1 = max(0, int(ys.min()) - 4)
        x2 = min(rgb_image.shape[1] - 1, int(xs.max()) + 4)
        y2 = min(rgb_image.shape[0] - 1, int(ys.max()) + 4)
        result["bbox"] = (x1, y1, x2 - x1, y2 - y1)

        crop       = rgb_image[y1:y2+1, x1:x2+1].copy()
        cnn_result = self.cnn.classify(crop, mean_rgb=mean_rgb)
        result["cnn_result"] = cnn_result

        # ── RGB distance ─────────────────────────────────────────────
        best_rgb, best_dist_rgb = None, float('inf')
        for cname, prof in self.COLOR_PROFILES.items():
            d = float(np.linalg.norm(mean_rgb - prof["rgb"]))
            if d < best_dist_rgb:
                best_dist_rgb = d
                best_rgb      = cname

        # ── HSV match ────────────────────────────────────────────────
        mean_u8  = np.uint8([[mean_rgb[:3]]])
        mean_bgr = mean_u8[:, :, ::-1]
        mean_hsv = cv2.cvtColor(mean_bgr, cv2.COLOR_BGR2HSV)[0, 0]

        best_hsv, best_hsv_hits = None, -1
        for cname, prof in self.COLOR_PROFILES.items():
            hits = 0
            for (lo, hi) in prof["hsv_ranges"]:
                if (lo[0] <= mean_hsv[0] <= hi[0] and
                    lo[1] <= mean_hsv[1] <= hi[1] and
                    lo[2] <= mean_hsv[2] <= hi[2]):
                    hits += 1
            if hits > best_hsv_hits:
                best_hsv_hits = hits
                best_hsv      = cname

        # ── Fusion ───────────────────────────────────────────────────
        rgb_conf = max(0.0, min(1.0, 1.0 - best_dist_rgb / 280.0))
        if best_hsv_hits > 0 and best_hsv is not None:
            final_color = best_hsv
            confidence  = min(1.0, 0.55 + rgb_conf * 0.45)
            method      = "HSV"
        else:
            final_color = best_rgb
            confidence  = rgb_conf
            method      = "RGB-dist"

        result["shape_analysis"] = self._analyze_shape(body_mask)
        display_color = self.COLOR_DISPLAY.get(final_color, final_color)
        result["color"]      = display_color
        result["raw_color"]  = final_color
        result["shape"]      = self.COLOR_PROFILES[final_color]["shape"]
        result["confidence"] = round(confidence, 3)
        result["method"]     = method

        self._detection_history.append(
            {"color": display_color, "confidence": confidence})
        if len(self._detection_history) > 10:
            self._detection_history = self._detection_history[-10:]
        return result

    def _analyze_shape(self, binary_mask):
        m  = binary_mask.astype(np.uint8) * 255
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return {"vertices": 0, "circularity": 0, "aspect_ratio": 1, "solidity": 0}
        cnt  = max(cs, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        per  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * per if per > 0 else 1, True)
        circ   = 4 * math.pi * area / (per * per) if per > 0 else 0
        _, _, w, h = cv2.boundingRect(cnt)
        hull  = cv2.convexHull(cnt)
        solid = area / max(cv2.contourArea(hull), 1)
        return {
            "vertices":     len(approx),
            "circularity":  round(circ, 4),
            "aspect_ratio": round(float(w) / max(h, 1), 4),
            "solidity":     round(solid, 4),
            "area_px":      int(area),
        }

    # ── HSV SCENE DETECTION (no seg mask) ───────────────────────────
    @staticmethod
    def detect_peg_color_hsv(bgr_image):
        if bgr_image is None or bgr_image.size == 0:
            return []
        hsv  = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        v_ch = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(v_ch)
        hsv  = cv2.merge([h_ch, s_ch, v_ch])
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dets = []
        for cname, prof in VisionSystem.COLOR_PROFILES.items():
            combined = None
            for (lo, hi) in prof["hsv_ranges"]:
                msk = cv2.inRange(hsv, lo, hi)
                combined = msk if combined is None else cv2.bitwise_or(combined, msk)
            if combined is None:
                continue
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k)
            cnts, _  = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < 150:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx  = int(M["m10"] / M["m00"])
                cy  = int(M["m01"] / M["m00"])
                x, y, w, h2 = cv2.boundingRect(cnt)
                per  = cv2.arcLength(cnt, True)
                circ = 4 * math.pi * area / (per**2) if per > 0 else 0
                dets.append({
                    "color":       cname,
                    "shape":       prof["shape"],
                    "center_xy":   (cx, cy),
                    "bbox":        (x, y, w, h2),
                    "pixel_count": int(area),
                    "circularity": round(circ, 4),
                })
        return dets

    # ── MAGAZINE SCAN ────────────────────────────────────────────────
    def scan_magazine_colors(self, rgb_image, seg_mask, peg_body_ids):
        results = []
        for pid in peg_body_ids:
            det = self.detect_peg_color(rgb_image, seg_mask, pid)
            det["peg_id"] = pid
            results.append(det)
        return results

    # ── ANNOTATION ───────────────────────────────────────────────────
    def annotate_detections(self, image, detections):
        out = image.copy()
        for det in detections:
            if det.get("color") is None:
                continue
            bgr  = self.COLOR_BGR.get(det["color"], (255, 255, 255))
            bbox = det.get("bbox")
            if bbox and len(bbox) == 4:
                x, y, w, h2 = bbox
                cv2.rectangle(out, (x, y), (x+w, y+h2), bgr, 2)
                lbl  = f"{det['color'].upper()} {det.get('shape','?').upper()} {det.get('confidence',0):.0%}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(out, (x, y-th-8), (x+tw+8, y), bgr, -1)
                cv2.putText(out, lbl, (x+4, y-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)
        return out

    def annotate_hole(self, image, hole_center, hole_contour=None):
        out = image.copy()
        if hole_center is None:
            return out
        cx, cy = hole_center
        green  = (0, 255, 0)
        cv2.drawMarker(out, (cx, cy), green, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
        cv2.circle(out,    (cx, cy), 12, green, 1, cv2.LINE_AA)
        cv2.putText(out, "HOLE", (cx+16, cy-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, green, 1, cv2.LINE_AA)
        if hole_contour is not None:
            cv2.drawContours(out, [hole_contour], -1, green, 1, cv2.LINE_AA)
        return out

    @staticmethod
    def annotate_color_detections(image, detections, seg_mask=None):
        out  = image.copy()
        cbgr = {
            "pink":   (102,  0, 255), "red":    (102,  0, 255),
            "yellow": (  0, 204, 255),
            "cyan":   (255, 204,   0), "orange": (  0, 128, 255),
            "blue":   (255,  76,   0), "green":  ( 51, 204,   0),
        }
        for det in detections:
            if det.get("color") is None:
                continue
            bgr = cbgr.get(det["color"], (255, 255, 255))
            if det.get("bbox"):
                x, y, w, h2 = det["bbox"]
                cv2.rectangle(out, (x, y), (x+w, y+h2), bgr, 2)
                cv2.putText(out, det["color"].upper(), (x, y-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)
            elif det.get("center_xy"):
                cx, cy = det["center_xy"]
                cv2.circle(out, (cx, cy), 12, bgr, 2)
                cv2.putText(out, det["color"].upper(), (cx+15, cy+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)
        return out
