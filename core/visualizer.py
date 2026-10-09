import math
import time
import cv2
import numpy as np

# Color definitions (BGR)
COLOR_NEON_GREEN = (50, 255, 120)
COLOR_NEON_CYAN = (255, 230, 40)
COLOR_NEON_ORANGE = (30, 140, 255)
COLOR_NEON_PURPLE = (230, 80, 200)
COLOR_NEON_YELLOW = (40, 230, 255)
COLOR_ALERT_RED = (60, 60, 255)

# Color map for 80 COCO classes
np.random.seed(42)
CLASS_COLORS = [tuple(int(c) for c in np.random.randint(60, 240, size=3)) for _ in range(100)]

# Pose Limb Color Schemes (BGR)
LIMB_COLORS = {
    "head": (255, 200, 100),       # light cyan
    "torso": (100, 255, 180),      # bright green
    "left_arm": (255, 160, 50),    # sky blue
    "right_arm": (50, 160, 255),   # orange
    "left_leg": (255, 100, 200),   # purple
    "right_leg": (50, 230, 255)    # amber yellow
}

KEYPOINT_TO_LIMB = [
    ("head", (0, 1)), ("head", (0, 2)), ("head", (1, 3)), ("head", (2, 4)),
    ("torso", (5, 6)), ("torso", (5, 11)), ("torso", (6, 12)), ("torso", (11, 12)),
    ("left_arm", (5, 7)), ("left_arm", (7, 9)),
    ("right_arm", (6, 8)), ("right_arm", (8, 10)),
    ("left_leg", (11, 13)), ("left_leg", (13, 15)),
    ("right_leg", (12, 14)), ("right_leg", (14, 16))
]


class VisionVisualizer:
    def __init__(self):
        self.show_boxes = True
        self.show_labels = True
        self.show_conf = True
        self.show_skeleton = True
        self.show_keypoints = True
        self.show_drivable = True
        self.show_lanes = True
        self.show_hud = True
        # Sci-Fi Features
        self.show_tactical_lock = True   # 方案 A: 战术火控瞄准锁定与威胁研判
        self.show_cyber_profile = True   # 方案 A: 赛博黑客全息身份档案
        self.show_bev_radar = True       # 方案 B: 上帝视角 3D 俯视雷达小地图
        self.kpt_conf_threshold = 0.20   # 骨骼关节置信度阈值 (远距离小目标增强降至 0.20)

    def draw_corner_rect(self, img, pt1, pt2, color, thickness=2, corner_len=18):
        """Draws aesthetic corner bracket bounding box for sci-fi/modern look."""
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])

        # Main subtle rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        # Ensure corner length isn't larger than half the box
        cl_x = min(corner_len, abs(x2 - x1) // 3)
        cl_y = min(corner_len, abs(y2 - y1) // 3)

        # Top-Left
        cv2.line(img, (x1, y1), (x1 + cl_x, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + cl_y), color, thickness, cv2.LINE_AA)

        # Top-Right
        cv2.line(img, (x2, y1), (x2 - cl_x, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + cl_y), color, thickness, cv2.LINE_AA)

        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + cl_x, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - cl_y), color, thickness, cv2.LINE_AA)

        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - cl_x, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - cl_y), color, thickness, cv2.LINE_AA)

    def draw_tag(self, img, text, pos, bg_color, text_color=(255, 255, 255), scale=0.5, thickness=1):
        """Draws rounded tag background with crisp text."""
        x, y = int(pos[0]), int(pos[1])
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        
        # Clamp coordinates inside image
        h, w = img.shape[:2]
        y_top = max(0, y - th - 8)
        y_bottom = y_top + th + 8
        x_left = max(0, x)
        x_right = min(w - 1, x_left + tw + 10)

        # Background badge
        cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom), bg_color, -1)
        cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom), (255, 255, 255), 1, cv2.LINE_AA)

        # Text
        cv2.putText(img, text, (x_left + 5, y_bottom - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness, cv2.LINE_AA)

    def draw_tactical_lock(self, img, x1, y1, x2, y2, label="TARGET", conf=0.9, anim_angle=45.0, is_primary=True):
        """Draws Terminator / Fighter Jet style rotating fire-control reticle with threat level."""
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        bw = x2 - x1
        bh = y2 - y1

        # Distance estimation based on box height (normalized camera perspective)
        dist = round(max(2.5, min(75.0, 1600.0 / max(bh, 1))), 1)
        if dist < 6.0:
            threat_str, threat_col = "CRITICAL", (40, 40, 255)    # Alert Red
        elif dist < 15.0:
            threat_str, threat_col = "ELEVATED", (30, 160, 255)   # Amber Orange
        else:
            threat_str, threat_col = "NOMINAL", (255, 230, 40)    # Electric Cyan

        # 1. Tech Corner Brackets
        cl = max(10, min(24, min(bw, bh) // 4))
        th = 2
        # TL
        cv2.line(img, (x1, y1), (x1 + cl, y1), threat_col, th, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + cl), threat_col, th, cv2.LINE_AA)
        # TR
        cv2.line(img, (x2, y1), (x2 - cl, y1), threat_col, th, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + cl), threat_col, th, cv2.LINE_AA)
        # BL
        cv2.line(img, (x1, y2), (x1 + cl, y2), threat_col, th, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - cl), threat_col, th, cv2.LINE_AA)
        # BR
        cv2.line(img, (x2, y2), (x2 - cl, y2), threat_col, th, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - cl), threat_col, th, cv2.LINE_AA)

        # Distant/secondary objects: sleek compact label without heavy badges
        if not is_primary or bh < 26:
            cv2.putText(img, f"{label[:4].upper()} {dist}m", (x1 + 3, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, threat_col, 1, cv2.LINE_AA)
            return

        # 2. Animated Rotating Crosshair Reticle at center
        r = max(16, min(36, min(bw, bh) // 4))
        for seg in range(3):
            start_a = anim_angle + seg * 120
            end_a = start_a + 60
            cv2.ellipse(img, (cx, cy), (r, r), 0, start_a, end_a, threat_col, 2, cv2.LINE_AA)

        # Center crosshair tick
        cr = max(6, r // 2)
        cv2.line(img, (cx - cr, cy), (cx - 4, cy), threat_col, 1, cv2.LINE_AA)
        cv2.line(img, (cx + 4, cy), (cx + cr, cy), threat_col, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy - cr), (cx, cy - 4), threat_col, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy + 4), (cx, cy + cr), threat_col, 1, cv2.LINE_AA)

        # 3. Tactical Readout Badges
        txt_top = f"LOCK // {label.upper()} {int(conf*100)}%"
        txt_sub = f"DIST: {dist}m | THREAT: {threat_str}"

        # Top badge
        cv2.rectangle(img, (x1, max(0, y1 - 22)), (x1 + 175, y1), (15, 20, 30), -1)
        cv2.rectangle(img, (x1, max(0, y1 - 22)), (x1 + 175, y1), threat_col, 1)
        cv2.putText(img, txt_top, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Bottom info tag
        cv2.rectangle(img, (x1, y2), (x1 + 200, y2 + 18), (15, 20, 30), -1)
        cv2.rectangle(img, (x1, y2), (x1 + 200, y2 + 18), (60, 80, 100), 1)
        cv2.putText(img, txt_sub, (x1 + 6, y2 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, threat_col, 1, cv2.LINE_AA)

    def draw_cyber_profile(self, img, x1, y1, x2, y2, person_id=1, anim_t=0.0):
        """Draws Watch Dogs / Cyberpunk 2077 style floating holographic citizen data card."""
        h_img, w_img = img.shape[:2]
        ax = min(w_img - 1, x2)
        ay = max(30, min(h_img - 1, y1 + int((y2 - y1) * 0.2)))

        # Card dimensions
        card_w = 210
        card_h = 76
        flip_left = (ax + card_w + 60 >= w_img)

        if not flip_left:
            p1 = (ax, ay)
            p2 = (ax + 28, ay - 20)
            p3 = (ax + 55, ay - 20)
            cw_x = p3[0]
            cw_y = p3[1] - 15
        else:
            p1 = (max(0, x1), ay)
            p2 = (max(0, x1 - 28), ay - 20)
            p3 = (max(0, x1 - 55), ay - 20)
            cw_x = max(10, p3[0] - card_w)
            cw_y = p3[1] - 15

        # Clamp vertical
        cw_y = max(10, min(h_img - card_h - 10, cw_y))

        # Connecting tech polyline
        cv2.line(img, p1, p2, COLOR_NEON_CYAN, 1, cv2.LINE_AA)
        cv2.line(img, p2, p3, COLOR_NEON_CYAN, 1, cv2.LINE_AA)
        cv2.circle(img, p1, 3, COLOR_NEON_CYAN, -1, cv2.LINE_AA)

        # Translucent card background
        overlay = img.copy()
        cv2.rectangle(overlay, (cw_x, cw_y), (cw_x + card_w, cw_y + card_h), (12, 18, 28), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        # Card borders with tech header line
        cv2.rectangle(img, (cw_x, cw_y), (cw_x + card_w, cw_y + card_h), (255, 200, 40), 1, cv2.LINE_AA)
        cv2.line(img, (cw_x, cw_y + 18), (cw_x + card_w, cw_y + card_h - 58), (60, 100, 140), 1, cv2.LINE_AA)

        # Content generator
        roles = ["NETRUNNER", "SYS-ADMIN", "SPECIALIST", "TACTICAL AGENT", "CIVILIAN"]
        role = roles[person_id % len(roles)]
        id_tag = f"#NET-{(person_id * 1973 + 1042) % 9000 + 1000}"
        bpm = 70 + (person_id * 7) % 25

        # Header
        cv2.putText(img, f"CITIZEN {id_tag}", (cw_x + 8, cw_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_NEON_CYAN, 1, cv2.LINE_AA)
        # Role
        cv2.putText(img, f"ROLE: {role}", (cw_x + 8, cw_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 240, 255), 1, cv2.LINE_AA)
        # Bio-pulse with simulated ECG pulse wave
        cv2.putText(img, f"BIO-PULSE: {bpm} BPM", (cw_x + 8, cw_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_NEON_GREEN, 1, cv2.LINE_AA)
        for i in range(12):
            x_pt = cw_x + 130 + i * 5
            pulse = math.sin((anim_t * 6.0) + i * 0.8) * 5 if (i in (4, 5, 6, 7)) else 0
            y_pt = int(cw_y + 48 - pulse)
            cv2.circle(img, (x_pt, y_pt), 1, COLOR_NEON_GREEN, -1)

        cv2.putText(img, "STATUS: VERIFIED // WEAPON: NONE", (cw_x + 8, cw_y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 180, 200), 1, cv2.LINE_AA)

    def draw_bev_radar(self, img, detections, poses=None, sweep_angle=45.0):
        """Draws top-down 3D Bird's Eye View (BEV) tactical radar minimap in bottom-right corner."""
        h, w = img.shape[:2]
        R = min(95, min(w, h) // 4)
        if R < 40:
            return
        cx = w - R - 20
        cy = h - R - 25

        # 1. Dark radar disc background
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), R, (10, 15, 22), -1)
        cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)

        # 2. Concentric Range Rings
        for ring_r in (int(R * 0.33), int(R * 0.66), R):
            cv2.circle(img, (cx, cy), ring_r, (40, 70, 60), 1, cv2.LINE_AA)

        # 3. Crosshair grid & Forward 60-deg FOV cone
        cv2.line(img, (cx - R, cy), (cx + R, cy), (30, 60, 50), 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy - R), (cx, cy + R), (30, 60, 50), 1, cv2.LINE_AA)
        for fov_deg in (-30, 30):
            rad = math.radians(fov_deg - 90)
            fx = int(cx + R * math.cos(rad))
            fy = int(cy + R * math.sin(rad))
            cv2.line(img, (cx, cy), (fx, fy), (50, 90, 80), 1, cv2.LINE_AA)

        # 4. Rotating Sweep Beam + trailing soft wedge
        rad = math.radians(sweep_angle)
        sx = int(cx + R * math.cos(rad))
        sy = int(cy + R * math.sin(rad))
        cv2.line(img, (cx, cy), (sx, sy), (80, 255, 140), 2, cv2.LINE_AA)
        for trail in range(1, 16):
            trad = math.radians(sweep_angle - trail * 2)
            tx = int(cx + R * math.cos(trad))
            ty = int(cy + R * math.sin(trad))
            col_alpha = max(10, 140 - trail * 9)
            cv2.line(img, (cx, cy), (tx, ty), (0, col_alpha, int(col_alpha * 0.6)), 1, cv2.LINE_AA)

        # 5. Ego Vehicle Icon (Cyan Arrow pointing Up)
        ego_pts = np.array([
            [cx, cy - 8],
            [cx - 5, cy + 6],
            [cx, cy + 3],
            [cx + 5, cy + 6]
        ], np.int32)
        cv2.fillPoly(img, [ego_pts], COLOR_NEON_CYAN)

        # 6. Map Target Detections onto Radar
        horizon = h * 0.45
        for det in detections:
            box = det["box"]
            bx = (box[0] + box[2]) / 2.0
            by = box[3]
            rel_x = (bx - (w / 2.0)) / (w / 2.0)
            rel_y = max(0.05, min(1.0, (by - horizon) / max(1.0, (h - horizon))))
            dist_norm = 1.0 - rel_y
            radar_dist = 12 + dist_norm * (R - 16)
            angle_rad = math.radians(-90 + rel_x * 40)
            blip_x = int(cx + radar_dist * math.cos(angle_rad))
            blip_y = int(cy + radar_dist * math.sin(angle_rad))

            if (blip_x - cx)**2 + (blip_y - cy)**2 <= (R - 2)**2:
                # Vehicle: Amber dot with white center
                cv2.circle(img, (blip_x, blip_y), 4, (40, 220, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (blip_x, blip_y), 2, (255, 255, 255), -1, cv2.LINE_AA)

        if poses:
            for pose in poses:
                box = pose.get("box")
                if not box:
                    continue
                bx = (box[0] + box[2]) / 2.0
                by = box[3]
                rel_x = (bx - (w / 2.0)) / (w / 2.0)
                rel_y = max(0.05, min(1.0, (by - horizon) / max(1.0, (h - horizon))))
                dist_norm = 1.0 - rel_y
                radar_dist = 12 + dist_norm * (R - 16)
                angle_rad = math.radians(-90 + rel_x * 40)
                blip_x = int(cx + radar_dist * math.cos(angle_rad))
                blip_y = int(cy + radar_dist * math.sin(angle_rad))

                if (blip_x - cx)**2 + (blip_y - cy)**2 <= (R - 2)**2:
                    # Person: Purple dot
                    cv2.circle(img, (blip_x, blip_y), 4, COLOR_NEON_PURPLE, -1, cv2.LINE_AA)
                    cv2.circle(img, (blip_x, blip_y), 2, (255, 255, 255), -1, cv2.LINE_AA)

        # 7. Radar Border & Header
        cv2.circle(img, (cx, cy), R, COLOR_NEON_GREEN, 2, cv2.LINE_AA)
        cv2.rectangle(img, (cx - 75, cy - R - 18), (cx + 75, cy - R), (10, 15, 24), -1)
        cv2.rectangle(img, (cx - 75, cy - R - 18), (cx + 75, cy - R), COLOR_NEON_GREEN, 1)
        cv2.putText(img, "BEV TACTICAL RADAR", (cx - 70, cy - R - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_NEON_GREEN, 1, cv2.LINE_AA)

    def render(self, frame_bgr, results, fps=0.0, hw_info=None):
        """
        Renders bounding boxes, human skeletons, keypoints, and HUD stats onto frame.
        """
        img = frame_bgr.copy()
        h, w = img.shape[:2]

        now = time.perf_counter()
        anim_angle = (now * 140.0) % 360
        sweep_angle = (now * 160.0) % 360

        detections = results.get("detections", [])
        poses = results.get("poses", [])
        stats = results.get("stats", {})
        latency = results.get("latency_ms", 0.0)

        # 0. Render Road Surface & Lane Markings (YOLOPv2)
        da_mask = results.get("da_mask")
        ll_mask = results.get("ll_mask")

        if self.show_drivable and da_mask is not None:
            mask_da = (da_mask == 1)
            if np.any(mask_da):
                color_da = np.array([30, 220, 100], dtype=np.uint8)  # Neon emerald green (BGR)
                img[mask_da] = (img[mask_da].astype(np.float32) * 0.65 + color_da * 0.35).astype(np.uint8)

        if self.show_lanes and ll_mask is not None:
            mask_ll = (ll_mask == 1)
            if np.any(mask_ll):
                color_ll = np.array([40, 50, 255], dtype=np.uint8)   # Laser crimson red (BGR)
                img[mask_ll] = (img[mask_ll].astype(np.float32) * 0.25 + color_ll * 0.75).astype(np.uint8)

        # 1. Render Object Detections (Tactical Lock vs Classic Corner Boxes)
        primary_indices = set()
        if detections:
            sorted_dets = sorted(
                enumerate(detections),
                key=lambda x: (x[1]["box"][2] - x[1]["box"][0]) * (x[1]["box"][3] - x[1]["box"][1]),
                reverse=True
            )
            primary_indices = set(idx for idx, _ in sorted_dets[:6])

        for i, det in enumerate(detections):
            box = det["box"]
            conf = det["conf"]
            cls_id = det["class_id"]
            cls_name = det["class_name"]
            is_veh = det.get("is_vehicle", False)
            is_person = (cls_id == 0 or cls_name.lower() == "person")

            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            if self.show_tactical_lock:
                self.draw_tactical_lock(
                    img, x1, y1, x2, y2,
                    label=cls_name, conf=conf,
                    anim_angle=anim_angle + i * 40,
                    is_primary=(i in primary_indices)
                )
            elif self.show_boxes:
                color = COLOR_NEON_YELLOW if is_veh else CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                self.draw_corner_rect(img, (x1, y1), (x2, y2), color, thickness=2)
                if self.show_labels:
                    conf_str = f" {int(conf * 100)}%" if self.show_conf else ""
                    prefix = "[VEHICLE] " if is_veh else ""
                    tag_text = f"{prefix}{cls_name.upper()}{conf_str}"
                    self.draw_tag(img, tag_text, (x1, y1), color)

            # If person detected and cyber profile enabled, draw floating holographic data card
            if self.show_cyber_profile and is_person:
                self.draw_cyber_profile(img, x1, y1, x2, y2, person_id=i, anim_t=now)

        # 2. Render Human Poses (Skeleton + Keypoints + Cyber Profile)
        for i, pose in enumerate(poses):
            box = pose.get("box")
            conf = pose.get("conf", 0.0)
            kpts = pose.get("keypoints")

            # Draw person box or tactical lock
            if box is not None:
                px1, py1, px2, py2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                if self.show_tactical_lock:
                    self.draw_tactical_lock(img, px1, py1, px2, py2, label="PERSON", conf=conf, anim_angle=anim_angle + i * 60)
                elif self.show_boxes:
                    self.draw_corner_rect(img, (px1, py1), (px2, py2), COLOR_NEON_PURPLE, thickness=2)
                    if self.show_labels:
                        conf_str = f" {int(conf * 100)}%" if self.show_conf else ""
                        self.draw_tag(img, f"PERSON{conf_str}", (px1, py1), COLOR_NEON_PURPLE)

                if self.show_cyber_profile:
                    self.draw_cyber_profile(img, px1, py1, px2, py2, person_id=i, anim_t=now)

            # Draw Limbs (Bones)
            if self.show_skeleton and kpts is not None and len(kpts) == 17:
                for limb_name, (idx1, idx2) in KEYPOINT_TO_LIMB:
                    pt1 = kpts[idx1]
                    pt2 = kpts[idx2]

                    conf1 = pt1[2] if len(pt1) > 2 else 1.0
                    conf2 = pt2[2] if len(pt2) > 2 else 1.0

                    if conf1 >= self.kpt_conf_threshold and conf2 >= self.kpt_conf_threshold:
                        p1 = (int(pt1[0]), int(pt1[1]))
                        p2 = (int(pt2[0]), int(pt2[1]))
                        bone_color = LIMB_COLORS.get(limb_name, (200, 200, 200))
                        cv2.line(img, p1, p2, bone_color, 3, cv2.LINE_AA)

            # Draw Keypoint Joints
            if self.show_keypoints and kpts is not None:
                for kp in kpts:
                    kp_conf = kp[2] if len(kp) > 2 else 1.0
                    if kp_conf >= self.kpt_conf_threshold:
                        cx, cy = int(kp[0]), int(kp[1])
                        cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(img, (cx, cy), 3, COLOR_NEON_CYAN, -1, cv2.LINE_AA)

        # 3. Top-Down 3D BEV Tactical Radar Minimap
        if self.show_bev_radar:
            self.draw_bev_radar(img, detections, poses, sweep_angle=sweep_angle)

        # 4. Render High-Tech HUD Overlay
        if self.show_hud:
            is_road = (da_mask is not None or ll_mask is not None)
            self._render_hud(img, fps, latency, stats, hw_info, is_road=is_road)

        return img

    def _render_hud(self, img, fps, latency_ms, stats, hw_info, is_road=False):
        """Draws top HUD status banner with real-time stats."""
        overlay = img.copy()
        hud_h = 75
        cv2.rectangle(overlay, (0, 0), (img.shape[1], hud_h), (15, 18, 24), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        cv2.line(img, (0, hud_h), (img.shape[1], hud_h), (50, 60, 80), 1, cv2.LINE_AA)

        # FPS badge
        fps_color = COLOR_NEON_GREEN if fps >= 30 else COLOR_ALERT_RED
        fps_str = f"FPS: {fps:.1f}"
        cv2.putText(img, fps_str, (16, 28), cv2.FONT_HERSHEY_DUPLEX, 0.75, fps_color, 2, cv2.LINE_AA)

        # Latency
        lat_str = f"Inference: {latency_ms:.1f}ms"
        cv2.putText(img, lat_str, (160, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        # CUDA / Tensor Core Status
        if hw_info and hw_info.get("cuda_available"):
            dev_str = f"Device: {hw_info.get('device_name', 'CUDA GPU')}"
            cv2.putText(img, dev_str, (360, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA)

            if hw_info.get("tensor_core_active"):
                tc_str = "[CUDA + TensorCore FP16 Active]"
                tc_color = COLOR_NEON_GREEN
            else:
                tc_str = "[CUDA FP32 Active]"
                tc_color = COLOR_NEON_ORANGE
            cv2.putText(img, tc_str, (img.shape[1] - 320, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tc_color, 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "[CPU Mode]", (360, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 150, 255), 1, cv2.LINE_AA)

        # Counts Bar on line 2
        if is_road:
            v_cnt = stats.get("vehicle_count", 0)
            stats_str = f"Tesla-Style Vision: Drivable Road [ONLINE]   |   Lane Lines [TRACKED]   |   Vehicles: {v_cnt}"
        else:
            p_cnt = stats.get("person_count", 0)
            v_cnt = stats.get("vehicle_count", 0)
            tot_cnt = stats.get("total_objects", 0)
            stats_str = f"Total Detections: {tot_cnt}   |   Persons (Skeleton): {p_cnt}   |   Vehicles: {v_cnt}"

        cv2.putText(img, stats_str, (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
