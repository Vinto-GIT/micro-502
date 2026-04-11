import numpy as np
import cv2

# =============================================================================
# PARAMÈTRES CAMÉRA
# =============================================================================
CAM_WIDTH  = 324
CAM_HEIGHT = 244
CAM_FOV_H  = np.radians(90)
CAM_FOV_V  = np.radians(90 * CAM_HEIGHT / CAM_WIDTH)
CAM_FOCAL  = (CAM_WIDTH / 2.0) / np.tan(CAM_FOV_H / 2.0)   # Focale en pixels

# Taille réelle connue du gate (hauteur de l'ouverture, fixe)
GATE_REAL_HEIGHT = 0.4   # mètres

# =============================================================================
# PARAMÈTRES FILTRE HSV
# =============================================================================
HSV_LOWER_MAG1 = np.array([140,  80,  80])
HSV_UPPER_MAG1 = np.array([180, 255, 255])
HSV_LOWER_MAG2 = np.array([  0,  80,  80])
HSV_UPPER_MAG2 = np.array([ 15, 255, 255])

MIN_CONTOUR_AREA = 300

NUM_GATES = 5
OVERFLY_ALT = 2.0          # altitude pour passer au-dessus du gate
CRUISE_ALT  = 1.5          # altitude de croisière normale


# =============================================================================
# DÉTECTION HSV + COINS
# =============================================================================
def detect_gates(image):
    detections = []

    if image is None or image.size == 0:
        return detections, None, None

    if image.ndim == 3 and image.shape[2] == 4:
        img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        img = image.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, HSV_LOWER_MAG1, HSV_UPPER_MAG1)
    mask2 = cv2.inRange(hsv, HSV_LOWER_MAG2, HSV_UPPER_MAG2)
    mask  = cv2.bitwise_or(mask1, mask2)

    kernel_open  = np.ones((5,  5),  np.uint8)
    kernel_close = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug_mask = np.zeros_like(img)
    angle_per_pixel = CAM_FOV_H / CAM_WIDTH

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        cv2.drawContours(debug_mask, [contour], -1, (255, 255, 255), 2)

        x, y, w, h = cv2.boundingRect(contour)

        M = cv2.moments(contour)
        if M["m00"] == 0:
            cx = x + w / 2.0
            cy = y + h / 2.0
        else:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

        angle_h = (cx - CAM_WIDTH / 2.0) * angle_per_pixel
        angle_v = (CAM_HEIGHT / 2.0 - cy) * (CAM_FOV_V / CAM_HEIGHT)

        MIN_BBOX_SIZE = 20
        if w < MIN_BBOX_SIZE or h < MIN_BBOX_SIZE:
            continue

        # Filtre noir à droite
        BLACK_MARGIN = 10
        right_edge = min(x + w + BLACK_MARGIN, mask.shape[1] - 1)
        has_black_right = False
        if right_edge > x + w:
            right_strip = mask[y:y+h, x+w:right_edge]
            white_ratio = np.sum(right_strip > 0) / right_strip.size
            has_black_right = (white_ratio <= 0.2)

        if not has_black_right:
            continue

        # Estimation de distance via hauteur apparente
        # D = (hauteur_réelle * focale) / hauteur_pixels
        dist_est = (GATE_REAL_HEIGHT * CAM_FOCAL) / h if h > 0 else None

        perimeter = cv2.arcLength(contour, closed=True)
        epsilon   = 0.04 * perimeter
        approx    = cv2.approxPolyDP(contour, epsilon, closed=True)
        corner_count = len(approx)
        has_4_corners = (corner_count == 4)

        if has_4_corners:
            for pt in approx:
                cv2.circle(debug_mask, tuple(pt[0]), 5, (0, 255, 255), -1)

        detections.append({
            'cx': cx, 'cy': cy,
            'x': x, 'y': y, 'w': w, 'h': h,
            'area': area,
            'angle_h': angle_h,
            'angle_v': angle_v,
            'dist_est': dist_est,
            'has_4_corners': has_4_corners,
            'corner_count':  corner_count,
            'corners': approx if has_4_corners else None,
        })

    detections.sort(key=lambda d: d['area'], reverse=True)
    return detections, debug_mask, mask


# =============================================================================
# ASSIGNMENT
# =============================================================================
class MyAssignment:
    def __init__(self):
        # États :
        # takeoff     → monter à CRUISE_ALT
        # turn_90     → tourner face au mur
        # scan_ccw    → tourner CCW jusqu'à 4 coins + noir droite
        # refine      → continuer jusqu'à noir gauche aussi → estimer position gate
        # approach    → servo visuel : centrer + avancer, accumuler centroïde
        # overfly     → monter + avancer pour passer au-dessus du gate
        # scan_clear  → tourner CW jusqu'à plus de magenta visible, puis CCW
        # lap2        → voler à travers les gates connus rapidement
        self.state = 'takeoff'
        self.debug_mask = None

        self.yaw_90_target = None

        # Suivi refine
        self.best_area    = 0.0
        self.best_angle_h = None
        self.refine_shrink_count = 0

        # Position estimée du gate courant (lissée)
        self.gate_rough_pos  = None
        self.lost_count      = 0
        self.last_good_yaw   = None

        # Stockage des gates détectés (lap 1)
        self.gate_positions = []        # liste de np.array([x, y, z])
        self.current_gate_idx = 0

    def _reset_for_next_gate(self):
        """Réinitialise les variables de tracking pour chercher le gate suivant."""
        self.gate_rough_pos = None
        self.lost_count = 0
        self.last_good_yaw = None
        self.best_area = 0.0
        self.best_angle_h = None
        self.refine_shrink_count = 0

    def compute_command(self, sensor_data, camera_data, dt):

        pos = np.array([sensor_data['x_global'],
                        sensor_data['y_global'],
                        sensor_data['z_global']])
        yaw = sensor_data['yaw']

        # =====================================================================
        # TAKEOFF
        # =====================================================================
        if self.state == 'takeoff':
            if sensor_data['z_global'] < 1.4:
                return [pos[0], pos[1], CRUISE_ALT, yaw]
            else:
                self.yaw_90_target = self._normalize_angle(yaw - np.pi/6)
                self.state = 'turn_90'
                print(f"[STATE] Décollage OK → turn_90 (cible={np.degrees(self.yaw_90_target):.1f}°)")

        # =====================================================================
        # TURN_90
        # =====================================================================
        if self.state == 'turn_90':
            error = self._angle_diff(self.yaw_90_target, yaw)
            if abs(error) < 0.05:
                self.state = 'scan_ccw'
                print("[STATE] Rotation atteinte → scan_ccw")
            return [pos[0], pos[1], CRUISE_ALT, self.yaw_90_target]

        # =====================================================================
        # SCAN_CCW : tourner CCW jusqu'à 4 coins + noir droite
        # =====================================================================
        if self.state == 'scan_ccw':
            detections, self.debug_mask, _ = detect_gates(camera_data)

            for det in detections:
                if det['has_4_corners']:
                    self.best_area    = det['area']
                    self.best_angle_h = det['angle_h']
                    self.refine_shrink_count = 0
                    self.state = 'refine'
                    print(f"[STATE] 4 coins + noir droite (aire={det['area']:.0f}) → refine")
                    break

            if self.state == 'scan_ccw':
                new_yaw = self._normalize_angle(yaw + 10 * dt)
                return [pos[0], pos[1], CRUISE_ALT, new_yaw]

        # =====================================================================
        # REFINE : tourner jusqu'à noir gauche aussi → estimer position du gate
        # =====================================================================
        if self.state == 'refine':
            detections, self.debug_mask, mask_ref = detect_gates(camera_data)

            for det in detections:
                if not det['has_4_corners']:
                    continue

                BLACK_MARGIN = 10
                x, y, w, h = det['x'], det['y'], det['w'], det['h']
                left_edge = max(x - BLACK_MARGIN, 0)

                has_black_left = False
                if mask_ref is not None and left_edge < x:
                    left_strip = mask_ref[y:y+h, left_edge:x]
                    white_ratio_left = np.sum(left_strip > 0) / left_strip.size
                    has_black_left = (white_ratio_left <= 0.2)

                if has_black_left:
                    # Gate entier visible → estimer sa position 3D
                    dist = det['dist_est'] if det['dist_est'] else 2.0
                    gate_direction = yaw - det['angle_h']   # signe corrigé
                    gate_x = pos[0] + dist * np.cos(gate_direction)
                    gate_y = pos[1] + dist * np.sin(gate_direction)
                    gate_z = pos[2] + dist * np.tan(det['angle_v'])
                    gate_z = np.clip(gate_z, 0.5, 2.5)
                    self.gate_rough_pos = np.array([gate_x, gate_y, gate_z])
                    self.lost_count = 0
                    self.last_good_yaw = yaw
                    print(f"[STATE] Gate estimé à {self.gate_rough_pos} → approach")
                    self.state = 'approach'
                    break

            if self.state == 'refine':
                new_yaw = self._normalize_angle(yaw + 10 * dt)
                return [pos[0], pos[1], CRUISE_ALT, new_yaw]

        # =====================================================================
        # APPROACH : servo visuel continu
        #   - Centrer le gate + avancer avec mise à jour continue du centroïde
        #   - Quand assez proche → enregistrer la position et passer à overfly
        # =====================================================================
        if self.state == 'approach':
            detections, self.debug_mask, _ = detect_gates(camera_data)

            det = detections[0] if detections else None

            if det is not None:
                # ---- Gate visible : reset compteur de perte ----
                self.lost_count = 0
                self.last_good_yaw = yaw

                # Mettre à jour la position estimée du gate en continu
                dist = det['dist_est'] if det['dist_est'] else 2.0
                gate_direction = yaw - det['angle_h']
                gate_x = pos[0] + dist * np.cos(gate_direction)
                gate_y = pos[1] + dist * np.sin(gate_direction)
                gate_z = pos[2] + dist * np.tan(det['angle_v'])
                gate_z = np.clip(gate_z, 0.5, 2.5)
                # Lissage exponentiel pour éviter les sauts
                alpha = 0.3
                if self.gate_rough_pos is not None:
                    self.gate_rough_pos = alpha * np.array([gate_x, gate_y, gate_z]) \
                                        + (1 - alpha) * self.gate_rough_pos
                else:
                    self.gate_rough_pos = np.array([gate_x, gate_y, gate_z])

                # -- Assez proche → enregistrer le gate et passer au-dessus --
                CLOSE_THRESHOLD_H = 80
                if det['h'] > CLOSE_THRESHOLD_H or (det['dist_est'] and det['dist_est'] < 0.6):
                    self.gate_positions.append(self.gate_rough_pos.copy())
                    self.current_gate_idx = len(self.gate_positions)
                    # Sauvegarder la direction d'approche + calculer le point de sortie
                    approach_dir = yaw - det['angle_h']
                    overshoot = 0.8
                    self.overfly_target = np.array([
                        self.gate_rough_pos[0] + overshoot * np.cos(approach_dir),
                        self.gate_rough_pos[1] + overshoot * np.sin(approach_dir),
                        OVERFLY_ALT
                    ])
                    self.overfly_yaw = approach_dir
                    print(f"[STATE] Gate {len(self.gate_positions)}/{NUM_GATES} "
                          f"enregistré à {self.gate_rough_pos} → overfly")
                    self.state = 'overfly'
                    return [pos[0], pos[1], float(self.gate_rough_pos[2]), yaw]

                # -- Sinon : corriger le yaw pour centrer le gate + avancer --
                target_yaw = self._normalize_angle(yaw - det['angle_h'])
                APPROACH_SPEED = 0.15
                step_dir = yaw - det['angle_h']
                target_x = pos[0] + APPROACH_SPEED * np.cos(step_dir)
                target_y = pos[1] + APPROACH_SPEED * np.sin(step_dir)
                target_z = float(self.gate_rough_pos[2])
                return [target_x, target_y, target_z, target_yaw]

            else:
                # ---- Gate pas visible ----
                self.lost_count += 1
                MAX_LOST_FRAMES = 50

                if self.lost_count > MAX_LOST_FRAMES:
                    print(f"[STATE] Gate perdu ({self.lost_count} frames) → scan_ccw")
                    self.state = 'scan_ccw'
                    self.lost_count = 0
                    return [pos[0], pos[1], CRUISE_ALT, yaw]

                # Garder le cap et avancer doucement vers la dernière position
                if self.gate_rough_pos is not None:
                    dx = self.gate_rough_pos[0] - pos[0]
                    dy = self.gate_rough_pos[1] - pos[1]
                    dist = np.linalg.norm([dx, dy])
                    if dist > 0.1:
                        SLOW_SPEED = 0.08
                        step = min(SLOW_SPEED, dist)
                        target_x = pos[0] + step * (dx / dist)
                        target_y = pos[1] + step * (dy / dist)
                        target_yaw = np.arctan2(dy, dx)
                        return [target_x, target_y, float(self.gate_rough_pos[2]), target_yaw]

                return [pos[0], pos[1], CRUISE_ALT, yaw]

        # =====================================================================
        # OVERFLY : monter puis voler vers le point de sortie pré-calculé
        # =====================================================================
        if self.state == 'overfly':
            # Phase 1 : monter
            if pos[2] < OVERFLY_ALT - 0.1:
                return [pos[0], pos[1], OVERFLY_ALT, self.overfly_yaw]

            # Phase 2 : voler vers le point de sortie
            dx = self.overfly_target[0] - pos[0]
            dy = self.overfly_target[1] - pos[1]
            dist = np.linalg.norm([dx, dy])

            if dist < 0.3:
                self._reset_for_next_gate()
                if len(self.gate_positions) >= NUM_GATES:
                    print(f"[STATE] {NUM_GATES} gates trouvés ! → lap2")
                    self.state = 'lap2'
                else:
                    print(f"[STATE] Gate dépassé → scan_clear")
                    self.state = 'scan_clear'
                return [pos[0], pos[1], OVERFLY_ALT, self.overfly_yaw]

            OVERFLY_SPEED = 0.2
            step = min(OVERFLY_SPEED, dist)
            move_x = pos[0] + step * (dx / dist)
            move_y = pos[1] + step * (dy / dist)
            return [move_x, move_y, OVERFLY_ALT, self.overfly_yaw]

        # =====================================================================
        # SCAN_CLEAR : tourner CW jusqu'à ne plus voir de magenta
        #              (on dépasse visuellement le gate courant)
        #              puis repasser en scan_ccw pour trouver le suivant
        # =====================================================================
        if self.state == 'scan_clear':
            # D'abord redescendre à l'altitude de croisière
            if pos[2] > CRUISE_ALT + 0.1:
                return [pos[0], pos[1], CRUISE_ALT, yaw]

            detections, self.debug_mask, _ = detect_gates(camera_data)

            if detections:
                # On voit encore du magenta → tourner CW pour s'en éloigner
                new_yaw = self._normalize_angle(yaw - 15 * dt)
                return [pos[0], pos[1], CRUISE_ALT, new_yaw]
            else:
                # Plus de magenta → le gate courant est derrière nous
                print("[STATE] Champ clair → scan_ccw (gate suivant)")
                self.state = 'scan_ccw'
                return [pos[0], pos[1], CRUISE_ALT, yaw]

        # =====================================================================
        # LAP2 : voler à travers les gates connus dans l'ordre CCW
        # (TODO : à implémenter pour les laps 2 et 3)
        # =====================================================================
        if self.state == 'lap2':
            print(f"[LAP2] Gates connus : {self.gate_positions}")
            self.state = 'finished'
            return [pos[0], pos[1], CRUISE_ALT, yaw]

        # =====================================================================
        # FINISHED
        # =====================================================================
        if self.state == 'finished':
            return [pos[0], pos[1], CRUISE_ALT, yaw]

        return [pos[0], pos[1], CRUISE_ALT, yaw]

    def _normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _angle_diff(self, target, current):
        return self._normalize_angle(target - current)


# =============================================================================
# Interface avec main.py
# =============================================================================
_controller = MyAssignment()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)

def get_debug_mask():
    return _controller.debug_mask