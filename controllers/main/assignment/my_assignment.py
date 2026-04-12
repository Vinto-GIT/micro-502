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
GATE_REAL_WIDTH  = 0.4   # largeur assumée (gates carrés selon les instructions)

# Matrice intrinsèque caméra (pas de distorsion en simulation)
CAM_K = np.array([[CAM_FOCAL, 0.0,       CAM_WIDTH  / 2.0],
                  [0.0,       CAM_FOCAL, CAM_HEIGHT / 2.0],
                  [0.0,       0.0,       1.0             ]], dtype=np.float32)

# Points 3D du gate dans son repère propre (centre à l'origine, plan Z=0)
# Ordre TL → TR → BR → BL  (Y croissant vers le bas en coords image)
_hw = GATE_REAL_WIDTH  / 2.0
_hh = GATE_REAL_HEIGHT / 2.0
GATE_OBJ_POINTS = np.array([[-_hw, -_hh, 0.0],   # TL
                             [ _hw, -_hh, 0.0],   # TR
                             [ _hw,  _hh, 0.0],   # BR
                             [-_hw,  _hh, 0.0]],  # BL
                            dtype=np.float32)

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
TAKEOFF_POS = np.array([1.4, 0.1])   # position fixe du pad de décollage


# =============================================================================
# HELPER : tri des 4 coins en ordre TL → TR → BR → BL
# =============================================================================
def order_corners_2d(pts):
    """
    Trie 4 coins 2D dans l'ordre TL → TR → BR → BL
    (coordonnées image : y croissant vers le bas).
    Utilise la somme x+y (min=TL, max=BR) et la différence y-x (min=TR, max=BL).
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)          # x+y  :  min → TL,  max → BR
    d = pts[:, 1] - pts[:, 0]   # y-x  :  min → TR,  max → BL
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]   # TL
    ordered[1] = pts[np.argmin(d)]   # TR
    ordered[2] = pts[np.argmax(s)]   # BR
    ordered[3] = pts[np.argmax(d)]   # BL
    return ordered


# =============================================================================
# DÉTECTION HSV + COINS + solvePnP
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

        # --- solvePnP + dessin des coins ordonnés ---
        pnp_dist    = None
        pnp_tvec    = None
        pnp_angle_h = None
        pnp_angle_v = None

        if has_4_corners:
            ordered = order_corners_2d(approx.reshape(4, 2))

            # Couleurs par rôle : TL=cyan, TR=orange, BR=rouge, BL=magenta
            corner_colors = [(0, 255, 255), (0, 165, 255), (0, 0, 255), (255, 0, 255)]
            for k, pt in enumerate(ordered):
                cv2.circle(debug_mask, tuple(pt.astype(int)), 6, corner_colors[k], -1)

            # solvePnP — IPPE : solution fermée optimale pour cibles planaires
            success, rvec, tvec = cv2.solvePnP(
                GATE_OBJ_POINTS, ordered, CAM_K, None,
                flags=cv2.SOLVEPNP_IPPE
            )
            if success:
                t = tvec.flatten()
                pnp_dist    = float(np.linalg.norm(t))
                pnp_tvec    = t
                # t[0]=droite, t[1]=bas, t[2]=profondeur  (conv. OpenCV)
                # angle horizontal (+ = gate à droite de l'axe caméra)
                pnp_angle_h = float(np.arctan2(t[0], t[2]))
                # angle vertical   (+ = gate au-dessus de l'horizon)
                pnp_angle_v = float(np.arctan2(-t[1], t[2]))

        # Dessiner le centroïde (point vert)
        cv2.circle(debug_mask, (int(cx), int(cy)), 5, (0, 255, 0), -1)

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
            # --- nouvelles valeurs issues de solvePnP ---
            'pnp_dist':    pnp_dist,
            'pnp_tvec':    pnp_tvec,
            'pnp_angle_h': pnp_angle_h,
            'pnp_angle_v': pnp_angle_v,
        })

    detections.sort(key=lambda d: d['area'], reverse=True)
    return detections, debug_mask, mask


# =============================================================================
# ASSIGNMENT
# =============================================================================
class MyAssignment:
    def __init__(self):
        # États :
        # takeoff      → monter à CRUISE_ALT
        # turn_90      → tourner face au mur
        # scan_ccw     → tourner CCW jusqu'à 4 coins + noir droite
        # refine       → continuer jusqu'à noir gauche aussi → estimer position gate
        # approach     → servo visuel : centrer + avancer, accumuler centroïde
        # fly_through  → avancer à hauteur du gate pour passer dedans + point de sortie
        # scan_clear   → tourner CW jusqu'à plus de magenta visible, puis CCW
        # lap2         → voler à travers les gates connus rapidement
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

        # Variables pour les laps rapides (lap 2 et 3)
        self.lap2_gate_idx  = 0   # indice du gate courant dans l'ordre CCW
        self.lap2_lap_count = 0   # nombre de laps rapides effectués (max 2)

    def _reset_for_next_gate(self):
        """Réinitialise les variables de tracking pour chercher le gate suivant."""
        self.gate_rough_pos = None
        self.lost_count = 0
        self.last_good_yaw = None
        self.best_area = 0.0
        self.best_angle_h = None
        self.refine_shrink_count = 0

    def _reorder_gates_ccw(self, gate_positions):
        """
        Réordonne les gates dans le sens anti-horaire (CCW) en partant
        du premier gate rencontré en tournant CCW depuis TAKEOFF_POS.

        Principe :
          1. Centre de l'arène = barycentre des gates détectés
          2. Angle de chaque gate depuis ce centre
          3. Angle relatif à la direction décollage→centre (= angle 0)
          4. Tri croissant de cet angle relatif → ordre CCW depuis le décollage
        """
        if len(gate_positions) < 2:
            return gate_positions

        center = np.mean([g[:2] for g in gate_positions], axis=0)

        # Angle du pad de décollage vu depuis le centre
        takeoff_angle = np.arctan2(TAKEOFF_POS[1] - center[1],
                                   TAKEOFF_POS[0] - center[0])

        # Pour chaque gate : angle relatif au décollage, modulo 2π (→ CCW)
        def ccw_key(gp):
            a = np.arctan2(gp[1] - center[1], gp[0] - center[0])
            return (a - takeoff_angle) % (2 * np.pi)

        ordered = sorted(gate_positions, key=ccw_key)
        print(f"[CCW] Centre arène estimé : {center}")
        for i, gp in enumerate(ordered):
            print(f"[CCW] Gate {i+1} → {gp}")
        return ordered

    def _closest_detection_to_target(self, detections, pos, yaw):
        """
        Parmi toutes les détections, retourne celle dont la position estimée
        est la plus proche de self.gate_rough_pos.
        Si gate_rough_pos est None, retourne simplement la plus grande détection.
        Rejette les détections à plus de 1.5m de la cible.
        Préfère les valeurs PnP (pnp_dist / pnp_angle_h) quand disponibles.
        """
        if self.gate_rough_pos is None:
            return detections[0] if detections else None

        best_det  = None
        best_dist = float('inf')
        for d in detections:
            # Préférer la distance PnP, sinon dist_est, sinon valeur par défaut
            dist_d   = d['pnp_dist']    if d['pnp_dist']    is not None else \
                       (d['dist_est']   if d['dist_est']     else 2.0)
            angle_h  = d['pnp_angle_h'] if d['pnp_angle_h'] is not None else d['angle_h']
            gate_dir = yaw - angle_h
            gx = pos[0] + dist_d * np.cos(gate_dir)
            gy = pos[1] + dist_d * np.sin(gate_dir)
            dist_to_tracked = np.linalg.norm([gx - self.gate_rough_pos[0],
                                              gy - self.gate_rough_pos[1]])
            if dist_to_tracked < best_dist:
                best_dist = dist_to_tracked
                best_det  = d

        # Seuil : si trop loin de la cible → considérer comme gate différent
        return best_det if best_dist < 1.5 else None

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
                    # Préférer PnP quand disponible
                    dist         = det['pnp_dist']    if det['pnp_dist']    is not None else \
                                   (det['dist_est']   if det['dist_est']    else 2.0)
                    angle_h_used = det['pnp_angle_h'] if det['pnp_angle_h'] is not None else det['angle_h']
                    angle_v_used = det['pnp_angle_v'] if det['pnp_angle_v'] is not None else det['angle_v']
                    gate_direction = yaw - angle_h_used   # signe corrigé
                    gate_x = pos[0] + dist * np.cos(gate_direction)
                    gate_y = pos[1] + dist * np.sin(gate_direction)
                    gate_z = pos[2] + dist * np.tan(angle_v_used)
                    gate_z = np.clip(gate_z, 0.5, 2.5)
                    self.gate_rough_pos = np.array([gate_x, gate_y, gate_z])
                    self.lost_count = 0
                    self.last_good_yaw = yaw
                    print(f"[STATE] Gate estimé à {self.gate_rough_pos} "
                          f"(dist={'PnP' if det['pnp_dist'] else 'h-ratio'}={dist:.2f}m) → approach")
                    self.state = 'approach'
                    break

            if self.state == 'refine':
                new_yaw = self._normalize_angle(yaw + 10 * dt)
                return [pos[0], pos[1], CRUISE_ALT, new_yaw]

        # =====================================================================
        # APPROACH : servo visuel continu
        #   - Centrer le gate + avancer avec mise à jour continue du centroïde
        #   - Focus sur le gate tracké : on ignore les détections trop loin de
        #     gate_rough_pos (autres gates visibles en même temps)
        #   - Quand assez proche → enregistrer la position et passer à overfly
        #   - Toutes les estimations de position utilisent PnP en priorité
        # =====================================================================
        if self.state == 'approach':
            detections, self.debug_mask, _ = detect_gates(camera_data)

            # -- Ne garder que la détection la plus proche de notre gate cible --
            det = self._closest_detection_to_target(detections, pos, yaw)

            if det is not None:
                # ---- Gate visible : reset compteur de perte ----
                self.lost_count = 0
                self.last_good_yaw = yaw

                # Préférer PnP pour la distance et les angles
                dist         = det['pnp_dist']    if det['pnp_dist']    is not None else \
                               (det['dist_est']   if det['dist_est']    else 2.0)
                angle_h_used = det['pnp_angle_h'] if det['pnp_angle_h'] is not None else det['angle_h']
                angle_v_used = det['pnp_angle_v'] if det['pnp_angle_v'] is not None else det['angle_v']

                # Mettre à jour la position estimée du gate en continu
                gate_direction = yaw - angle_h_used
                gate_x = pos[0] + dist * np.cos(gate_direction)
                gate_y = pos[1] + dist * np.sin(gate_direction)
                gate_z = pos[2] + dist * np.tan(angle_v_used)
                gate_z = np.clip(gate_z, 0.5, 2.5) # limiter les altitudes extrêmes dues au bruit
                # Lissage exponentiel pour éviter les sauts
                #permet de faire une triangulation continue et stable 
                alpha = 0.3
                if self.gate_rough_pos is not None:
                    self.gate_rough_pos = alpha * np.array([gate_x, gate_y, gate_z]) \
                                        + (1 - alpha) * self.gate_rough_pos
                else:
                    self.gate_rough_pos = np.array([gate_x, gate_y, gate_z])

                # -- Assez proche pour une bonne idée du centroïde → enregistrer le gate et passer au-dessus --
                CLOSE_THRESHOLD_H = 80
                if det['h'] > CLOSE_THRESHOLD_H or (det['dist_est'] and det['dist_est'] < 0.6): #si det[...] is not none and det['dist_est'] < 0.6
                    self.gate_positions.append(self.gate_rough_pos.copy()) # stocker la position du gate trouvé
                    self.current_gate_idx = len(self.gate_positions) # indice du gate courant (pour lap 2)
                    # Sauvegarder la direction d'approche + calculer le point de sortie : point 0.8m après le gate dans la même direction, à hauteur du gate
                    approach_dir = yaw - angle_h_used
                    overshoot = 0.8
                    through_z = float(self.gate_rough_pos[2])
                    self.fly_through_target = np.array([
                        self.gate_rough_pos[0] + overshoot * np.cos(approach_dir),
                        self.gate_rough_pos[1] + overshoot * np.sin(approach_dir),
                        through_z
                    ])
                    self.fly_through_yaw = approach_dir
                    print(f"[STATE] Gate {len(self.gate_positions)}/{NUM_GATES} "
                          f"enregistré à {self.gate_rough_pos} → fly_through")
                    self.state = 'fly_through'
                    return [pos[0], pos[1], through_z, yaw]

                # -- Sinon : corriger le yaw pour centrer le gate + avancer --
                target_yaw = self._normalize_angle(yaw - angle_h_used)
                APPROACH_SPEED = 0.35
                step_dir = yaw - angle_h_used
                target_x = pos[0] + APPROACH_SPEED * np.cos(step_dir)
                target_y = pos[1] + APPROACH_SPEED * np.sin(step_dir)
                target_z = float(self.gate_rough_pos[2])
                return [target_x, target_y, target_z, target_yaw]

            else:
                # ---- Gate pas visible ----
                self.lost_count += 1
                MAX_LOST_FRAMES = 300

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
                        SLOW_SPEED = 0.15
                        step = min(SLOW_SPEED, dist)
                        target_x = pos[0] + step * (dx / dist)
                        target_y = pos[1] + step * (dy / dist)
                        target_yaw = np.arctan2(dy, dx)
                        return [target_x, target_y, float(self.gate_rough_pos[2]), target_yaw]

                return [pos[0], pos[1], CRUISE_ALT, yaw]

        # =====================================================================
        # FLY_THROUGH : avancer directement à travers le gate à sa hauteur réelle
        # =====================================================================
        if self.state == 'fly_through':
            dx = self.fly_through_target[0] - pos[0]
            dy = self.fly_through_target[1] - pos[1]
            dist = np.linalg.norm([dx, dy])

            if dist < 0.3:
                self._reset_for_next_gate()
                if len(self.gate_positions) >= NUM_GATES:
                    # Réordonner les gates dans le sens CCW depuis le décollage
                    self.gate_positions = self._reorder_gates_ccw(self.gate_positions)
                    self.lap2_gate_idx  = 0
                    self.lap2_lap_count = 0
                    print(f"[STATE] {NUM_GATES} gates trouvés et réordonnés → lap2")
                    self.state = 'lap2'
                else:
                    print(f"[STATE] Gate traversé → scan_clear")
                    self.state = 'scan_clear'
                return [pos[0], pos[1], self.fly_through_target[2], self.fly_through_yaw]

            FLY_THROUGH_SPEED = 0.5
            step = min(FLY_THROUGH_SPEED, dist)
            move_x = pos[0] + step * (dx / dist)
            move_y = pos[1] + step * (dy / dist)
            return [move_x, move_y, self.fly_through_target[2], self.fly_through_yaw]

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
        # LAP2 : voler à travers les gates connus dans l'ordre CCW, 2 fois
        #   - Setpoint direct sur chaque gate (pas de step limité) → vitesse max
        #   - Passage validé quand le drone est à moins de 0.5 m du centre du gate
        # =====================================================================
        if self.state == 'lap2':

            # Fin d'un tour : passer au suivant ou terminer
            if self.lap2_gate_idx >= NUM_GATES:
                self.lap2_lap_count += 1
                self.lap2_gate_idx   = 0
                if self.lap2_lap_count >= 2:
                    print("[STATE] 2 laps rapides terminés → finished")
                    self.state = 'finished'
                    return [pos[0], pos[1], CRUISE_ALT, yaw]
                print(f"[LAP2] Tour {self.lap2_lap_count + 1}/2 — retour gate 1")

            target = self.gate_positions[self.lap2_gate_idx]
            dx = target[0] - pos[0]
            dy = target[1] - pos[1]
            dist_h = np.linalg.norm([dx, dy])

            # Gate considéré comme traversé quand on est assez proche
            GATE_PASS_DIST = 0.5
            if dist_h < GATE_PASS_DIST:
                print(f"[LAP2] Gate {self.lap2_gate_idx + 1}/{NUM_GATES} traversé "
                      f"(tour {self.lap2_lap_count + 1}/2)")
                self.lap2_gate_idx += 1
                return [pos[0], pos[1], float(target[2]), yaw]

            # Setpoint direct sur le gate (le PID gère la vitesse d'approche)
            target_yaw = np.arctan2(dy, dx)
            return [float(target[0]), float(target[1]), float(target[2]), target_yaw]

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

def get_map_data():
    """Expose les données de position pour l'affichage de la carte dans main.py."""
    return {
        'gate_positions': list(_controller.gate_positions),   # gates confirmés
        'gate_rough_pos': _controller.gate_rough_pos,         # gate en cours de tracking
        'state':          _controller.state,
    }