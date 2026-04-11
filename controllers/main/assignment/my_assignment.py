import numpy as np
import cv2

# =============================================================================
# PARAMÈTRES CAMÉRA
# =============================================================================
CAM_WIDTH  = 324
CAM_HEIGHT = 244
CAM_FOV_H  = np.radians(90)

# =============================================================================
# PARAMÈTRES FILTRE HSV
# =============================================================================
HSV_LOWER_MAG1 = np.array([140,  80,  80])   # S=80 : exclut les murs peu saturés
HSV_UPPER_MAG1 = np.array([180, 255, 255])
HSV_LOWER_MAG2 = np.array([  0,  80,  80])
HSV_UPPER_MAG2 = np.array([ 15, 255, 255])

MIN_CONTOUR_AREA = 300


# =============================================================================
# DÉTECTION HSV + COINS — appelée une seule fois par frame
# =============================================================================
def detect_gates(image):
    """
    Détecte les gates magenta dans une image BGRA.

    Retourne (detections, debug_mask).

    Chaque détection est un dict :
        'cx', 'cy'       : centroïde en pixels
        'x','y','w','h'  : bounding box
        'area'           : aire du contour en px²
        'angle_h'        : angle horizontal de visée (rad)
        'has_4_corners'  : True si le contour est approximé par 4 coins (gate de face)
        'corner_count'   : nombre de coins détectés par approxPolyDP
        'corners'        : les coins en pixels (np.array) si has_4_corners, sinon None

    COMMENT FONCTIONNE LA DÉTECTION DES 4 COINS :
    - cv2.approxPolyDP(contour, epsilon, closed=True) simplifie le contour
      en polygone avec le minimum de points pour rester à moins de epsilon pixels
      du contour original.
    - epsilon = 0.04 * périmètre du contour : tolérance de 4%
    - Si le polygone simplifié a exactement 4 points → rectangle → gate vu de face
    - Si plus de 4 points → gate vu de côté ou partiellement visible
    """
    detections = []

    if image is None or image.size == 0:
        return detections, None

    # BGRA → BGR
    if image.ndim == 3 and image.shape[2] == 4:
        img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        img = image.copy()

    # BGR → HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Masque magenta
    mask1 = cv2.inRange(hsv, HSV_LOWER_MAG1, HSV_UPPER_MAG1)
    mask2 = cv2.inRange(hsv, HSV_LOWER_MAG2, HSV_UPPER_MAG2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # Nettoyage morphologique
    kernel_open  = np.ones((5,  5),  np.uint8)
    kernel_close = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Image debug : fond noir, contours blancs
    debug_mask = np.zeros_like(img)

    angle_per_pixel = CAM_FOV_H / CAM_WIDTH

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        # Dessiner le contour sur le masque debug
        cv2.drawContours(debug_mask, [contour], -1, (255, 255, 255), 2)

        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Centroïde
        M = cv2.moments(contour)
        if M["m00"] == 0:
            cx = x + w / 2.0
            cy = y + h / 2.0
        else:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

        # Angle de visée horizontal
        angle_h = (cx - CAM_WIDTH / 2.0) * angle_per_pixel

        # --- Filtres géométriques pour rejeter les faux positifs ---

        # 1. Taille minimum : la bordure caméra est très petite
        MIN_BBOX_SIZE = 20   # px — bounding box doit faire au moins 20x20
        if w < MIN_BBOX_SIZE or h < MIN_BBOX_SIZE:
            continue
 
        # 2. Ratio hauteur/largeur : un gate est approximativement carré (0.5 à 2.0)
        #    La bordure caméra est un trait très allongé (ratio >> 2 ou << 0.5)
        aspect_ratio = h / w if w > 0 else 999
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            continue

    
        # --- Détection des coins avec approxPolyDP ---
        # epsilon = tolérance en pixels (4% du périmètre du contour)
        perimeter = cv2.arcLength(contour, closed=True)
        epsilon   = 0.04 * perimeter
        approx    = cv2.approxPolyDP(contour, epsilon, closed=True)
        corner_count = len(approx)
        has_4_corners = (corner_count == 4)

        # Dessiner les coins en jaune sur le masque debug si 4 coins
        if has_4_corners:
            for pt in approx:
                cv2.circle(debug_mask, tuple(pt[0]), 5, (0, 255, 255), -1)

        detections.append({
            'cx': cx, 'cy': cy,
            'x': x, 'y': y, 'w': w, 'h': h,
            'area': area,
            'angle_h': angle_h,
            'has_4_corners': has_4_corners,
            'corner_count':  corner_count,
            'corners': approx if has_4_corners else None,
        })

    detections.sort(key=lambda d: d['area'], reverse=True)

    return detections, debug_mask


# =============================================================================
# ASSIGNMENT
# =============================================================================
class MyAssignment:
    def __init__(self):
        # --- États ---
        # 'takeoff'   : monter à 1.5m
        # 'turn_90'  : tourner de 90° face au mur (caméra non utilisée)
        # 'scan_ccw'  : tourner CCW et chercher le gate 0 (4 coins détectés)
        # 'done'      : gate 0 trouvé, rester sur place (navigation à venir)
        self.state = 'takeoff'
        self.debug_mask = None

        self.yaw_90_target = None   # Yaw cible pour turn_90
        self.gate0_angle_h  = None   # Angle de visée vers gate 0 au moment de la détection

        # Suivi pendant refine
        self.best_area     = 0.0
        self.best_angle_h  = None
        self.refine_shrink_count = 0   # Combien de frames consécutives l'aire a diminué

    def compute_command(self, sensor_data, camera_data, dt):

        pos = np.array([sensor_data['x_global'],
                        sensor_data['y_global'],
                        sensor_data['z_global']])
        yaw = sensor_data['yaw']

        # =====================================================================
        # TAKEOFF : monter à 1.5m
        # =====================================================================
        if self.state == 'takeoff':
            if sensor_data['z_global'] < 1.4:
                return [pos[0], pos[1], 1.5, yaw]
            else:
                self.yaw_90_target = self._normalize_angle(yaw - np.pi/4) #déf la target à +90° par rapport à l'orientation actuelle
                self.state = 'turn_90'
                print(f"[STATE] Décollage OK → turn_90 (cible={np.degrees(self.yaw_90_target):.1f}°)")

        # =====================================================================
        # TURN_90 : tourner face au mur, caméra non utilisée
        # =====================================================================
        if self.state == 'turn_90':
            error = self._angle_diff(self.yaw_90_target, yaw) #différence entre l'angle cible et l'angle actuel, positive si on doit tourner CCW
            if abs(error) < 0.05:   # ~3° de précision
                self.state = 'scan_ccw'
                print("[STATE] 90° atteint → scan_ccw")
            return [pos[0], pos[1], 1.5, self.yaw_90_target]

        # =====================================================================
        # SCAN_CCW : tourner CCW jusqu'à voir les premiers coins
        # =====================================================================
        if self.state == 'scan_ccw':
            detections, self.debug_mask = detect_gates(camera_data)
 
            for det in detections:
                if det['has_4_corners']:
                    # Premiers coins trouvés → passer en refine
                    self.best_area    = det['area']
                    self.best_angle_h = det['angle_h']
                    self.refine_shrink_count = 0
                    self.state = 'refine'
                    print(f"[STATE] Premiers coins détectés (aire={det['area']:.0f}) → refine")
                    break
 
            if self.state == 'scan_ccw':
                new_yaw = self._normalize_angle(yaw + 15 * dt)
                return [pos[0], pos[1], 1.5, new_yaw]
 
         # =====================================================================
        # REFINE : continuer à tourner CCW, s'arrêter quand l'aire est stable
        # "Stable" = variation < STABLE_THRESHOLD % sur N frames consécutives
        # =====================================================================
        if self.state == 'refine':
            STABLE_THRESHOLD = 0.02   # 2% de variation max pour considérer l'aire stable
            STABLE_FRAMES    = 7      # Nombre de frames stables consécutives pour valider
 
            detections, self.debug_mask = detect_gates(camera_data)
 
            current_area    = 0.0
            current_angle_h = self.best_angle_h
            for det in detections:
                if det['has_4_corners']:
                    current_area    = det['area']
                    current_angle_h = det['angle_h']
                    break
 
            if current_area > 0:
                # Calculer la variation relative par rapport à la meilleure aire vue
                if self.best_area > 0:
                    variation = abs(current_area - self.best_area) / self.best_area
                else:
                    variation = 1.0
 
                # Mettre à jour la meilleure aire
                if current_area > self.best_area:
                    self.best_area = current_area
 
                self.best_angle_h = current_angle_h
 
                if variation < STABLE_THRESHOLD:
                    self.refine_shrink_count += 1
                    if self.refine_shrink_count >= STABLE_FRAMES:
                        print(f"[STATE] Gate 0 centré ! aire={self.best_area:.0f}  "
                              f"angle_h={np.degrees(self.best_angle_h):.1f}°")
                        self.state = 'done'
                else:
                    # Variation trop grande → réinitialiser le compteur
                    self.refine_shrink_count = 0
 
            # Continuer à tourner dans tous les cas sauf si done
            if self.state == 'refine':
                new_yaw = self._normalize_angle(yaw + 10 * dt)
                return [pos[0], pos[1], 1.5, new_yaw]
            
        # =====================================================================
        # DONE : gate 0 trouvé — navigation à coder ici
        # =====================================================================
        if self.state == 'done':
            return [pos[0], pos[1], 1.5, yaw]

        return [pos[0], pos[1], 1.5, yaw]

    def _normalize_angle(self, angle):
        """Ramène un angle dans [-π, π]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _angle_diff(self, target, current): 
        """Différence signée entre deux angles (résultat dans [-π, π])"""
        return self._normalize_angle(target - current)


# =============================================================================
# Interface avec main.py
# =============================================================================
_controller = MyAssignment()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)

def get_debug_mask():
    return _controller.debug_mask