import numpy as np
import cv2

# =============================================================================
# DÉTECTION DES GATES PAR FILTRE HSV (OpenCV)
# =============================================================================
#
# PIPELINE DE DÉTECTION :
#
# 1. CONVERSION BGR → HSV
#    - L'image de la caméra Webots arrive en BGRA (4 canaux)
#    - On retire le canal alpha, puis on convertit en HSV
#    - HSV = Hue (teinte 0-180), Saturation (0-255), Value (0-255)
#    - Avantage sur BGR : une couleur = une plage de H étroite,
#      indépendamment de l'éclairage (capturé par S et V)
#
# 2. MASQUE COULEUR MAGENTA
#    - Le magenta est à la frontière H≈0 et H≈180 en OpenCV
#    - On fait donc DEUX masques et on les combine avec OR :
#        mask1 : H ∈ [140, 180]  (côté violet-magenta)
#        mask2 : H ∈ [0,   10]   (côté rouge-magenta, wrapping)
#    - cv2.inRange(hsv, lower, upper) → image binaire : 255 si dans la plage, 0 sinon
#
# 3. NETTOYAGE MORPHOLOGIQUE
#    - MORPH_OPEN (érosion puis dilatation) : supprime les petits pixels isolés (bruit)
#    - DILATE : reconnecte les parties du gate séparées par une zone sombre
#
# 4. DÉTECTION DE CONTOURS
#    - cv2.findContours() : trouve les formes blanches dans le masque binaire
#    - On filtre par aire minimale pour ignorer le bruit résiduel
#    - cv2.boundingRect() : donne le rectangle englobant (x, y, w, h) en pixels
#    - cv2.moments() : donne le centroïde exact de la forme (plus précis que boundingRect)
#
# RÉSULTAT : pour chaque gate détecté, on retourne son centroïde en pixels
#            et son angle de visée — sans aucune estimation de distance
#
# =============================================================================

# --- Paramètres caméra ---
# À vérifier avec drone.camera.getWidth() / getHeight() dans main.py
CAM_WIDTH  = 324
CAM_HEIGHT = 244
CAM_FOV_H  = np.radians(90)   # Field of view horizontal — à ajuster selon ta caméra Webots

# --- Filtre couleur magenta en HSV ---
# Ajuste ces valeurs si la détection est mauvaise (voir section CALIBRATION plus bas)
HSV_LOWER_MAG1 = np.array([140,  80,  80])   # Violet-magenta
HSV_UPPER_MAG1 = np.array([180, 255, 255])
HSV_LOWER_MAG2 = np.array([  0,  80,  80])   # Rouge-magenta (wrapping hue)
HSV_UPPER_MAG2 = np.array([ 10, 255, 255])

# --- Seuil de détection ---
MIN_CONTOUR_AREA = 200   # Pixels². Augmente si trop de faux positifs, baisse si tu rates des gates lointains


# =============================================================================
# CALIBRATION HSV — comment trouver les bonnes valeurs
# =============================================================================
# Dans main.py, ajoute temporairement dans la boucle :
#
#   camera_data = drone.read_camera()
#   img_bgr = camera_data[:, :, :3]
#   hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
#   h, s, v = hsv[CAM_HEIGHT//2, CAM_WIDTH//2]
#   print(f"HSV centre : H={h}, S={s}, V={v}")
#
# Pointe la caméra vers un gate et lis les valeurs H, S, V affichées.
# Ensuite : lower = [H-10, S-40, V-40],  upper = [H+10, 255, 255]
# =============================================================================


def detect_gates(image):
    """
    Détecte les gates magenta dans une image caméra.

    Paramètres
    ----------
    image : np.ndarray
        Image BGRA (H x W x 4) telle que retournée par drone.read_camera()

    Retourne
    --------
    detections : list of dict, triée par aire décroissante, chaque dict contenant :
        'cx'     : float  — centroïde X en pixels (0 = bord gauche)
        'cy'     : float  — centroïde Y en pixels (0 = bord haut)
        'x'      : int    — coin haut-gauche du bounding box
        'y'      : int    — coin haut-gauche du bounding box
        'w'      : int    — largeur du bounding box en pixels
        'h'      : int    — hauteur du bounding box en pixels
        'area'   : float  — aire du contour en pixels²
        'angle_h': float  — angle horizontal de visée vers le gate (rad)
                            0 = centre image, négatif = gauche, positif = droite
    """
    detections = []

    if image is None or image.size == 0:
        return detections

    # --- Étape 1 : Retirer le canal alpha si présent (BGRA → BGR) ---
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # --- Étape 2 : Conversion BGR → HSV ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- Étape 3 : Masque magenta (deux plages HSV combinées) ---
    mask1 = cv2.inRange(hsv, HSV_LOWER_MAG1, HSV_UPPER_MAG1)
    mask2 = cv2.inRange(hsv, HSV_LOWER_MAG2, HSV_UPPER_MAG2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # --- Étape 4 : Nettoyage morphologique ---
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)  # Supprime le bruit
    mask = cv2.dilate(mask, kernel, iterations=1)            # Reconnecte les parties

    # --- Étape 5 : Détection des contours ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Étape 6 : Filtrage et extraction ---
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        # Bounding box de l'aire ==> x et y ; position  du coin haut-gauche, w et h : dimensions de la boîte englobante
        x, y, w, h = cv2.boundingRect(contour)

        # Centroïde précis via les moments d'image
        # M["m00"] = aire,  m10/m00 = X moyen,  m01/m00 = Y moyen
        M = cv2.moments(contour)
        if M["m00"] == 0:
            cx = x + w / 2.0
            cy = y + h / 2.0
        else:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

        # Angle horizontal de visée : combien de radians à gauche/droite du centre image
        # angle_per_pixel = FOV_total / nb_pixels_total
        angle_per_pixel = CAM_FOV_H / CAM_WIDTH
        angle_h = (cx - CAM_WIDTH / 2.0) * angle_per_pixel   # rad, signé

        detections.append({
            'cx':      cx,
            'cy':      cy,
            'x':       x,
            'y':       y,
            'w':       w,
            'h':       h,
            'area':    area,
            'angle_h': angle_h,
        })

    # Trier par aire décroissante
    detections.sort(key=lambda d: d['area'], reverse=True)

    return detections


def draw_detections(image, detections):
    """
    Dessine les détections sur l'image pour le debug visuel.
    Retourne une image BGR annotée — à afficher avec cv2.imshow() depuis main.py.

    Exemple dans main.py :
        annotated = draw_detections(camera_data, detections)
        cv2.imshow("Gates", annotated)
        cv2.waitKey(1)
    """
    if image.shape[2] == 4:
        image_out = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        image_out = image.copy()

    for det in detections:
        # Bounding box en vert
        cv2.rectangle(image_out,
                      (det['x'], det['y']),
                      (det['x'] + det['w'], det['y'] + det['h']),
                      (0, 255, 0), 2)

        # Centroïde en rouge
        cv2.circle(image_out, (int(det['cx']), int(det['cy'])), 5, (0, 0, 255), -1)

        # Label : angle de visée et aire
        label = f"a={np.degrees(det['angle_h']):.1f}deg  A={det['area']:.0f}"
        cv2.putText(image_out, label,
                    (det['x'], max(det['y'] - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return image_out


# =============================================================================
# ASSIGNMENT — structure minimale, détection branchée, navigation à venir
# =============================================================================

class MyAssignment:
    def __init__(self):
        self.state = 'takeoff'
        # La triangulation et la navigation seront ajoutées ici

    def compute_command(self, sensor_data, camera_data, dt):

        pos = np.array([sensor_data['x_global'],
                        sensor_data['y_global'],
                        sensor_data['z_global']])
        yaw = sensor_data['yaw']

        # --- Détection des gates à chaque frame ---
        detections = detect_gates(camera_data)

        if detections:
            best = detections[0]  # Plus grande aire = plus en face / plus proche
            print(f"[VISION] Gate : centroïde=({best['cx']:.1f}px, {best['cy']:.1f}px)  "
                  f"angle_h={np.degrees(best['angle_h']):.1f}°  aire={best['area']:.0f}px²")

        # --- Décollage minimal pour tester la détection ---
        if sensor_data['z_global'] < 0.49:
            return [pos[0], pos[1], 1.0, yaw]

        # Reste sur place — navigation à coder ensuite
        return [pos[0], pos[1], 1.0, yaw]


_controller = MyAssignment()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)