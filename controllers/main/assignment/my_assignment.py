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
# PARAMÈTRES FILTRE HSV  (une seule plage, à ajuster selon l'éclairage)
# =============================================================================
HSV_LOWER_MAG1 = np.array([138,  20,  110])
HSV_UPPER_MAG1 = np.array([158, 210, 255])

MIN_CONTOUR_AREA = 400

NUM_GATES = 5
OVERFLY_ALT = 2.0          # altitude pour passer au-dessus du gate
CRUISE_ALT  = 1.5          # altitude de croisière normale
TAKEOFF_POS    = np.array([1.0, 4.0])   # position 2D fixe du pad de décollage [x=1, y=4]
CIRCUIT_CENTER = np.array([4.0, 4.0])  # centre fixe de l'arène (circle_centre dans main.py)


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

    mask = cv2.inRange(hsv, HSV_LOWER_MAG1, HSV_UPPER_MAG1)

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

        MIN_BBOX_SIZE = 10
        if w < MIN_BBOX_SIZE or h < MIN_BBOX_SIZE:
            continue

        # Rejeter les contours très allongés (probablement du bruit ou des barres isolées
        # d'un gate, pas le gate entier). Un gate vu même de biais garde un ratio raisonnable.
        aspect_ratio = w / max(h, 1)
        if aspect_ratio > 3.0 or aspect_ratio < 0.33:
            continue

        # Filtre noir à droite : s'assure que le gate détecté est isolé à sa droite
        # (pas un morceau de gate collé à un autre). Seuil assoupli à 40% pour tolérer
        # les scènes avec plusieurs gates proches les uns des autres.
        BLACK_MARGIN = 10
        right_edge = min(x + w + BLACK_MARGIN, mask.shape[1] - 1)
        has_black_right = False
        if right_edge > x + w:
            right_strip = mask[y:y+h, x+w:right_edge]
            white_ratio = np.sum(right_strip > 0) / right_strip.size
            has_black_right = (white_ratio <= 0.4)

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

        # Timeout scan_ccw : si on tourne depuis longtemps sans voir 4 coins,
        # on accepte le plus gros contour comme fallback
        self.scan_frames = 0

        # Stop-and-detect en scan_ccw :
        # Le drone alterne entre rotation (SCAN_ROT_FRAMES frames) et pause
        # (SCAN_PAUSE_FRAMES frames) pour laisser l'image se stabiliser avant
        # de tenter la détection. Détection uniquement pendant la pause.
        self.scan_phase        = 'rotate'   # 'rotate' ou 'pause'
        self.scan_phase_frames = 0          # frames écoulées dans la phase courante

        # Position estimée du gate courant (lissée)
        self.gate_rough_pos  = None
        self.lost_count      = 0
        self.last_good_yaw   = None

        # Stockage des gates détectés (lap 1)
        self.gate_positions = []        # liste de np.array([x, y, z])
        self.current_gate_idx = 0

        # Variables pour les laps rapides (lap 2 et 3)
        self.lap2_gate_idx  = 0    # indice du gate courant dans l'ordre CCW
        self.lap2_lap_count = 0    # nombre de laps rapides effectués (max 2)
        self.lap2_started   = False  # False tant qu'on n'est pas revenu au décollage

        # Trajectoire polynomiale min-snap pour lap2
        self.trajectory      = None   # dict {'polys': [...], 'durations': [...], 'total_time': T}
        self.traj_start_time = None   # temps simulation au démarrage de la trajectoire
        self.last_time       = 0.0    # temps courant cumulé depuis traj_start_time

        # Position réelle au sol enregistrée au premier step (plus fiable que TAKEOFF_POS)
        self.takeoff_pos_recorded = None

    def _reset_for_next_gate(self):
        """Réinitialise les variables de tracking pour chercher le gate suivant."""
        self.gate_rough_pos = None
        self.lost_count = 0
        self.last_good_yaw = None
        self.best_area = 0.0
        self.best_angle_h = None
        self.refine_shrink_count = 0
        self.scan_frames = 0
        self.scan_phase = 'rotate'
        self.scan_phase_frames = 0

    def _reorder_gates_ccw(self, gate_positions):
        """
        Réordonne les gates dans l'ordre de la course (CCW = drone_angle croissant).

        Utilise exactement la même convention angulaire que check_segment dans main.py :
          drone_angle = arctan2(gy - cy, gx - cx) + π  ∈ [0, 2π]
        avec cx, cy = CIRCUIT_CENTER = [4, 4].

        Le segment 0 (décollage) couvre drone_angle ∈ [330°, 15°].
        Le gate 1 (segment 1) commence à ~45°.
        On trie par drone_angle croissant puis on fait pivoter la liste
        pour que le premier gate soit celui juste après la zone de décollage.
        """
        def drone_angle(gp):
            rel = np.array(gp[:2]) - CIRCUIT_CENTER
            norm = np.linalg.norm(rel)
            if norm < 1e-6:
                return 0.0
            return (np.arctan2(rel[1] / norm, rel[0] / norm) + np.pi) % (2 * np.pi)

        sorted_gates = sorted(gate_positions, key=drone_angle)

        # Borne supérieure du segment 0 : π/12 rad ≈ 15°
        # Le premier gate de la course a drone_angle > π/12
        TAKEOFF_ZONE_END = np.pi / 12

        start_idx = 0
        for i, gp in enumerate(sorted_gates):
            if drone_angle(gp) > TAKEOFF_ZONE_END:
                start_idx = i
                break

        # Rotation de la liste pour partir du bon gate
        ordered = sorted_gates[start_idx:] + sorted_gates[:start_idx]

        print(f"[CCW] Centre arène : {CIRCUIT_CENTER}")
        for i, gp in enumerate(ordered):
            print(f"[CCW] Gate {i+1} → {gp} (drone_angle={np.degrees(drone_angle(gp)):.1f}°)")
        return ordered

    def _estimate_gate_normal(self, gate_pos, prev_pos, next_pos):
        """
        Estime la normale de traversée d'un gate (direction dans laquelle le drone
        doit le traverser). La logique : le drone vient du gate précédent, traverse
        ce gate, puis va vers le gate suivant. La normale est la moyenne des deux
        directions (tangente locale du chemin).

        Retourne un vecteur 2D unitaire.
        """
        dir_in  = np.array(gate_pos[:2]) - np.array(prev_pos[:2])
        dir_out = np.array(next_pos[:2]) - np.array(gate_pos[:2])
        d_in  = dir_in  / (np.linalg.norm(dir_in)  + 1e-9)
        d_out = dir_out / (np.linalg.norm(dir_out) + 1e-9)
        tangent = (d_in + d_out) / 2.0
        n = np.linalg.norm(tangent)
        return tangent / n if n > 1e-6 else d_out

    # =========================================================================
    # TRAJECTOIRE MIN-SNAP (polynômes d'ordre 7)
    # =========================================================================
    def _compute_min_snap_segment(self, p0, v0, a0, j0, p1, v1, a1, j1, T):
        """
        Calcule un polynôme d'ordre 7 p(t) = c0 + c1 t + ... + c7 t^7
        qui passe de (p0, v0, a0, j0) à t=0 à (p1, v1, a1, j1) à t=T.

        Ordre 7 = 8 coefficients, 8 contraintes (4 au début + 4 à la fin)
        → système linéaire carré résolu exactement.

        Paramètres :
          p0, v0, a0, j0 : position, vitesse, accélération, jerk de départ
          p1, v1, a1, j1 : position, vitesse, accélération, jerk d'arrivée
          T              : durée du segment

        Retourne : np.array([c0, c1, c2, c3, c4, c5, c6, c7]) de taille (8, d)
                   où d = dimension de p (3 pour x/y/z traités en bloc)
        """
        # Matrice des contraintes : chaque ligne = une équation
        #   p(0) = c0                      → [1, 0, 0, 0, 0, 0, 0, 0]
        #   v(0) = c1                      → [0, 1, 0, 0, 0, 0, 0, 0]
        #   a(0) = 2 c2                   → [0, 0, 2, 0, 0, 0, 0, 0]
        #   j(0) = 6 c3                   → [0, 0, 0, 6, 0, 0, 0, 0]
        #   p(T) = Σ ck T^k               → [1, T, T², T³, T⁴, T⁵, T⁶, T⁷]
        #   v(T) = Σ k ck T^(k-1)
        #   a(T) = Σ k(k-1) ck T^(k-2)
        #   j(T) = Σ k(k-1)(k-2) ck T^(k-3)
        A = np.array([
            [1, 0, 0,  0,     0,      0,       0,         0],
            [0, 1, 0,  0,     0,      0,       0,         0],
            [0, 0, 2,  0,     0,      0,       0,         0],
            [0, 0, 0,  6,     0,      0,       0,         0],
            [1, T, T**2, T**3,    T**4,      T**5,       T**6,         T**7],
            [0, 1, 2*T,  3*T**2,  4*T**3,    5*T**4,     6*T**5,       7*T**6],
            [0, 0, 2,    6*T,    12*T**2,   20*T**3,    30*T**4,      42*T**5],
            [0, 0, 0,    6,      24*T,      60*T**2,   120*T**3,     210*T**4],
        ], dtype=np.float64)

        b = np.array([p0, v0, a0, j0, p1, v1, a1, j1], dtype=np.float64)
        # np.linalg.solve gère les tableaux multi-colonnes : résout pour x, y, z simultanément
        return np.linalg.solve(A, b)

    def _eval_poly(self, coeffs, t):
        """Évalue un polynôme et ses dérivées à t. Retourne (pos, vel, acc)."""
        # coeffs shape : (8, d)  avec d = 3 pour position 3D
        t_pow  = np.array([t**k for k in range(8)])
        dt_pow = np.array([k * t**(k-1) if k >= 1 else 0 for k in range(8)])
        # pos = Σ ck t^k ; on fait un produit scalaire ligne par ligne
        pos = np.einsum('k,kd->d', t_pow,  coeffs)
        vel = np.einsum('k,kd->d', dt_pow, coeffs)
        return pos, vel

    def _compute_trajectory(self, start_pos, waypoints, end_pos,
                            avg_speed=3, gate_speed_factor=0.6):
        """
        Calcule une trajectoire min-snap par morceaux entre start_pos, tous les
        waypoints (passages de gates), et end_pos.

        La trajectoire est composée de N segments polynomiaux d'ordre 7, avec :
          - Position, vitesse, accélération, jerk continus à chaque joint
          - Vitesse et accélération nulles aux extrémités (start et end)
          - Vitesse aux gates imposée dans la direction de la normale, mais RÉDUITE
            (gate_speed_factor < 1) pour forcer le drone à passer précisément par
            le centroïde. Moins de vitesse = moins d'inertie = moins d'écart
            lors du passage.
          - Durée de chaque segment : d / avg_speed (cruise plein).
            Plafond 5s pour éviter l'instabilité numérique du polynôme d'ordre 7
            (T^7 explose si T est grand, rendant les coefficients aberrants).

        Paramètres :
          avg_speed          : vitesse CRUISE entre gates (m/s) → monter pour + rapide
          gate_speed_factor  : facteur ]0,1] appliqué à la vitesse AU centroïde
                               des gates. 0.7 = traversée 30% plus lente pour
                               garantir le passage au centre.

        Retourne :
          dict {'polys':     liste des matrices de coefficients (8, 3) par segment,
                'durations': liste des durées par segment,
                'total_time': durée totale}
        """
        # Construction de la liste complète de points [start, wp1, wp2, ..., end]
        all_points = [{'pos': start_pos, 'normal': None}]
        all_points.extend(waypoints)
        all_points.append({'pos': end_pos, 'normal': None})
        N = len(all_points) - 1   # nombre de segments

        gate_speed = avg_speed * gate_speed_factor

        # --- Durées des segments : d / avg_speed (cruise plein) ---
        # Ce calcul utilise avg_speed DIRECTEMENT → monter avg_speed = aller plus vite.
        # Plancher 0.3s : évite les segments trop courts numériquement instables.
        # Plafond  5.0s : évite T^7 énorme qui rend les coefficients aberrants
        #                 (trajectoire qui "explose" entre deux waypoints).
        durations = []
        for i in range(N):
            d = np.linalg.norm(all_points[i+1]['pos'] - all_points[i]['pos'])
            durations.append(float(np.clip(d / avg_speed, 0.3, 5.0)))

        # --- Vitesse cible à chaque point ---
        # start et end : vitesse nulle
        # waypoints intermédiaires : vitesse dans la direction de la normale (2D)
        #                            magnitude = gate_speed (réduite !)
        velocities = [np.zeros(3)]
        for i in range(1, N):
            wp = all_points[i]
            if wp['normal'] is not None:
                n = wp['normal']
                v = np.array([n[0] * gate_speed, n[1] * gate_speed, 0.0])
            else:
                dp = all_points[i+1]['pos'] - all_points[i-1]['pos']
                dp = dp / (np.linalg.norm(dp) + 1e-9) * gate_speed
                v = dp
            velocities.append(v)
        velocities.append(np.zeros(3))  # end : vitesse nulle

        # --- Accélération et jerk nuls partout (simplification pragmatique) ---
        accs  = [np.zeros(3) for _ in range(N + 1)]
        jerks = [np.zeros(3) for _ in range(N + 1)]

        # --- Calcul des polynômes segment par segment ---
        polys = []
        for i in range(N):
            p0 = all_points[i]['pos']
            p1 = all_points[i+1]['pos']
            coeffs = self._compute_min_snap_segment(
                p0, velocities[i],   accs[i],   jerks[i],
                p1, velocities[i+1], accs[i+1], jerks[i+1],
                durations[i]
            )
            polys.append(coeffs)

        total_time = sum(durations)
        print(f"[TRAJ] {N} segments, durée totale = {total_time:.2f}s, "
              f"vitesse cruise = {avg_speed:.1f}m/s, gate = {gate_speed:.1f}m/s")

        return {'polys': polys, 'durations': durations, 'total_time': total_time}

    def _sample_trajectory(self, traj, t):
        """
        Évalue la trajectoire à l'instant global t.
        Retourne la position (np.array(3)) et la vitesse (np.array(3)).

        Si t dépasse total_time, retourne la position finale avec vitesse nulle.
        """
        if t >= traj['total_time']:
            # Fin de trajectoire : évaluer le dernier segment à sa durée max
            last_poly = traj['polys'][-1]
            last_dur  = traj['durations'][-1]
            pos, _ = self._eval_poly(last_poly, last_dur)
            return pos, np.zeros(3)

        # Trouver le segment courant et le temps local
        t_local = t
        for i, dur in enumerate(traj['durations']):
            if t_local <= dur:
                pos, vel = self._eval_poly(traj['polys'][i], t_local)
                return pos, vel
            t_local -= dur

        # Sécurité (ne devrait pas arriver)
        last_poly = traj['polys'][-1]
        last_dur  = traj['durations'][-1]
        pos, _ = self._eval_poly(last_poly, last_dur)
        return pos, np.zeros(3)

    # =========================================================================
    # MODIFICATION : waypoint décalé à droite du gate pour compenser l'inertie
    # =========================================================================
    def _build_lap2_waypoints_shrunk(self, ordered_gates, start_pos,
                                     lateral_offset=0.08):
        """
        Construit la liste des waypoints pour 2 tours à partir des gates ordonnés
        (liste de np.array([x, y, z])).

        Pour chaque gate on calcule sa normale = tangente locale du chemin
        (direction moyenne gate_précédent → gate → gate_suivant).

        Le waypoint visé est décalé latéralement par rapport au centroïde.
        Quand le drone va vite en CCW, l'inertie le pousse vers l'extérieur
        du virage (gauche dans le sens de traversée). En décalant le waypoint
        vers la DROITE, on compense ce drift pour que le drone passe dans le gate.

          "Droite" dans le sens de traversée :
          si normale = (nx, ny)  →  droite = (ny, -nx)   [rotation de -90°]

        lateral_offset > 0 → décalage vers la droite (compense dérive CCW)
        lateral_offset = 0 → centroïde exact (pas de compensation)

        Retourne : liste de dicts {'pos': np.array(3), 'normal': np.array(2)}
                   répétée 2 fois pour 2 laps.
        """
        n = len(ordered_gates)
        base_waypoints = []
        for i, g in enumerate(ordered_gates):
            # Pour la normale : point précédent = gate i-1 (ou start pour i=0)
            #                   point suivant  = gate i+1 (ou gate 0 pour i=n-1, boucle)
            prev_pos = ordered_gates[i - 1] if i > 0 else start_pos
            next_pos = ordered_gates[(i + 1) % n]
            normal   = self._estimate_gate_normal(g, prev_pos, next_pos)

            # Vecteur perpendiculaire à droite de la normale (rotation de -90°) :
            #   normale (nx, ny)  →  droite (ny, -nx)
            right_2d = np.array([normal[1], -normal[0]])

            # Position visée = centroïde + offset latéral vers la droite
            g_target = np.array(g, dtype=np.float64).copy()
            g_target[0] += lateral_offset * right_2d[0]
            g_target[1] += lateral_offset * right_2d[1]

            base_waypoints.append({
                'pos':    g_target,
                'normal': normal,
            })
        # Répéter pour 2 laps
        return base_waypoints + base_waypoints

    def _plot_trajectory(self, traj, ordered_gates, waypoints, start_pos, end_pos,
                         filename='trajectory.png'):
        """
        Génère un plot 3D de la trajectoire complète avec les gates.
        Sauvegardé en PNG à la racine du projet Webots.

        ordered_gates : liste des np.array([x, y, z]) (centroïdes uniquement)
        waypoints     : liste de dicts {'pos', 'normal'} (contient normales estimées)
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa
        except ImportError:
            print("[PLOT] matplotlib non disponible — plot ignoré")
            return

        # Échantillonner la trajectoire à haute résolution pour le tracé
        t_samples = np.linspace(0, traj['total_time'], 500)
        path = np.array([self._sample_trajectory(traj, t)[0] for t in t_samples])

        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection='3d')

        # Trajectoire
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'k-', lw=2, label='Trajectoire min-snap')

        # Gates : rectangle dans le plan perpendiculaire à la normale
        # On n'a pas accès aux coins 3D, donc on dessine un rectangle symbolique
        # de taille GATE_REAL_WIDTH × GATE_REAL_HEIGHT centré sur le centroïde
        for i, g in enumerate(ordered_gates):
            # Normale du gate (depuis les waypoints du premier lap)
            n = waypoints[i]['normal'] if i < len(waypoints) else np.array([1.0, 0.0])
            # Vecteur tangent dans le plan XY (perpendiculaire à n)
            tangent = np.array([-n[1], n[0], 0.0])
            up      = np.array([0.0, 0.0, 1.0])
            half_w  = GATE_REAL_WIDTH  / 2.0
            half_h  = GATE_REAL_HEIGHT / 2.0
            corners = np.array([
                g + tangent * half_w + up * half_h,
                g - tangent * half_w + up * half_h,
                g - tangent * half_w - up * half_h,
                g + tangent * half_w - up * half_h,
                g + tangent * half_w + up * half_h,  # fermer
            ])
            ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], 'r-', lw=2)
            ax.scatter(*g, c='cyan', s=80, edgecolors='k', zorder=5)
            ax.text(g[0], g[1], g[2] + 0.15, f'G{i+1}', fontsize=10, ha='center')

        # Start / end
        ax.scatter(*start_pos, c='green', s=120, marker='o', label='Start', edgecolors='k', zorder=5)
        ax.scatter(*end_pos,   c='red',   s=120, marker='X', label='End',   edgecolors='k', zorder=5)

        ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
        ax.set_title(f"Trajectoire min-snap ({traj['total_time']:.1f}s)")
        ax.legend()
        ax.set_box_aspect([1, 1, 0.4])

        try:
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            print(f"[PLOT] Trajectoire sauvegardée → {filename}")
            plt.close(fig)
        except Exception as e:
            print(f"[PLOT] Erreur sauvegarde : {e}")

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
            # Enregistrer la position réelle du pad dès le premier step
            if self.takeoff_pos_recorded is None:
                self.takeoff_pos_recorded = pos[:2].copy()
                print(f"[TAKEOFF] Position décollage enregistrée : {self.takeoff_pos_recorded}")
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
        # SCAN_CCW : rotation CCW par petites impulsions, avec pauses de détection
        #
        # Principe stop-and-detect :
        #   - Phase 'rotate' (SCAN_ROT_FRAMES frames)   : tourner CCW, pas de détection
        #   - Phase 'pause'  (SCAN_PAUSE_FRAMES frames) : immobile, détecter le violet
        #
        # La détection n'a lieu QUE pendant la pause, quand l'image est stable.
        # Fallback : après SCAN_TIMEOUT frames totales sans succès, accepter le
        # plus gros contour visible même sans 4 coins.
        # =====================================================================
        if self.state == 'scan_ccw':
            SCAN_ROT_FRAMES   = 12   # frames de rotation (~0.24s à 50Hz) → ~20° par impulsion
            SCAN_PAUSE_FRAMES = 12   # frames d'arrêt pour laisser l'image se stabiliser
            SCAN_TIMEOUT      = 400  # frames totales avant fallback (~8s à 50Hz)

            self.scan_frames       += 1
            self.scan_phase_frames += 1

            # --- Transition entre phases ---
            if self.scan_phase == 'rotate' and self.scan_phase_frames >= SCAN_ROT_FRAMES:
                self.scan_phase        = 'pause'
                self.scan_phase_frames = 0
            elif self.scan_phase == 'pause' and self.scan_phase_frames >= SCAN_PAUSE_FRAMES:
                self.scan_phase        = 'rotate'
                self.scan_phase_frames = 0

            # --- Détection uniquement pendant la pause (image stable) ---
            if self.scan_phase == 'pause':
                detections, self.debug_mask, _ = detect_gates(camera_data)

                for det in detections:
                    if det['has_4_corners']:
                        self.best_area    = det['area']
                        self.best_angle_h = det['angle_h']
                        self.refine_shrink_count = 0
                        self.state = 'refine'
                        self.scan_frames = 0
                        self.scan_phase = 'rotate'
                        self.scan_phase_frames = 0
                        print(f"[STATE] Gate détecté (pause) aire={det['area']:.0f} → refine")
                        break

                # Fallback timeout : accepter le plus gros contour même sans 4 coins
                if self.state == 'scan_ccw' and self.scan_frames > SCAN_TIMEOUT and detections:
                    det = detections[0]
                    self.best_area    = det['area']
                    self.best_angle_h = det['angle_h']
                    self.refine_shrink_count = 0
                    self.state = 'refine'
                    self.scan_frames = 0
                    self.scan_phase = 'rotate'
                    self.scan_phase_frames = 0
                    print(f"[STATE] Timeout scan → refine fallback "
                          f"aire={det['area']:.0f} ({det['corner_count']} coins)")

            # --- Commande de mouvement ---
            if self.state == 'scan_ccw':
                if self.scan_phase == 'rotate':
                    new_yaw = self._normalize_angle(yaw + 10 * dt)
                    return [pos[0], pos[1], CRUISE_ALT, new_yaw]
                else:
                    # Pause : rester immobile au yaw courant
                    return [pos[0], pos[1], CRUISE_ALT, yaw]

        # =====================================================================
        # REFINE : tourner jusqu'à noir gauche aussi → estimer position du gate
        #   Fallback : si on tourne trop longtemps sans avoir les deux côtés noirs,
        #   accepter avec le plus gros contour et estimation PnP ou dist_est
        # =====================================================================
        if self.state == 'refine':
            detections, self.debug_mask, mask_ref = detect_gates(camera_data)
            self.scan_frames += 1

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
                    # Seuil assoupli (0.2 → 0.4) pour tolérer les gates proches d'autres gates
                    has_black_left = (white_ratio_left <= 0.4)

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
                    self.scan_frames = 0
                    print(f"[STATE] Gate estimé à {self.gate_rough_pos} "
                          f"(dist={'PnP' if det['pnp_dist'] else 'h-ratio'}={dist:.2f}m) → approach")
                    self.state = 'approach'
                    break

            # Fallback refine : si on tourne depuis longtemps sans avoir les deux côtés
            # noirs, on accepte quand même avec le plus gros contour visible
            REFINE_TIMEOUT = 200
            if self.state == 'refine' and self.scan_frames > REFINE_TIMEOUT and detections:
                det = detections[0]
                dist         = det['pnp_dist']    if det['pnp_dist']    is not None else \
                               (det['dist_est']   if det['dist_est']    else 2.0)
                angle_h_used = det['pnp_angle_h'] if det['pnp_angle_h'] is not None else det['angle_h']
                angle_v_used = det['pnp_angle_v'] if det['pnp_angle_v'] is not None else det['angle_v']
                gate_direction = yaw - angle_h_used
                gate_x = pos[0] + dist * np.cos(gate_direction)
                gate_y = pos[1] + dist * np.sin(gate_direction)
                gate_z = pos[2] + dist * np.tan(angle_v_used)
                gate_z = np.clip(gate_z, 0.5, 2.5)
                self.gate_rough_pos = np.array([gate_x, gate_y, gate_z])
                self.lost_count = 0
                self.last_good_yaw = yaw
                self.scan_frames = 0
                print(f"[STATE] Timeout refine → approach (fallback contour partiel "
                      f"{det['corner_count']} coins, aire={det['area']:.0f})")
                self.state = 'approach'

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
                CLOSE_THRESHOLD_H = 100
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
                MAX_LOST_FRAMES = 500

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
                    self.lap2_started   = False  # devra d'abord revenir au décollage
                    self.trajectory     = None   # sera calculée après retour au home
                    self.traj_start_time = None
                    print(f"[STATE] {NUM_GATES} gates trouvés et réordonnés → lap2 (retour décollage)")
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
                new_yaw = self._normalize_angle(yaw - 10 * dt)
                return [pos[0], pos[1], CRUISE_ALT, new_yaw]
            else:
                # Plus de magenta → le gate courant est derrière nous
                print("[STATE] Champ clair → scan_ccw (gate suivant)")
                self.state = 'scan_ccw'
                return [pos[0], pos[1], CRUISE_ALT, yaw]

        # =====================================================================
        # LAP2 : trajectoire polynomiale min-snap à travers les gates, 2 tours
        #   - Phase 0 : retourner au pad de décollage avant de démarrer
        #   - Phase 1 : calculer la trajectoire min-snap une seule fois
        #               (polynômes d'ordre 7 continus C³ à chaque gate)
        #   - Phase 2 : suivre la trajectoire en évaluant pos(t) à chaque tick
        # =====================================================================
        if self.state == 'lap2':

            # -- Phase 0 : retourner au pad de décollage avant de démarrer --
            if not self.lap2_started:
                home = self.takeoff_pos_recorded if self.takeoff_pos_recorded is not None \
                       else TAKEOFF_POS
                dx_h = home[0] - pos[0]
                dy_h = home[1] - pos[1]
                dist_home = np.linalg.norm([dx_h, dy_h])
                if dist_home < 0.4:
                    self.lap2_started = True
                    print("[LAP2] Pad de décollage atteint → calcul trajectoire")
                else:
                    target_yaw = np.arctan2(dy_h, dx_h)
                    return [float(home[0]), float(home[1]), CRUISE_ALT, target_yaw]

            # -- Phase 1 : calculer la trajectoire min-snap (une seule fois) --
            if self.trajectory is None:
                start_pos = np.array([pos[0], pos[1], CRUISE_ALT])
                waypoints = self._build_lap2_waypoints_shrunk(
                    self.gate_positions, start_pos,
                    lateral_offset=0.08   # offset (m) vers la droite du gate.
                                         # 0.0 = centroïde exact.
                                         # Augmenter (ex: 0.05) si le drone
                                         # coupe les virages par la gauche.
                )
                end_pos = start_pos.copy()   # retour au point de départ après 2 laps
                self.trajectory = self._compute_trajectory(
                    start_pos, waypoints, end_pos,
                    avg_speed=1.5,             # vitesse cruise (m/s) — à tuner
                    gate_speed_factor=0.6      # 60% de avg_speed au centroïde des gates
                                               # → traversée lente et précise
                )
                self.traj_start_time = sensor_data.get('t', 0.0)
                # Plot de la trajectoire (une seule fois)
                self._plot_trajectory(self.trajectory, self.gate_positions,
                                      waypoints, start_pos, end_pos,
                                      filename='lap2_trajectory.png')

            # -- Phase 2 : suivre la trajectoire --
            t_now   = sensor_data.get('t', 0.0)
            t_local = t_now - self.traj_start_time

            # Fin de trajectoire
            if t_local >= self.trajectory['total_time']:
                print("[STATE] Trajectoire min-snap terminée → finished")
                self.state = 'finished'
                return [pos[0], pos[1], CRUISE_ALT, yaw]

            # Évaluer pos(t) et vel(t) pour ce tick
            target_pos, target_vel = self._sample_trajectory(self.trajectory, t_local)

            # Yaw = direction du mouvement (tangent à la trajectoire)
            vxy_norm = np.linalg.norm(target_vel[:2])
            if vxy_norm > 0.1:
                target_yaw = np.arctan2(target_vel[1], target_vel[0])
            else:
                target_yaw = yaw  # vitesse trop faible : garder le cap actuel

            return [float(target_pos[0]), float(target_pos[1]),
                    float(target_pos[2]), float(target_yaw)]

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