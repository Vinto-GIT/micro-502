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

GATE_REAL_HEIGHT = 0.4   # mètres
GATE_REAL_WIDTH  = 0.4

CAM_K = np.array([[CAM_FOCAL, 0.0,       CAM_WIDTH  / 2.0],
                  [0.0,       CAM_FOCAL, CAM_HEIGHT / 2.0],
                  [0.0,       0.0,       1.0             ]], dtype=np.float32)

_hw = GATE_REAL_WIDTH  / 2.0
_hh = GATE_REAL_HEIGHT / 2.0
GATE_OBJ_POINTS = np.array([[-_hw, -_hh, 0.0],
                             [ _hw, -_hh, 0.0],
                             [ _hw,  _hh, 0.0],
                             [-_hw,  _hh, 0.0]], dtype=np.float32)

# =============================================================================
# PARAMÈTRES FILTRE HSV
# Plage unique robuste — dès qu'un blob est détecté ici, c'est forcément une gate.
# Aucun filtre de forme supplémentaire : tout magenta ≥ MIN_CONTOUR_AREA = gate.
# =============================================================================
HSV_LOWER_MAG1 = np.array([138,  20, 110])
HSV_UPPER_MAG1 = np.array([158, 210, 255])

# Seuil bas (avant 20) pour capter aussi les gates vues quasi de profil :
# leurs blobs sont fins/petits mais doivent encore déclencher la chaîne
# refine→approach→manœuvre perp pour qu'on ait une chance de les approcher.
MIN_CONTOUR_AREA = 10

NUM_GATES   = 5
CRUISE_ALT  = 1.5
TAKEOFF_POS = np.array([1.0, 4.0])
CIRCUIT_CENTER = np.array([4.0, 4.0])

# =============================================================================
# VITESSES DE ROTATION (rad/s, multipliées par dt)
# Encore réduites par rapport à la version précédente : on a > 60s de marge
# sur les 240s du run, donc on privilégie la robustesse aux gates
# perpendiculaires/lointaines plutôt que la vitesse pure.
# =============================================================================
YAW_RATE_SCAN   = 2.0   # rad/s — scan_ccw rotate (avant 4.0) — pas plus petits par cycle
YAW_RATE_REFINE = 2.5   # rad/s — refine     (avant 4.0) — moins de risque de dépasser le centre
YAW_RATE_CLEAR  = 6.0   # rad/s — scan_clear (CW pour s'éloigner du gate passé)


# =============================================================================
# HELPER : tri des 4 coins en ordre TL → TR → BR → BL
# =============================================================================
def order_corners_2d(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]   # TL
    ordered[1] = pts[np.argmin(d)]   # TR
    ordered[2] = pts[np.argmax(s)]   # BR
    ordered[3] = pts[np.argmax(d)]   # BL
    return ordered


# =============================================================================
# DÉTECTION HSV + solvePnP
# Pipeline : inRange → MORPH_CLOSE(3×3) → contours → area ≥ MIN_CONTOUR_AREA
# Pas de filtre de forme, taille ou isolement : tout magenta = gate.
# =============================================================================
def detect_gates(image):
    detections = []

    if image is None or image.size == 0:
        return detections, None, None

    if image.ndim == 3 and image.shape[2] == 4:
        img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        img = image.copy()

    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER_MAG1, HSV_UPPER_MAG1)

    # Fermeture minimale pour connecter les pixels proches d'un même blob.
    # Pas de kernel_open : on ne veut pas effacer les petits gates lointains.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug_mask   = np.zeros_like(img)
    angle_per_px = CAM_FOV_H / CAM_WIDTH

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        cv2.drawContours(debug_mask, [contour], -1, (255, 255, 255), 2)
        x, y, w, h = cv2.boundingRect(contour)

        M  = cv2.moments(contour)
        cx = M["m10"] / M["m00"] if M["m00"] != 0 else x + w / 2.0
        cy = M["m01"] / M["m00"] if M["m00"] != 0 else y + h / 2.0

        angle_h  = (cx - CAM_WIDTH  / 2.0) * angle_per_px
        angle_v  = (CAM_HEIGHT / 2.0 - cy) * (CAM_FOV_V / CAM_HEIGHT)
        dist_est = (GATE_REAL_HEIGHT * CAM_FOCAL) / h if h > 0 else None

        perimeter     = cv2.arcLength(contour, closed=True)
        approx        = cv2.approxPolyDP(contour, 0.04 * perimeter, closed=True)
        corner_count  = len(approx)
        has_4_corners = (corner_count == 4)

        pnp_dist = pnp_tvec = pnp_angle_h = pnp_angle_v = None
        if has_4_corners:
            ordered = order_corners_2d(approx.reshape(4, 2))
            corner_colors = [(0, 255, 255), (0, 165, 255), (0, 0, 255), (255, 0, 255)]
            for k, pt in enumerate(ordered):
                cv2.circle(debug_mask, tuple(pt.astype(int)), 6, corner_colors[k], -1)
            success, rvec, tvec = cv2.solvePnP(
                GATE_OBJ_POINTS, ordered, CAM_K, None,
                flags=cv2.SOLVEPNP_IPPE)
            if success:
                t = tvec.flatten()
                pnp_dist    = float(np.linalg.norm(t))
                pnp_tvec    = t
                pnp_angle_h = float(np.arctan2(t[0], t[2]))
                pnp_angle_v = float(np.arctan2(-t[1], t[2]))

        cv2.circle(debug_mask, (int(cx), int(cy)), 5, (0, 255, 0), -1)

        detections.append({
            'cx': cx, 'cy': cy,
            'x': x, 'y': y, 'w': w, 'h': h,
            'area': area,
            'angle_h': angle_h, 'angle_v': angle_v,
            'dist_est': dist_est,
            'has_4_corners': has_4_corners,
            'corner_count': corner_count,
            'corners': approx if has_4_corners else None,
            'pnp_dist': pnp_dist, 'pnp_tvec': pnp_tvec,
            'pnp_angle_h': pnp_angle_h, 'pnp_angle_v': pnp_angle_v,
        })

    detections.sort(key=lambda d: d['area'], reverse=True)
    return detections, debug_mask, mask


# =============================================================================
# ASSIGNMENT
# =============================================================================
class MyAssignment:
    def __init__(self):
        self.state      = 'takeoff'
        self.debug_mask = None

        self.yaw_90_target = None

        # Suivi scan/refine
        self.best_area            = 0.0
        self.best_angle_h         = None
        self.refine_shrink_count  = 0
        self.scan_frames          = 0

        # Stop-and-detect : alterne rotation et pause pour stabiliser l'image
        self.scan_phase        = 'rotate'
        self.scan_phase_frames = 0

        # Stabilisation à l'entrée de scan_ccw : absorbe l'inertie résiduelle
        # de scan_clear (qui tournait CW). Sans ça le drone "rebondit" et
        # tourne d'abord lentement avant d'atteindre la vitesse cible.
        self.scan_just_entered  = True
        self.scan_settle_frames = 0

        # Drift latéral : si scan_ccw fait un tour complet sans rien détecter,
        # on déplace le drone à gauche (0.4m) puis à droite (0.4m) pour
        # changer l'angle de vue. C'est ce qui sauve les gates perpendiculaires
        # à la sortie d'une autre gate : depuis la position d'arrivée, elles
        # sont vues quasi de profil et leur blob est trop fin pour être
        # détectable de façon stable. Un déport latéral révèle leur face.
        self.scan_origin_pos     = None    # position au début du scan (centre du drift)
        self.scan_origin_yaw     = 0.0     # yaw au début du scan (référence latérale)
        self.scan_total_rotation = 0.0     # rad CCW accumulés depuis le dernier reset
        self.scan_yaw_prev       = 0.0     # yaw frame précédente (pour calcul delta)
        self.scan_lateral_phase  = 0       # 0=centre, 1=gauche, 2=droite

        # Position estimée du gate courant (lissée exponentiellement)
        self.gate_rough_pos = None
        self.lost_count     = 0
        self.last_good_yaw  = None

        # Manœuvre "gate perpendiculaire" : déplacement déterministe (~5s)
        # vers la droite (1ère tentative) puis la gauche (2ème tentative)
        # de la gate, avec yaw figé pointant vers la gate. Si les deux essais
        # ne révèlent pas la gate, on retombe sur l'approche normale avec
        # l'estimation de position courante.
        self.perp_active        = False
        self.perp_frames        = 0
        self.perp_target        = None    # np.array(3) : point cible (figé)
        self.perp_lock_dir      = None    # rad : yaw figé vers la gate
        self.perp_attempts      = 0       # 0=aucune, 1=droite faite, 2=gauche faite, 99=bloqué

        # Gates détectés en lap1
        self.gate_positions   = []
        self.current_gate_idx = 0

        # Trajectoire lap2
        self.lap2_gate_idx  = 0
        self.lap2_lap_count = 0
        self.trajectory      = None
        self.traj_start_time = None
        self.last_time       = 0.0

        self.takeoff_pos_recorded = None

    def _reset_for_next_gate(self):
        """Réinitialise les variables de tracking pour le gate suivant."""
        self.gate_rough_pos       = None
        self.lost_count           = 0
        self.last_good_yaw        = None
        self.best_area            = 0.0
        self.best_angle_h         = None
        self.refine_shrink_count  = 0
        self.scan_frames          = 0
        self.scan_phase           = 'rotate'
        self.scan_phase_frames    = 0
        self.scan_just_entered    = True
        self.scan_settle_frames   = 0
        # Reset du drift latéral : on recommence le scan complet pour le gate suivant
        self.scan_origin_pos      = None
        self.scan_origin_yaw      = 0.0
        self.scan_total_rotation  = 0.0
        self.scan_yaw_prev        = 0.0
        self.scan_lateral_phase   = 0
        self.perp_active          = False
        self.perp_frames          = 0
        self.perp_target          = None
        self.perp_lock_dir        = None
        self.perp_attempts        = 0

    def _closest_detection_to_target(self, detections, pos, yaw):
        if not detections:
            return None
        if self.gate_rough_pos is None:
            return detections[0]
        best_det  = None
        best_dist = float('inf')
        for d in detections:
            dist_d  = d['pnp_dist']    if d['pnp_dist']    is not None else \
                      (d['dist_est']   if d['dist_est']     else 2.0)
            angle_h = d['pnp_angle_h'] if d['pnp_angle_h'] is not None else d['angle_h']
            gx = pos[0] + dist_d * np.cos(yaw - angle_h)
            gy = pos[1] + dist_d * np.sin(yaw - angle_h)
            d2t = np.linalg.norm([gx - self.gate_rough_pos[0],
                                   gy - self.gate_rough_pos[1]])
            if d2t < best_dist:
                best_dist = d2t
                best_det  = d
        return best_det if best_dist < 1.5 else None

    def _estimate_gate_normal(self, gate_pos, prev_pos, next_pos):
        dir_in  = np.array(gate_pos[:2]) - np.array(prev_pos[:2])
        dir_out = np.array(next_pos[:2]) - np.array(gate_pos[:2])
        d_in  = dir_in  / (np.linalg.norm(dir_in)  + 1e-9)
        d_out = dir_out / (np.linalg.norm(dir_out) + 1e-9)
        tangent = (d_in + d_out) / 2.0
        n = np.linalg.norm(tangent)
        return tangent / n if n > 1e-6 else d_out

    # =========================================================================
    # TRAJECTOIRE MIN-SNAP (polynômes d'ordre 7, continus C³)
    # =========================================================================
    def _compute_min_snap_segment(self, p0, v0, a0, j0, p1, v1, a1, j1, T):
        A = np.array([
            [1, 0, 0, 0,      0,      0,       0,       0      ],
            [0, 1, 0, 0,      0,      0,       0,       0      ],
            [0, 0, 2, 0,      0,      0,       0,       0      ],
            [0, 0, 0, 6,      0,      0,       0,       0      ],
            [1, T, T**2, T**3, T**4,   T**5,    T**6,    T**7   ],
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4, 6*T**5, 7*T**6 ],
            [0, 0, 2,   6*T,  12*T**2, 20*T**3, 30*T**4, 42*T**5],
            [0, 0, 0,   6,    24*T,   60*T**2, 120*T**3, 210*T**4],
        ], dtype=np.float64)
        b = np.array([p0, v0, a0, j0, p1, v1, a1, j1], dtype=np.float64)
        return np.linalg.solve(A, b)

    def _eval_poly(self, coeffs, t):
        t_pow  = np.array([t**k for k in range(8)])
        dt_pow = np.array([k * t**(k-1) if k >= 1 else 0 for k in range(8)])
        return (np.einsum('k,kd->d', t_pow,  coeffs),
                np.einsum('k,kd->d', dt_pow, coeffs))

    def _compute_trajectory(self, start_pos, waypoints, end_pos,
                            avg_speed=1.2, gate_speed_factor=0.6):
        all_pts = [{'pos': start_pos, 'normal': None}] + waypoints + \
                  [{'pos': end_pos,   'normal': None}]
        N          = len(all_pts) - 1
        gate_speed = avg_speed * gate_speed_factor

        durations = [float(np.clip(
            np.linalg.norm(all_pts[i+1]['pos'] - all_pts[i]['pos']) / avg_speed,
            0.3, 5.0)) for i in range(N)]

        velocities = [np.zeros(3)]
        for i in range(1, N):
            wp = all_pts[i]
            if wp['normal'] is not None:
                n = wp['normal']
                velocities.append(np.array([n[0]*gate_speed, n[1]*gate_speed, 0.0]))
            else:
                dp = all_pts[i+1]['pos'] - all_pts[i-1]['pos']
                velocities.append(dp / (np.linalg.norm(dp) + 1e-9) * gate_speed)
        velocities.append(np.zeros(3))

        zeros = [np.zeros(3)] * (N + 1)
        polys = [self._compute_min_snap_segment(
                     all_pts[i]['pos'], velocities[i],   zeros[i],   zeros[i],
                     all_pts[i+1]['pos'], velocities[i+1], zeros[i+1], zeros[i+1],
                     durations[i]) for i in range(N)]

        total = sum(durations)
        print(f"[TRAJ] {N} seg, {total:.1f}s, cruise={avg_speed:.1f}m/s, "
              f"gate={gate_speed:.1f}m/s")
        return {'polys': polys, 'durations': durations, 'total_time': total}

    def _sample_trajectory(self, traj, t):
        if t >= traj['total_time']:
            pos, _ = self._eval_poly(traj['polys'][-1], traj['durations'][-1])
            return pos, np.zeros(3)
        t_loc = t
        for i, dur in enumerate(traj['durations']):
            if t_loc <= dur:
                return self._eval_poly(traj['polys'][i], t_loc)
            t_loc -= dur
        pos, _ = self._eval_poly(traj['polys'][-1], traj['durations'][-1])
        return pos, np.zeros(3)

    def _build_lap2_waypoints_shrunk(self, ordered_gates, start_pos,
                                     lateral_offset=0.15):
        n = len(ordered_gates)
        base = []
        for i, g in enumerate(ordered_gates):
            prev = ordered_gates[i - 1] if i > 0 else start_pos
            nxt  = ordered_gates[(i + 1) % n]
            normal  = self._estimate_gate_normal(g, prev, nxt)
            right2d = np.array([normal[1], -normal[0]])
            g_target = np.array(g, dtype=np.float64).copy()
            g_target[0] += lateral_offset * right2d[0]
            g_target[1] += lateral_offset * right2d[1]
            base.append({'pos': g_target, 'normal': normal})
        return base + base

    def _plot_trajectory(self, traj, ordered_gates, waypoints, start_pos, end_pos,
                         filename='lap2_trajectory.png'):
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa
        except ImportError:
            print("[PLOT] matplotlib non disponible — ignoré")
            return
        pts  = np.linspace(0, traj['total_time'], 500)
        path = np.array([self._sample_trajectory(traj, t)[0] for t in pts])
        fig  = plt.figure(figsize=(10, 8))
        ax   = fig.add_subplot(111, projection='3d')
        ax.plot(path[:,0], path[:,1], path[:,2], 'k-', lw=2, label='Min-snap')
        for i, g in enumerate(ordered_gates):
            n  = waypoints[i]['normal'] if i < len(waypoints) else np.array([1., 0.])
            tg = np.array([-n[1], n[0], 0.])
            up = np.array([0., 0., 1.])
            c  = np.array([g+tg*.2+up*.2, g-tg*.2+up*.2,
                           g-tg*.2-up*.2, g+tg*.2-up*.2, g+tg*.2+up*.2])
            ax.plot(c[:,0], c[:,1], c[:,2], 'r-', lw=2)
            ax.text(g[0], g[1], g[2]+.15, f'G{i+1}', fontsize=9, ha='center')
        ax.scatter(*start_pos, c='green', s=100, marker='o', label='Start')
        ax.scatter(*end_pos,   c='red',   s=100, marker='X', label='End')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f"Min-snap {traj['total_time']:.1f}s")
        ax.legend()
        try:
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            print(f"[PLOT] → {filename}")
            plt.close(fig)
        except Exception as e:
            print(f"[PLOT] Erreur : {e}")

    # =========================================================================
    # COMMANDE PRINCIPALE
    # =========================================================================
    def compute_command(self, sensor_data, camera_data, dt):
        pos = np.array([sensor_data['x_global'],
                        sensor_data['y_global'],
                        sensor_data['z_global']])
        yaw = sensor_data['yaw']

        # =====================================================================
        # TAKEOFF
        # =====================================================================
        if self.state == 'takeoff':
            if self.takeoff_pos_recorded is None:
                self.takeoff_pos_recorded = pos[:2].copy()
                print(f"[TAKEOFF] pad={self.takeoff_pos_recorded}")
                print(f"[MONITOR] RUN_START t={sensor_data.get('t',0.0):.2f}")
            if sensor_data['z_global'] < 1.4:
                return [pos[0], pos[1], CRUISE_ALT, yaw]
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
                self.scan_just_entered = True
                self.scan_settle_frames = 0
                print("[STATE] Rotation atteinte → scan_ccw")
            return [pos[0], pos[1], CRUISE_ALT, self.yaw_90_target]

        # =====================================================================
        # SCAN_CCW : rotation CCW stop-and-detect + drift latéral en fallback
        #
        # Étape 0 : stabilisation (SCAN_SETTLE_FRAMES) à l'entrée pour absorber
        #           l'inertie résiduelle de scan_clear (qui tournait CW).
        # Étape 1 : alterner rotation lente (SCAN_ROT_FRAMES) et pause longue
        #           (SCAN_PAUSE_FRAMES). Détection uniquement pendant la pause.
        # Étape 2 : si 1 tour complet sans détection, drift latéral (gauche
        #           LATERAL_OFFSET_M, puis droite LATERAL_OFFSET_M). Le drone
        #           continue de tourner CCW pendant que la position drift
        #           doucement vers la cible latérale. Critique pour les gates
        #           perpendiculaires invisibles depuis la position d'arrivée
        #           d'une gate précédente.
        # =====================================================================
        if self.state == 'scan_ccw':
            SCAN_SETTLE_FRAMES = 25     # ~0.5s à 50Hz : anti-inertie scan_clear→scan_ccw
            SCAN_ROT_FRAMES    = 6      # rotation très lente — pas plus petits, moins de chances de manquer une gate entre 2 pauses
            SCAN_PAUSE_FRAMES  = 25     # pause longue — détection fiable des gates lointaines/edge-on
            LATERAL_OFFSET_M   = 0.40   # m : amplitude du drift latéral (gauche/droite)
            LATERAL_SPEED      = 0.10   # m/s : drift très doux, yaw continue de tourner pendant
            FULL_ROTATION_RAD  = 2*np.pi + 0.3   # 1 tour complet + petite marge

            # ----- Phase 0 : stabilisation à l'entrée (rester immobile au yaw courant) -----
            if self.scan_just_entered:
                if self.scan_settle_frames == 0:
                    # Première frame en scan_ccw : enregistrer position+yaw
                    # comme référence pour les drifts latéraux ultérieurs.
                    self.scan_origin_pos     = pos[:2].copy()
                    self.scan_origin_yaw     = yaw
                    self.scan_total_rotation = 0.0
                    self.scan_yaw_prev       = yaw
                    self.scan_lateral_phase  = 0
                self.scan_settle_frames += 1
                if self.scan_settle_frames >= SCAN_SETTLE_FRAMES:
                    self.scan_just_entered = False
                    self.scan_phase = 'rotate'
                    self.scan_phase_frames = 0
                    self.scan_frames = 0
                    print(f"[SCAN] Stabilisation OK ({self.scan_settle_frames}f)")
                return [pos[0], pos[1], CRUISE_ALT, yaw]

            self.scan_frames       += 1
            self.scan_phase_frames += 1

            # ----- Suivi de la rotation totale CCW depuis le début du tour courant -----
            delta_yaw = self._angle_diff(yaw, self.scan_yaw_prev)
            if delta_yaw > 0:   # ne compte que la rotation CCW (positive)
                self.scan_total_rotation += delta_yaw
            self.scan_yaw_prev = yaw

            # ----- Si 1 tour complet sans détection : passer à la phase latérale suivante -----
            if self.scan_total_rotation > FULL_ROTATION_RAD:
                if self.scan_lateral_phase < 2:
                    self.scan_lateral_phase += 1
                    self.scan_total_rotation = 0.0
                    phase_name = ['centre', 'gauche', 'droite'][self.scan_lateral_phase]
                    print(f"[SCAN] Tour complet sans détection → drift latéral '{phase_name}'")
                else:
                    # Tous les drifts essayés : continuer à tourner sur place
                    self.scan_total_rotation = 0.0

            # ----- Calcul de la position cible (selon la phase latérale courante) -----
            # left_dir = perpendiculaire gauche au yaw d'entrée du scan
            left_dir = np.array([-np.sin(self.scan_origin_yaw),
                                  np.cos(self.scan_origin_yaw)])
            if self.scan_lateral_phase == 1:
                lateral_target = self.scan_origin_pos + LATERAL_OFFSET_M * left_dir
            elif self.scan_lateral_phase == 2:
                lateral_target = self.scan_origin_pos - LATERAL_OFFSET_M * left_dir
            else:
                lateral_target = self.scan_origin_pos

            # Drift doux vers la cible (sans saut)
            to_target = lateral_target - pos[:2]
            dist_to_target = np.linalg.norm(to_target)
            if dist_to_target > 0.03:
                step = min(LATERAL_SPEED, dist_to_target)
                next_xy = pos[:2] + step * to_target / dist_to_target
            else:
                next_xy = pos[:2]

            # ----- Bascule rotate/pause -----
            if self.scan_phase == 'rotate' and self.scan_phase_frames >= SCAN_ROT_FRAMES:
                self.scan_phase        = 'pause'
                self.scan_phase_frames = 0
            elif self.scan_phase == 'pause' and self.scan_phase_frames >= SCAN_PAUSE_FRAMES:
                self.scan_phase        = 'rotate'
                self.scan_phase_frames = 0

            # ----- Détection (uniquement pendant la pause stable) -----
            if self.scan_phase == 'pause':
                detections, self.debug_mask, _ = detect_gates(camera_data)
                if detections:
                    candidates_4 = [d for d in detections if d['has_4_corners']]
                    det = candidates_4[0] if candidates_4 else detections[0]
                    self.best_area    = det['area']
                    self.best_angle_h = det['angle_h']
                    self.refine_shrink_count = 0
                    self.state = 'refine'
                    self.scan_frames = 0
                    self.scan_phase = 'rotate'
                    self.scan_phase_frames = 0
                    print(f"[STATE] Gate détecté (pause, drift_phase={self.scan_lateral_phase}) "
                          f"aire={det['area']:.0f} coins={det['corner_count']} → refine")

            # ----- Sortie : envoyer le setpoint -----
            # Position : suit le drift latéral (revient à origin si phase 0).
            # Yaw     : tourne pendant 'rotate', figé pendant 'pause'.
            if self.state == 'scan_ccw':
                if self.scan_phase == 'rotate':
                    new_yaw = self._normalize_angle(yaw + YAW_RATE_SCAN * dt)
                    return [float(next_xy[0]), float(next_xy[1]), CRUISE_ALT, new_yaw]
                else:
                    return [float(next_xy[0]), float(next_xy[1]), CRUISE_ALT, yaw]

        # =====================================================================
        # REFINE
        # =====================================================================
        if self.state == 'refine':
            detections, self.debug_mask, _ = detect_gates(camera_data)
            self.scan_frames += 1

            candidates_4 = [d for d in detections if d['has_4_corners']]
            det = candidates_4[0] if candidates_4 else (detections[0] if detections else None)

            if det is not None:
                dist         = det['pnp_dist']    if det['pnp_dist']    is not None else \
                               (det['dist_est']   if det['dist_est']    else 2.0)
                angle_h_used = det['pnp_angle_h'] if det['pnp_angle_h'] is not None else det['angle_h']
                angle_v_used = det['pnp_angle_v'] if det['pnp_angle_v'] is not None else det['angle_v']
                gate_dir = yaw - angle_h_used
                gate_x   = pos[0] + dist * np.cos(gate_dir)
                gate_y   = pos[1] + dist * np.sin(gate_dir)
                gate_z   = np.clip(pos[2] + dist * np.tan(angle_v_used), 0.5, 2.5)

                # Gate perpendiculaire détectée en refine (blob très petit/allongé) :
                # tourner sur place ne sert à rien, on perd le blob à chaque rotation.
                # → passer directement en approach pour déclencher la manœuvre perp douce.
                # Critère : aire petite OU aspect non carré (vue de biais probable).
                w, h    = det['w'], det['h']
                aspect  = min(w, h) / max(w, h, 1)
                is_perp = (det['area'] < 100) or (aspect < 0.6)

                REFINE_TIMEOUT  = 200
                ANGLE_THRESHOLD = 0.15

                if is_perp:
                    self.gate_rough_pos = np.array([gate_x, gate_y, gate_z])
                    self.lost_count     = 0
                    self.last_good_yaw  = yaw
                    self.scan_frames    = 0
                    print(f"[STATE] Gate perpendiculaire en refine "
                          f"(aire={det['area']:.0f}, aspect={aspect:.2f}) → approach (manœuvre perp)")
                    self.state = 'approach'
                elif abs(angle_h_used) < ANGLE_THRESHOLD or self.scan_frames > REFINE_TIMEOUT:
                    self.gate_rough_pos = np.array([gate_x, gate_y, gate_z])
                    self.lost_count     = 0
                    self.last_good_yaw  = yaw
                    self.scan_frames    = 0
                    reason = "centré" if abs(angle_h_used) < ANGLE_THRESHOLD else "timeout"
                    print(f"[STATE] Gate estimé ({reason}) @ {np.round(self.gate_rough_pos,2)} "
                          f"dist={dist:.2f}m → approach")
                    self.state = 'approach'
                else:
                    new_yaw = self._normalize_angle(yaw + YAW_RATE_REFINE * dt)
                    return [pos[0], pos[1], CRUISE_ALT, new_yaw]
            else:
                if self.scan_frames > 30:
                    print("[STATE] Blob perdu en refine → scan_ccw")
                    self._reset_for_next_gate()
                    self.state = 'scan_ccw'
                new_yaw = self._normalize_angle(yaw + YAW_RATE_REFINE * dt)
                return [pos[0], pos[1], CRUISE_ALT, new_yaw]

        # =====================================================================
        # APPROACH : servo visuel + gestion gate perpendiculaire
        #
        # Si la gate visible n'est pas carrée (ratio w/h < SQUARE_THRESH),
        # on déclenche une manœuvre DÉTERMINISTE de ~5s :
        #   - 1ère tentative : à droite de la gate + un peu en arrière
        #   - 2ème tentative : à gauche de la gate + un peu en arrière
        #   - yaw figé : pointe vers la gate (calculé une seule fois)
        # → pas d'oscillation pendant la manœuvre car aucune valeur n'est
        #   recalculée à partir de l'image instable.
        # =====================================================================
        if self.state == 'approach':
            detections, self.debug_mask, _ = detect_gates(camera_data)

            # ---- Manœuvre perpendiculaire en cours : suivre cible figée ----
            PERP_DURATION_FRAMES = 250   # ~5s à 50Hz (avant 150) — le drift est doux, il faut le temps d'arriver
            PERP_AREA_OK         = 300   # px² : si aire dépasse → la manœuvre a réussi à révéler la gate
            if self.perp_active:
                self.perp_frames += 1

                # Vérifier si la gate est maintenant plus visible (aire augmentée)
                # Si oui → la manœuvre a servi, repasser en approche normale.
                if detections:
                    best = detections[0]
                    if best['area'] > PERP_AREA_OK:
                        self.perp_active = False
                        # Marquer cette tentative comme conclusive : on n'en redéclenche pas
                        self.perp_attempts = 99
                        print(f"[PERP] Gate visible (aire={best['area']:.0f}) "
                              f"→ fin manœuvre, approche normale")
                        # Ne pas return ici → on retombe dans l'approche normale

                # Timeout : la cible courante n'a pas révélé la gate
                if self.perp_active and self.perp_frames >= PERP_DURATION_FRAMES:
                    self.perp_active = False
                    if self.perp_attempts == 1:
                        # 1ère tentative (droite) terminée sans succès → on tentera la gauche
                        # à la prochaine détection de gate perpendiculaire (logique plus bas).
                        print("[PERP] Tentative droite échouée → retour approche, prêt pour tentative gauche")
                    else:
                        # 2ème tentative (gauche) terminée → plus rien à faire, on reste en approche normale
                        self.perp_attempts = 99
                        print("[PERP] Les deux côtés essayés → approche normale forcée")

                # Avancer DOUCEMENT vers la cible (pas un saut)
                if self.perp_active:
                    PERP_SPEED = 0.18   # m/s (avant 0.15) — un peu plus rapide pour atteindre la cible dans les 5s
                    dx = self.perp_target[0] - pos[0]
                    dy = self.perp_target[1] - pos[1]
                    dist_to_target = np.linalg.norm([dx, dy])
                    if dist_to_target > 0.05:
                        step = min(PERP_SPEED, dist_to_target)
                        return [pos[0] + step * dx / dist_to_target,
                                pos[1] + step * dy / dist_to_target,
                                float(self.perp_target[2]),
                                float(self.perp_lock_dir)]
                    else:
                        # Cible atteinte avant timeout → rester sur place et continuer à observer
                        return [float(self.perp_target[0]),
                                float(self.perp_target[1]),
                                float(self.perp_target[2]),
                                float(self.perp_lock_dir)]

            det = self._closest_detection_to_target(detections, pos, yaw)

            if det is not None:
                self.lost_count    = 0
                self.last_good_yaw = yaw

                dist         = det['pnp_dist']    if det['pnp_dist']    is not None else \
                               (det['dist_est']   if det['dist_est']    else 2.0)
                angle_h_used = det['pnp_angle_h'] if det['pnp_angle_h'] is not None else det['angle_h']
                angle_v_used = det['pnp_angle_v'] if det['pnp_angle_v'] is not None else det['angle_v']

                gate_dir = yaw - angle_h_used
                gate_x   = pos[0] + dist * np.cos(gate_dir)
                gate_y   = pos[1] + dist * np.sin(gate_dir)
                gate_z   = np.clip(pos[2] + dist * np.tan(angle_v_used), 0.5, 2.5)

                # Lissage exponentiel de la position estimée du gate
                alpha = 0.3
                if self.gate_rough_pos is not None:
                    self.gate_rough_pos = (alpha * np.array([gate_x, gate_y, gate_z])
                                         + (1 - alpha) * self.gate_rough_pos)
                else:
                    self.gate_rough_pos = np.array([gate_x, gate_y, gate_z])

                # ---- Test gate perpendiculaire (déclenchement manœuvre 5s) ----
                w, h          = det['w'], det['h']
                aspect        = min(w, h) / max(w, h, 1)
                SQUARE_THRESH = 0.6

                # On déclenche tant qu'on n'a pas épuisé les 2 tentatives (perp_attempts < 2).
                # 0 → on essaie la droite, 1 → on essaie la gauche, 99 → bloqué (déjà géré).
                if aspect < SQUARE_THRESH and self.perp_attempts < 2:
                    # Verrouiller la cible et la direction (calculées UNE FOIS)
                    g  = self.gate_rough_pos
                    dx = g[0] - pos[0]
                    dy = g[1] - pos[1]
                    self.perp_lock_dir = float(np.arctan2(dy, dx))

                    # Repère "drone vers gate" :
                    #   forward = vers la gate
                    #   right   = à droite (perpendiculaire)
                    fdir  = self.perp_lock_dir
                    fwd   = np.array([np.cos(fdir), np.sin(fdir)])
                    right = np.array([np.sin(fdir), -np.cos(fdir)])

                    # Choix du côté selon la tentative :
                    #   1ère tentative (perp_attempts=0) → droite (+RIGHT_OFFSET)
                    #   2ème tentative (perp_attempts=1) → gauche (-RIGHT_OFFSET)
                    LATERAL_OFFSET_PERP = 0.5    # m — atteignable dans 5s
                    BACK_OFFSET_PERP    = 0.3    # m — recul modeste pour avoir une vue plus large
                    side_sign = 1.0 if self.perp_attempts == 0 else -1.0
                    side_name = "droite" if self.perp_attempts == 0 else "gauche"

                    target_xy = np.array([g[0], g[1]]) \
                              + side_sign * LATERAL_OFFSET_PERP * right \
                              - BACK_OFFSET_PERP * fwd
                    self.perp_target = np.array([target_xy[0], target_xy[1],
                                                  float(g[2])])
                    self.perp_active   = True
                    self.perp_frames   = 1
                    self.perp_attempts += 1
                    print(f"[PERP] Gate perpendiculaire (aspect={aspect:.2f}) "
                          f"→ tentative {self.perp_attempts}/2 ({side_name}) : "
                          f"cible={np.round(self.perp_target,2)}, "
                          f"yaw_lock={np.degrees(self.perp_lock_dir):.1f}°")
                    return [float(self.perp_target[0]),
                            float(self.perp_target[1]),
                            float(self.perp_target[2]),
                            float(self.perp_lock_dir)]

                # ---- Approche normale ----
                CLOSE_THRESHOLD_H = 80
                if det['h'] > CLOSE_THRESHOLD_H or (det['dist_est'] and det['dist_est'] < 0.6):
                    self.gate_positions.append(self.gate_rough_pos.copy())
                    self.current_gate_idx = len(self.gate_positions)
                    through_z    = float(self.gate_rough_pos[2])
                    approach_dir = yaw - angle_h_used
                    self.fly_through_target = np.array([
                        self.gate_rough_pos[0] + 0.8 * np.cos(approach_dir),
                        self.gate_rough_pos[1] + 0.8 * np.sin(approach_dir),
                        through_z])
                    self.fly_through_yaw = approach_dir
                    print(f"[STATE] Gate {len(self.gate_positions)}/{NUM_GATES} "
                          f"@ {np.round(self.gate_rough_pos,2)} → fly_through")
                    print(f"[MONITOR] GATE_DETECTED={len(self.gate_positions)} "
                          f"pos={np.round(self.gate_rough_pos,2).tolist()}")
                    self.state = 'fly_through'
                    return [pos[0], pos[1], through_z, yaw]

                # Centrer + avancer (vitesse réduite pour x1)
                APPROACH_SPEED = 0.30
                return [pos[0] + APPROACH_SPEED * np.cos(gate_dir),
                        pos[1] + APPROACH_SPEED * np.sin(gate_dir),
                        float(self.gate_rough_pos[2]),
                        self._normalize_angle(yaw - angle_h_used)]

            else:
                self.lost_count += 1
                MAX_LOST_FRAMES = 300
                if self.lost_count > MAX_LOST_FRAMES:
                    print(f"[STATE] Gate perdu ({self.lost_count}f) → scan_ccw")
                    print(f"[MONITOR] GATE_LOST at_gate={len(self.gate_positions)+1} "
                          f"pos={np.round(pos,2).tolist()}")
                    self._reset_for_next_gate()
                    self.state = 'scan_ccw'
                    return [pos[0], pos[1], CRUISE_ALT, yaw]

                if self.gate_rough_pos is not None:
                    dx   = self.gate_rough_pos[0] - pos[0]
                    dy   = self.gate_rough_pos[1] - pos[1]
                    dist = np.linalg.norm([dx, dy])
                    if dist > 0.1:
                        step = min(0.12, dist)
                        return [pos[0] + step*(dx/dist),
                                pos[1] + step*(dy/dist),
                                float(self.gate_rough_pos[2]),
                                np.arctan2(dy, dx)]
                return [pos[0], pos[1], CRUISE_ALT, yaw]

        # =====================================================================
        # FLY_THROUGH
        # =====================================================================
        if self.state == 'fly_through':
            dx   = self.fly_through_target[0] - pos[0]
            dy   = self.fly_through_target[1] - pos[1]
            dist = np.linalg.norm([dx, dy])

            if dist < 0.3:
                self._reset_for_next_gate()
                if len(self.gate_positions) >= NUM_GATES:
                    self.trajectory      = None
                    self.traj_start_time = None
                    print(f"[STATE] {NUM_GATES} gates → lap2 (démarrage immédiat)")
                    self.state = 'lap2'
                else:
                    print(f"[STATE] Gate traversé ({len(self.gate_positions)}"
                          f"/{NUM_GATES}) → scan_clear")
                    self.state = 'scan_clear'
                return [pos[0], pos[1],
                        self.fly_through_target[2], self.fly_through_yaw]

            step = min(0.4, dist)
            return [pos[0] + step*dx/dist,
                    pos[1] + step*dy/dist,
                    self.fly_through_target[2],
                    self.fly_through_yaw]

        # =====================================================================
        # SCAN_CLEAR : tourner CW jusqu'à plus de magenta
        # Puis repasser en scan_ccw avec stabilisation pour absorber l'inertie.
        # =====================================================================
        if self.state == 'scan_clear':
            if pos[2] > CRUISE_ALT + 0.1:
                return [pos[0], pos[1], CRUISE_ALT, yaw]

            detections, self.debug_mask, _ = detect_gates(camera_data)

            if detections:
                new_yaw = self._normalize_angle(yaw - YAW_RATE_CLEAR * dt)
                return [pos[0], pos[1], CRUISE_ALT, new_yaw]
            else:
                print("[STATE] Champ clair → scan_ccw (gate suivant)")
                self.state = 'scan_ccw'
                self.scan_just_entered  = True   # déclenche la stabilisation
                self.scan_settle_frames = 0
                return [pos[0], pos[1], CRUISE_ALT, yaw]

        # =====================================================================
        # LAP2 : trajectoire min-snap, 2 tours, démarrage immédiat
        # =====================================================================
        if self.state == 'lap2':
            if self.trajectory is None:
                start_pos = np.array([pos[0], pos[1], CRUISE_ALT])
                waypoints = self._build_lap2_waypoints_shrunk(
                    self.gate_positions, start_pos,
                    lateral_offset=0.15)

                last_n  = waypoints[-1]['normal']
                last_g  = self.gate_positions[-1]
                end_pos = np.array([last_g[0] + 1.0*last_n[0],
                                    last_g[1] + 1.0*last_n[1],
                                    CRUISE_ALT])

                self.trajectory = self._compute_trajectory(
                    start_pos, waypoints, end_pos,
                    avg_speed=1.2,
                    gate_speed_factor=0.6
                )
                self.traj_start_time = sensor_data.get('t', 0.0)
                print(f"[LAP2] Départ={np.round(pos[:2],2)}, "
                      f"total={self.trajectory['total_time']:.1f}s")
                print(f"[MONITOR] LAP2_START t={sensor_data.get('t',0.0):.2f} "
                      f"gates={[list(np.round(g,2)) for g in self.gate_positions]}")
                self._plot_trajectory(self.trajectory, self.gate_positions,
                                      waypoints, start_pos, end_pos)

            t_local = sensor_data.get('t', 0.0) - self.traj_start_time

            if t_local >= self.trajectory['total_time']:
                print("[STATE] Trajectoire terminée → finished")
                print(f"[MONITOR] LAP2_END t={sensor_data.get('t',0.0):.2f} "
                      f"duration={t_local:.2f}")
                self.state = 'finished'
                return [pos[0], pos[1], CRUISE_ALT, yaw]

            target_pos, target_vel = self._sample_trajectory(self.trajectory, t_local)
            vxy = np.linalg.norm(target_vel[:2])
            tgt_yaw = np.arctan2(target_vel[1], target_vel[0]) if vxy > 0.1 else yaw

            return [float(target_pos[0]), float(target_pos[1]),
                    float(target_pos[2]), float(tgt_yaw)]

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
    return {
        'gate_positions': list(_controller.gate_positions),
        'gate_rough_pos': _controller.gate_rough_pos,
        'state':          _controller.state,
    }