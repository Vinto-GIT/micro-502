"""
webots_monitor.py
=================
Script de monitoring automatique pour le projet drone racing Webots.

Usage :
    python3 webots_monitor.py --world /path/to/world.wbt [--runs 20] [--csv results.csv]

Ce qu'il fait :
    1. Lance Webots avec le monde spécifié
    2. Lit la sortie console en temps réel
    3. Détecte les tags [MONITOR] pour logger les événements
    4. Quand lap2 est terminé (ou timeout), relance le monde
    5. Écrit tout dans un CSV

Tags [MONITOR] reconnus dans my_assignment.py :
    [MONITOR] RUN_START t=<float>
    [MONITOR] GATE_DETECTED=<int> pos=<list>
    [MONITOR] GATE_LOST at_gate=<int> pos=<list>
    [MONITOR] LAP2_START t=<float> gates=<list>
    [MONITOR] LAP2_END t=<float> duration=<float>
"""

import subprocess
import csv
import time
import re
import argparse
import os
import sys
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================
WEBOTS_TIMEOUT_S   = 500    # timeout max par run (secondes réelles, pas sim)
                            # simulation tourne en moyenne à x0.5 → 500s réelles ≈ 250s sim
LAP2_WAIT_AFTER_S  = 5      # attendre Ns après LAP2_END avant de relancer
CSV_FIELDNAMES = [
    'run_id', 'timestamp', 'world',
    'run_start_t',          # temps sim au décollage
    'gates_detected',       # nombre de gates détectés en lap1 (0-5)
    'gate_positions',       # liste des positions [[x,y,z], ...]
    'gates_lost_count',     # nombre de fois où un gate a été perdu
    'lap2_start_t',         # temps sim au début lap2
    'lap2_end_t',           # temps sim à la fin lap2
    'lap2_duration_s',      # durée sim du lap2
    'lap2_started',         # True si lap2 a démarré
    'lap2_finished',        # True si trajectoire terminée normalement
    'timed_out',            # True si timeout réel atteint
    'notes',                # infos supplémentaires
]


# =============================================================================
# PARSING DES TAGS MONITOR
# =============================================================================
def parse_monitor_line(line):
    """Extrait les infos d'une ligne [MONITOR] xxx=yyy."""
    m = re.search(r'\[MONITOR\]\s+(.*)', line)
    if not m:
        return None, None

    content = m.group(1).strip()

    # RUN_START t=12.34
    m2 = re.match(r'RUN_START t=(\S+)', content)
    if m2:
        return 'RUN_START', {'t': float(m2.group(1))}

    # GATE_DETECTED=3 pos=[1.2, 3.4, 1.5]
    m2 = re.match(r'GATE_DETECTED=(\d+)\s+pos=(\[.*\])', content)
    if m2:
        return 'GATE_DETECTED', {
            'n': int(m2.group(1)),
            'pos': m2.group(2)
        }

    # GATE_LOST at_gate=2 pos=[x,y,z]
    m2 = re.match(r'GATE_LOST at_gate=(\d+)\s+pos=(\[.*\])', content)
    if m2:
        return 'GATE_LOST', {
            'at_gate': int(m2.group(1)),
            'pos': m2.group(2)
        }

    # LAP2_START t=45.1 gates=[[...], ...]
    m2 = re.match(r'LAP2_START t=(\S+)\s+gates=(\[.*)', content)
    if m2:
        return 'LAP2_START', {
            't': float(m2.group(1)),
            'gates': m2.group(2)
        }

    # LAP2_END t=88.3 duration=43.2
    m2 = re.match(r'LAP2_END t=(\S+)\s+duration=(\S+)', content)
    if m2:
        return 'LAP2_END', {
            't': float(m2.group(1)),
            'duration': float(m2.group(2))
        }

    return 'UNKNOWN', {'raw': content}


# =============================================================================
# ÉTAT D'UN RUN
# =============================================================================
class RunState:
    def __init__(self, run_id, world):
        self.run_id       = run_id
        self.world        = world
        self.timestamp    = datetime.now().isoformat(timespec='seconds')
        self.run_start_t  = None
        self.gates_detected   = 0
        self.gate_positions   = []
        self.gates_lost_count = 0
        self.lap2_start_t     = None
        self.lap2_end_t       = None
        self.lap2_duration_s  = None
        self.lap2_started     = False
        self.lap2_finished    = False
        self.timed_out        = False
        self.notes            = []

    def apply(self, tag, data):
        if tag == 'RUN_START':
            self.run_start_t = data['t']
        elif tag == 'GATE_DETECTED':
            self.gates_detected = data['n']
            self.gate_positions.append(data['pos'])
        elif tag == 'GATE_LOST':
            self.gates_lost_count += 1
            self.notes.append(f"gate_lost_at={data['at_gate']}")
        elif tag == 'LAP2_START':
            self.lap2_start_t = data['t']
            self.lap2_started = True
        elif tag == 'LAP2_END':
            self.lap2_end_t     = data['t']
            self.lap2_duration_s = data['duration']
            self.lap2_finished  = True

    def to_row(self):
        return {
            'run_id':           self.run_id,
            'timestamp':        self.timestamp,
            'world':            os.path.basename(self.world),
            'run_start_t':      self.run_start_t,
            'gates_detected':   self.gates_detected,
            'gate_positions':   ' | '.join(self.gate_positions),
            'gates_lost_count': self.gates_lost_count,
            'lap2_start_t':     self.lap2_start_t,
            'lap2_end_t':       self.lap2_end_t,
            'lap2_duration_s':  self.lap2_duration_s,
            'lap2_started':     self.lap2_started,
            'lap2_finished':    self.lap2_finished,
            'timed_out':        self.timed_out,
            'notes':            ' | '.join(self.notes),
        }

    def summary(self):
        status = '✓ OK' if self.lap2_finished else ('✗ TIMEOUT' if self.timed_out else '✗ FAIL')
        return (f"Run #{self.run_id:03d} [{status}] "
                f"gates={self.gates_detected}/5 "
                f"lap2={'YES' if self.lap2_started else 'NO'} "
                f"dur={f'{self.lap2_duration_s:.1f}s' if self.lap2_duration_s else 'N/A'} "
                f"lost={self.gates_lost_count}")


# =============================================================================
# LANCEMENT WEBOTS
# =============================================================================
def find_webots():
    """Cherche l'exécutable Webots sur les chemins classiques."""
    candidates = [
        'webots',
        '/usr/local/bin/webots',
        '/Applications/Webots.app/Contents/MacOS/webots',
        'C:/Program Files/Webots/msys64/mingw64/bin/webots.exe',
    ]
    for c in candidates:
        try:
            result = subprocess.run([c, '--version'],
                                    capture_output=True, timeout=5)
            if result.returncode == 0 or b'Webots' in result.stdout + result.stderr:
                return c
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def launch_webots(webots_bin, world_path):
    """Lance Webots en mode batch (pas d'interface graphique si possible)."""
    cmd = [
        webots_bin,
        '--batch',          # pas d'interface de confirmation
        '--mode=realtime',      # vitesse max
        '--stdout',         # rediriger stdout du contrôleur
        '--stderr',         # rediriger stderr du contrôleur
        world_path,
    ]
    print(f"[MONITOR] Lancement : {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,           # line-buffered
    )
    return proc


# =============================================================================
# BOUCLE PRINCIPALE
# =============================================================================
def run_monitoring(world_path, n_runs, csv_path, webots_bin):

    # Créer/ouvrir le CSV
    csv_exists = os.path.exists(csv_path)
    csvfile = open(csv_path, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
    if not csv_exists:
        writer.writeheader()
        csvfile.flush()

    print(f"\n{'='*60}")
    print(f"  DRONE RACING MONITOR")
    print(f"  Monde : {world_path}")
    print(f"  Runs  : {n_runs}")
    print(f"  CSV   : {csv_path}")
    print(f"{'='*60}\n")

    run_id = 1

    while run_id <= n_runs:
        print(f"\n--- Run #{run_id:03d} / {n_runs} ---")
        state = RunState(run_id, world_path)

        proc = launch_webots(webots_bin, world_path)
        start_real = time.time()
        lap2_end_seen = False

        try:
            for line in proc.stdout:
                line = line.rstrip()

                # Afficher toutes les lignes utiles
                if any(tag in line for tag in ['[MONITOR]', '[STATE]', '[LAP2]',
                                               '[SCAN]', '[TAKEOFF]', '[TRAJ]',
                                               '[CCW]', '[FILTER]']):
                    elapsed = time.time() - start_real
                    print(f"  [{elapsed:6.1f}s] {line}")

                # Parser les tags MONITOR
                tag, data = parse_monitor_line(line)
                if tag:
                    state.apply(tag, data)
                    if tag == 'LAP2_END':
                        lap2_end_seen = True
                        print(f"\n  ✓ LAP2 terminé ! Attente {LAP2_WAIT_AFTER_S}s puis relance...")
                        time.sleep(LAP2_WAIT_AFTER_S)
                        proc.terminate()
                        break

                # Vérifier le timeout réel
                if time.time() - start_real > WEBOTS_TIMEOUT_S:
                    state.timed_out = True
                    state.notes.append(f"timeout_at_real={time.time()-start_real:.0f}s")
                    print(f"\n  ✗ TIMEOUT ({WEBOTS_TIMEOUT_S}s) — run interrompu")
                    proc.terminate()
                    break

        except KeyboardInterrupt:
            print("\n[MONITOR] Interruption clavier — arrêt propre")
            proc.terminate()
            row = state.to_row()
            row['notes'] = (row['notes'] + ' | interrupted').strip(' | ')
            writer.writerow(row)
            csvfile.flush()
            break

        # Attendre que le process soit bien terminé
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

        # Logger le run
        writer.writerow(state.to_row())
        csvfile.flush()
        print(f"\n  {state.summary()}")

        run_id += 1

    csvfile.close()
    print(f"\n{'='*60}")
    print(f"  Monitoring terminé. Résultats dans : {csv_path}")
    print(f"{'='*60}\n")

    # Afficher un résumé rapide
    print_summary(csv_path)


# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================
def print_summary(csv_path):
    if not os.path.exists(csv_path):
        return
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    total        = len(rows)
    lap2_started = sum(1 for r in rows if r['lap2_started'] == 'True')
    lap2_done    = sum(1 for r in rows if r['lap2_finished'] == 'True')
    timeouts     = sum(1 for r in rows if r['timed_out'] == 'True')
    gates_5      = sum(1 for r in rows if r['gates_detected'] == '5')
    durations    = [float(r['lap2_duration_s']) for r in rows
                    if r['lap2_duration_s'] not in ('', 'None', None)]

    print(f"\n{'='*40}  RÉSUMÉ  {'='*40}")
    print(f"  Total runs      : {total}")
    print(f"  5 gates lap1    : {gates_5} / {total}  ({100*gates_5//total}%)")
    print(f"  Lap2 démarré    : {lap2_started} / {total}  ({100*lap2_started//total}%)")
    print(f"  Lap2 terminé    : {lap2_done} / {total}  ({100*lap2_done//total}%)")
    print(f"  Timeouts        : {timeouts} / {total}")
    if durations:
        print(f"  Durée lap2 (s)  : min={min(durations):.1f}  "
              f"moy={sum(durations)/len(durations):.1f}  "
              f"max={max(durations):.1f}")
    print(f"{'='*90}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Monitoring automatique Webots drone racing')
    parser.add_argument('--world',  required=True,
                        help='Chemin vers le fichier .wbt (ex: worlds/world_0.wbt)')
    parser.add_argument('--runs',   type=int, default=10,
                        help='Nombre de runs à effectuer (défaut: 10)')
    parser.add_argument('--csv',    default='results.csv',
                        help='Fichier CSV de sortie (défaut: results.csv)')
    parser.add_argument('--webots', default=None,
                        help='Chemin vers l\'exécutable webots (auto-détecté si absent)')
    parser.add_argument('--timeout', type=int, default=500,
                        help='Timeout par run en secondes réelles (défaut: 250)')
    args = parser.parse_args()

    WEBOTS_TIMEOUT_S = args.timeout

    # Trouver Webots
    webots_bin = args.webots or find_webots()
    if webots_bin is None:
        print("ERREUR : Webots introuvable. Précisez le chemin avec --webots /path/to/webots")
        sys.exit(1)
    print(f"[MONITOR] Webots trouvé : {webots_bin}")

    # Vérifier le monde
    if not os.path.exists(args.world):
        print(f"ERREUR : Monde introuvable : {args.world}")
        sys.exit(1)

    run_monitoring(
        world_path  = args.world,
        n_runs      = args.runs,
        csv_path    = args.csv,
        webots_bin  = webots_bin,
    )