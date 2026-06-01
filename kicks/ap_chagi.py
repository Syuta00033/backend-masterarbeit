
from .geometry import L_KNEE, R_KNEE, calc_angle, hip_rotation, leg_angles, L_ANKLE, R_ANKLE
import numpy as np


class ApChagi:

    NAME = "ap_chagi"

    # Maximaler Winkel für die Chamber-Position, um sie als solche zu erkennen
    CHAMBER_KNEE_MAX = 100
    CHAMBER_HIP_MAX = 120

    CHAMBER_QUALITY_KNEE_MAX = 90
    CHAMBER_QUALITY_HIP_MAX = 100

    # Mindest Winkel für die Kick-Position, um sie als solche zu erkennen
    KICK_KNEE_MIN = 150

    # Mindest Winkel, um zurück in den Idle-Zustand zu wechseln (Kick vorbei oder abgebrochen)
    IDLE_KNEE_MIN = 150
    IDLE_HIP_MIN = 150

    # Toleranz für das Absacken des Knies während des Kicks (in Metern) — das Knie sollte auf Höhe bleiben oder nur minimal absacken
    KNEE_DROP_TOLERANCE_M = 0.15

    # Maximaler Winkel des Standbeins - Sollte unter dem Wert bleiben
    STANDING_KNEE_MAX = 175

    # Winkel-Toleranz für Rechamber - Sollte 15 Grad oder weniger von der Chamber-Position abweichen, um als Rechamber zu gelten
    RECHAMBER_TOLERANCE_DEG = 15

    LIFT_THRESHHOLD_M = 0.05 # Fuß sollte mindestens 5 cm vom Boden abheben, um den Start des Kicks zu erkennen
    BASELINE_WINDOW_FRAMES = 10 

    def __init__(self):
        self.phase = "idle"
        self.phases_log = []
        self.kick_start_frame = None # Frame, bei dem Kick beginnt
        self.last_kick_duration_frames = None # Insgesamte Frame Dauer des Kicks in Frames
        self.knee_angle = 0.0 # Aktueller Kniewinkel
        self.hip_flexion = 0.0 # Aktuelle Hüftbeugung
        self.max_knee_in_kick = 0.0 # Maximaler Kniewinkel während des Kick-Phases
        self.min_knee_in_chamber = 180.0 # Minimaler Kniewinkel während der Chamber-Phase
        self.min_knee_in_rechamber = 180.0 # Minimaler Kniewinkel während der Rechamber-Phase
        self.min_hip_in_chamber = 180.0 # Minimaler Hüftbeugungswinkel während der Chamber-Phase
        self.min_hip_in_rechamber = 180.0 # Minimaler Hüftbeugungswinkel während der Rechamber-Phase
        self.min_knee_y = float("inf") # Minimaler y-Wert des Knies während des Chamber + Kick
        self.max_knee_y_in_kick = float("-inf") # Maximaler y-Wert des Knies während des Kicks (je größer, desto mehr sackt das Knie ab)
        self.standing_knee_angle = 0.0 # Aktueller Kniewinkel des Standbeins
        self.max_standing_knee_angle = 0.0 # Maximaler Kniewinkel des Standbeins während des Kicks (je größer, desto mehr durchgestreckt)
        self.torso_angle = 0.0 # Aktueller Winkel des Torsos (Schulter-Hüfte-Knie)
        self.hip_rotation = 0.0 # Aktuelle Hüftrotation
        self.max_abs_hip_rotation = 0.0 # Maximaler absoluter Wert der Hüftrotation während des Kicks (je größer, desto mehr Rotation)

        # Standfuß
        self.idle_ankle_history = [] # Historie der Knöchel-Positionen in Idle-Frames, um die Höhe des Standfußes über dem Boden zu bestimmen
        self.baseline_ankle_y = None # gemittelte Höhe

    def update(self, wl, kicking_side, frame_index):
        # Berechne Knie und Hüftwinkel für das kickende Bein
        self.knee_angle, self.hip_flexion = leg_angles(wl, kicking_side)

        # Berechne Kniewinkel des Standbeins
        standing_side = "right" if kicking_side == "left" else "left"
        self.standing_knee_angle, _ = leg_angles(wl, standing_side)

        # Berechne Hüftrotation
        self.hip_rotation = hip_rotation(wl)
        self.max_abs_hip_rotation = max(self.max_abs_hip_rotation, abs(self.hip_rotation))

        # Y-Wert des kickenden Knies für die Bewertung, ob das Knie während des Kicks zu stark absackt. Das Knie sollte auf Höhe bleiben oder nur minimal absacken.
        kicking_knee_idx = L_KNEE if kicking_side == "left" else R_KNEE
        knee_y = wl[kicking_knee_idx].y

        standing_ankle_idx = L_ANKLE if standing_side == "left" else R_ANKLE
        ankle_y = wl[standing_ankle_idx].y

        #self.torso_angle = calc_angle(wl[11], wl[23], wl[23] + np.array([0, 1, 0]))  # Schulter-Hüfte-Knie

        if self.phase == "idle":
            self.idle_ankle_history.append(ankle_y)

            # Detektiere Beginn des Kicks
            if len(self.idle_ankle_history) > self.BASELINE_WINDOW_FRAMES:
                self.idle_ankle_history.pop(0)
            if len(self.idle_ankle_history) >= self.BASELINE_WINDOW_FRAMES:
                self.baseline_ankle_y = np.mean(self.idle_ankle_history)
            
            if self.baseline_ankle_y is not None and self.kick_start_frame is None and (self.baseline_ankle_y - ankle_y) > self.LIFT_THRESHHOLD_M:
                self.kick_start_frame = frame_index

            # Wechsel in Chamber, wenn beide Bedingungen erfüllt sind: Knie genug angewinkelt und Hüfte genug gebeugt
            if self.knee_angle < self.CHAMBER_KNEE_MAX and self.hip_flexion < self.CHAMBER_HIP_MAX:
                self.phase = "chamber"
                self.phases_log.append(("chamber", frame_index))
                #self.kick_start_frame = frame_index

        elif self.phase == "chamber":
            # Aktualisiere die minimalen Winkel in der Chamber-Position für die spätere Bewertung
            if self.knee_angle < self.min_knee_in_chamber:
                self.min_knee_in_chamber = self.knee_angle
            
            # Aktualisiere die minimale Hüftbeugung in der Chamber-Position für die spätere Bewertung
            if self.hip_flexion < self.min_hip_in_chamber:
                self.min_hip_in_chamber = self.hip_flexion

            # Aktualisiere den minimalen y-Wert des Knies während der Chamber-Position, um später bewerten zu können, wie hoch das Knie gezogen wurde
            if knee_y < self.min_knee_y:
                self.min_knee_y = knee_y

            # Wechsel in Kick, wenn beide Bedingungen erfüllt sind: Knie genug gestreckt und Hüfte noch gebeugt genug
            if self.knee_angle > self.KICK_KNEE_MIN and self.hip_flexion < self.CHAMBER_HIP_MAX:
                self.phase = "kick"
                self.phases_log.append(("kick", frame_index))
            elif self.knee_angle > self.IDLE_KNEE_MIN and self.hip_flexion > self.IDLE_HIP_MIN: # Abbruch zurück in Idle, wenn die Person doch nicht kickt sondern z.B. nur das Knie hebt
                self.phase = "idle"
                self.phases_log.append(("idle", frame_index))
                self.reset()

        elif self.phase == "kick":
            # Aktualisiere den maximalen Kniewinkel während des Kicks für die spätere Bewertung der Streckung
            if self.knee_angle > self.max_knee_in_kick:
                self.max_knee_in_kick = self.knee_angle

            # Aktualisiere den maximalen y-Wert des Knies während des Kicks, um später bewerten zu können, wie stark das Knie absackt. Das Knie sollte auf Höhe bleiben oder nur minimal absacken.
            if knee_y < self.min_knee_y:
                self.min_knee_y = knee_y
            if knee_y > self.max_knee_y_in_kick:
                self.max_knee_y_in_kick = knee_y

            # Aktualisiere den maximalen Kniewinkel des Standbeins während des Kicks für die spätere Bewertung, ob das Standbein zu stark durchgestreckt ist
            if self.standing_knee_angle > self.max_standing_knee_angle:
                self.max_standing_knee_angle = self.standing_knee_angle

            # Aktualisiere die maximale absolute Hüftrotation während des Kicks für die spätere Bewertung, ob ausreichend Hüftrotation vorhanden ist
            if abs(self.hip_rotation) > self.max_abs_hip_rotation:
                self.max_abs_hip_rotation = abs(self.hip_rotation)

            # Wechsel zurück in Idle, wenn die Person das Bein wieder runternimmt oder wenn ein Rechamber erkannt wird (Knie wird nach dem Kick wieder hochgezogen in die Nähe der Chamber-Position)
            if abs(self.knee_angle - self.min_knee_in_chamber) <= self.RECHAMBER_TOLERANCE_DEG and abs(self.hip_flexion - self.min_hip_in_chamber) <= self.RECHAMBER_TOLERANCE_DEG:
                self.phase = "rechamber"
                self.phases_log.append(("rechamber", frame_index))
            elif self.knee_angle > self.IDLE_KNEE_MIN and self.hip_flexion > self.IDLE_HIP_MIN:
                self.phase = "idle"
                self.phases_log.append(("idle", frame_index))
                if self.kick_start_frame is not None:
                    self.last_kick_duration_frames = frame_index - self.kick_start_frame
                    self.reset()

        elif self.phase == "rechamber":
            # Aktualisiere die minimalen Winkel während der Rechamber-Phase
            if self.knee_angle < self.min_knee_in_rechamber:
                self.min_knee_in_rechamber = self.knee_angle
            if self.hip_flexion < self.min_hip_in_rechamber:
                self.min_hip_in_rechamber = self.hip_flexion

            # Wechsel zurück in Idle, wenn die Person das Bein wieder runternimmt
            if self.knee_angle > self.IDLE_KNEE_MIN and self.hip_flexion > self.IDLE_HIP_MIN:
                self.phase = "idle"
                self.phases_log.append(("idle", frame_index))
                if self.kick_start_frame is not None:
                    self.last_kick_duration_frames = frame_index - self.kick_start_frame
                    self.reset()

    def reset(self):
        self.kick_start_frame = None
        self.idle_ankle_history = []
        self.baseline_ankle_y = None

    def evaluate(self):
        phase_names = [p for (p, f) in self.phases_log]
        results = []

        results.append({
                    "name": "chamber_depth",
                    "passed": self.min_knee_in_chamber <= self.CHAMBER_QUALITY_KNEE_MAX,
                    "value": self.min_knee_in_chamber,
                    "feedback": "Chamber eng genug." if self.min_knee_in_chamber <= self.CHAMBER_QUALITY_KNEE_MAX
                                else "Knie nicht eng genug gezogen.",
                })
        
        results.append({
            "name": "hip_lift",
            "passed": self.min_hip_in_chamber <= self.CHAMBER_QUALITY_HIP_MAX,
            "value": self.min_hip_in_chamber,
            "feedback": "Hüfte ausreichend gehoben." if self.min_hip_in_chamber <= self.CHAMBER_QUALITY_HIP_MAX
                        else "Hüfte nicht weit genug gebeugt — Knie zu niedrig.",
        })

        knee_extended = self.max_knee_in_kick >= self.KICK_KNEE_MIN

        results.append({
            "name": "Knee Extension",
            "passed": knee_extended,
            "value": self.max_knee_in_kick,
            "feedback": "Bein voll gestreckt." if knee_extended
            else "Bein nicht vollständig gestreckt." 
        })

        if not knee_extended:
            return results

        if self.max_knee_y_in_kick > float("-inf") and self.min_knee_y < float("inf"):
            knee_drop = self.max_knee_y_in_kick - self.min_knee_y
        else:
            knee_drop = 0.0

        results.append({
            "name": "knee_height_maintained",
            "passed": knee_drop < self.KNEE_DROP_TOLERANCE_M,
            "value": round(knee_drop, 3),
            "feedback": "Knie bleibt auf Höhe während des Kicks." if knee_drop < self.KNEE_DROP_TOLERANCE_M
                        else f"Knie sackt im Kick um {round(knee_drop * 100)} cm ab — die Hüfte sollte die Beugung halten, während das Bein streckt.",
        })
        
        results.append({
            "name": "rechamber",
            "passed": "rechamber" in phase_names,
            "value": None,
            "feedback": "Rechamber durchgeführt." if "rechamber" in phase_names
                        else "Rechamber vergessen — nach dem Treffen sollte das Knie zurück in die Chamber-Position gezogen werden, bevor der Fuß abgesetzt wird.",
        })

        if "rechamber" in phase_names:
            knee_diff = abs(self.min_knee_in_rechamber - self.min_knee_in_chamber)
        

        results.append({
            "name": "supporting_leg_not_overextended",
            "passed": self.max_standing_knee_angle < self.STANDING_KNEE_MAX,
            "value": round(self.max_standing_knee_angle, 1),
            "feedback": "Standbein leicht gebeugt — Balance gut." if self.max_standing_knee_angle < self.STANDING_KNEE_MAX
                        else "Standbein durchgestreckt — eine leichte Beugung verbessert Balance und Kraftübertragung.",
        })

        results.append({
            "name": "hip_rotation",
            "passed": self.max_abs_hip_rotation > 10,
            "value": round(self.max_abs_hip_rotation, 1),
            "feedback": "Hüftrotation vorhanden" if self.max_abs_hip_rotation > 10
                        else "Hüfte rotiert kaum",
        })

        return results
