from .geometry import L_KNEE, R_KNEE, calc_angle, leg_angles
import numpy as np


class ApChagi:

    NAME = "ap_chagi"

    CHAMBER_KNEE_MAX = 130
    CHAMBER_HIP_MAX = 130
    KICK_KNEE_MIN = 150
    IDLE_KNEE_MIN = 160
    IDLE_HIP_MIN = 160

    KNEE_DROP_TOLERANCE_M = 0.15
    STANDING_KNEE_MAX = 170

    def __init__(self):
        self.phase = "idle"
        self.phases_log = []
        self._kick_start_frame = None
        self.last_kick_duration_frames = None
        self.knee_angle = 0.0
        self.hip_flexion = 0.0
        self.max_knee_in_kick = 0.0
        self.min_knee_in_chamber = 180.0
        self.min_hip_in_chamber = 180.0
        self.min_knee_y = float("inf")
        self.max_knee_y_in_kick = float("-inf")
        self.standing_knee_angle = 0.0
        self.max_standing_knee_angle = 0.0

        self.torso_angle = 0.0

    def update(self, wl, kicking_side, frame_index):
        self.knee_angle, self.hip_flexion = leg_angles(wl, kicking_side)

        standing_side = "right" if kicking_side == "left" else "left"
        self.standing_knee_angle, _ = leg_angles(wl, standing_side)

        kicking_knee_idx = L_KNEE if kicking_side == "left" else R_KNEE
        knee_y = wl[kicking_knee_idx].y

        #self.torso_angle = calc_angle(wl[11], wl[23], wl[23] + np.array([0, 1, 0]))  # Schulter-Hüfte-Knie

        if self.phase == "idle":
            if self.knee_angle < self.CHAMBER_KNEE_MAX and self.hip_flexion < self.CHAMBER_HIP_MAX:
                self.phase = "chamber"
                self.phases_log.append(("chamber", frame_index))
                self._kick_start_frame = frame_index

        elif self.phase == "chamber":
            if self.knee_angle < self.min_knee_in_chamber:
                self.min_knee_in_chamber = self.knee_angle
            if self.hip_flexion < self.min_hip_in_chamber:
                self.min_hip_in_chamber = self.hip_flexion
            if knee_y < self.min_knee_y:
                self.min_knee_y = knee_y

            if self.knee_angle > self.KICK_KNEE_MIN and self.hip_flexion < self.CHAMBER_HIP_MAX:
                self.phase = "kick"
                self.phases_log.append(("kick", frame_index))
            elif self.knee_angle > self.IDLE_KNEE_MIN and self.hip_flexion > self.IDLE_HIP_MIN:
                self.phase = "idle"
                self.phases_log.append(("idle", frame_index))
                self._kick_start_frame = None

        elif self.phase == "kick":
            if self.knee_angle > self.max_knee_in_kick:
                self.max_knee_in_kick = self.knee_angle
            if knee_y < self.min_knee_y:
                self.min_knee_y = knee_y
            if knee_y > self.max_knee_y_in_kick:
                self.max_knee_y_in_kick = knee_y
            if self.standing_knee_angle > self.max_standing_knee_angle:
                self.max_standing_knee_angle = self.standing_knee_angle

            if self.knee_angle < self.CHAMBER_KNEE_MAX and self.hip_flexion < self.CHAMBER_HIP_MAX:
                self.phase = "rechamber"
                self.phases_log.append(("rechamber", frame_index))
            elif self.knee_angle > self.IDLE_KNEE_MIN and self.hip_flexion > self.IDLE_HIP_MIN:
                self.phase = "idle"
                self.phases_log.append(("idle", frame_index))
                if self._kick_start_frame is not None:
                    self.last_kick_duration_frames = frame_index - self._kick_start_frame
                    self._kick_start_frame = None

        elif self.phase == "rechamber":
            if self.knee_angle > self.IDLE_KNEE_MIN and self.hip_flexion > self.IDLE_HIP_MIN:
                self.phase = "idle"
                self.phases_log.append(("idle", frame_index))
                if self._kick_start_frame is not None:
                    self.last_kick_duration_frames = frame_index - self._kick_start_frame
                    self._kick_start_frame = None

    def evaluate(self):
        phase_names = [p for (p, f) in self.phases_log]
        results = []

        results.append({
            "name": "Knee Extension",
            "passed": self.max_knee_in_kick > 165,
            "value": self.max_knee_in_kick,
            "feedback": "Bein voll gestreckt." if self.max_knee_in_kick > 165 
            else "Bein nicht vollständig gestreckt."
        })

        results.append({
            "name": "chamber_depth",
            "passed": self.min_knee_in_chamber < 100,
            "value": self.min_knee_in_chamber,
            "feedback": "Chamber sauber." if self.min_knee_in_chamber < 100
                        else "Knie nicht hoch genug gezogen vor dem Kick.",
        })

        results.append({
            "name": "rechamber",
            "passed": "rechamber" in phase_names,
            "value": None,
            "feedback": "Rechamber durchgeführt." if "rechamber" in phase_names
                        else "Rechamber vergessen — nach dem Treffen sollte das Knie zurück in die Chamber-Position gezogen werden, bevor der Fuß abgesetzt wird.",
        })

        results.append({
            "name": "hip_lift",
            "passed": self.min_hip_in_chamber < 120,
            "value": self.min_hip_in_chamber,
            "feedback": "Hüfte ausreichend gehoben." if self.min_hip_in_chamber < 120
                        else "Hüfte nicht weit genug gebeugt — Knie zu niedrig.",
        })

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
            "name": "supporting_leg_not_overextended",
            "passed": self.max_standing_knee_angle < self.STANDING_KNEE_MAX,
            "value": round(self.max_standing_knee_angle, 1),
            "feedback": "Standbein leicht gebeugt — Balance gut." if self.max_standing_knee_angle < self.STANDING_KNEE_MAX
                        else "Standbein durchgestreckt — eine leichte Beugung verbessert Balance und Kraftübertragung.",
        })

        return results
