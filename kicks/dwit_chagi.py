from collections import deque
import statistics

from .geometry import (
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
    landmark_to_array, leg_angles, angle_difference, score_linear,
    pelvis_facing, hip_kick_angle,
)
import numpy as np


class DwitChagi:
    NAME = "dwit_chagi"

    # --- Detection ---
    CHAMBER_KNEE_MAX = 90
    CHAMBER_HIP_MAX = 150
    KICK_KNEE_MIN = 100
    IDLE_KNEE_MIN = 160
    IDLE_HIP_MIN = 160
    KNEE_RETURN_MIN = 35
    RECHAMBER_FOOT_HIGH = 0.25
    LIFT_THRESHHOLD_M = 0.05
    BASELINE_WINDOW_FRAMES = 10

    ROTATION_MIN = 45      # ab so viel aufsummierter Drehung -> Rotationsphase
    MAX_STEP_DEG = 30      # größere Winkeländerung pro Frame = Landmark-Flip
    JUMP_FILTER_DEG = 60   # größerer Sprung in hip_alignment = Flip -> verwerfen

    # --- Quality ---
    CHAMBER_QUALITY_KNEE_MAX = 55
    KNEE_DROP_TOLERANCE_M = 0.15
    STANDING_KNEE_MAX = 165
    BODY_ROTATION_FAIL = 110   # Ausrichtung beim Treffer (~180 = Rücken zum Ziel)
    BODY_ROTATION_IDEAL = 170
    OVER_ROTATION_FAIL = 60    # ab hier gilt die Drehung als durchgedreht
    FOOT_GAP_FAIL = 0.40
    FOOT_GAP_IDEAL = 0.15
    KICK_REACH_MIN = 0.40      # min. Fußabstand für echten Kick (Meter)
    RECHAMBER_FOOT_GAP = 0.25  # Fuß gilt als zurück am Standbein

    # Kick-Erkennung über die Fußbahn, Werte in Beinlängen
    FOOT_HIGH_MIN = 0.30       # Fuß gilt als "oben"
    FOOT_LOW_MAX = 0.15        # Fuß gilt als abgesetzt
    KICK_GAP_MIN = 0.45        # min. horizontaler Fußabstand

    def __init__(self):
        # --- phases state ---
        self.phase = "idle"
        self.phases_log = []
        self.kick_detected = False
        self.kick_start_frame = None
        self.last_kick_duration_frames = None

        # --- active values for current frame ---
        self.knee_angle = 0.0
        self.hip_flexion = 0.0
        self.standing_knee_angle = 0.0
        self.torso_angle = 0.0
        self._knee_y = 0.0
        self._ankle_y = 0.0
        self._foot_gap = 0.0
        self._foot_height = 0.0   # Kickfuß über Standfuß, in Beinlängen
        self._foot_gap_h = 0.0    # horizontaler Knöchelabstand, in Beinlängen

        # --- Rotation (aufsummiert) ---
        self.prev_pelvis_angle = None
        self.total_rotation = 0.0

        # --- hip alignment (0..180) ---
        self.hip_alignment = 0.0
        self.max_hip_alignment = 0.0          # nur Debug
        self.hip_alignment_at_impact = 0.0
        self.prev_alignment = None

        # --- Über-Rotation ---
        self._best_align = 0.0
        self._rotation_at_best_align = None
        self.max_over_rotation = 0.0

        # --- Min/Max for Chamber ---
        self.min_knee_in_chamber = 180.0
        self.min_hip_in_chamber = 180.0

        # --- Min/Max for Kick ---
        self.max_knee_in_kick = 0.0
        self.min_knee_after_peak = 180.0
        self.min_knee_y = float("inf")
        self.max_knee_y_in_kick = -float("inf")
        self.standing_knee_history = []
        self.min_foot_gap = float("inf")
        self.max_foot_gap_in_kick = 0.0   # wie weit der Fuß rausging (für Rechamber-Erkennung)
        self.max_foot_height_in_chamber = 0.0

        # --- Min/Max for Rechamber ---
        self.min_knee_in_rechamber = 180.0
        self.min_hip_in_rechamber = 180.0

        # --- standing ankle for idle detection ---
        self.idle_ankle_history = []
        self.baseline_ankle_y = None

        # --- velocity ---
        self.max_foot_velocity = 0.0
        self.fps = 30
        self.prev_foot_pos = None
        self.velocity_history = deque(maxlen=3)

    def update(self, wl, kicking_side, frame_index):
        self._compute_current_values(wl, kicking_side)

        if self.phase == "idle":
            self._update_idle(frame_index)
        elif self.phase == "rotation":
            self._update_rotation(frame_index)
        elif self.phase == "chamber":
            self._update_chamber(frame_index)
        elif self.phase == "kick":
            self._update_kick(frame_index)
        elif self.phase == "rechamber":
            self._update_rechamber(frame_index)

    def _compute_current_values(self, wl, kicking_side):
        # Knie- und Hüftwinkel des Kickbeins
        self.knee_angle, self.hip_flexion = leg_angles(wl, kicking_side)

        # Standbein
        standing_side = "right" if kicking_side == "left" else "left"
        self.standing_knee_angle, _ = leg_angles(wl, standing_side)

        # aufsummierte Beckendrehung, Sprung-Filter gegen Landmark-Flips
        current = pelvis_facing(wl)
        if self.prev_pelvis_angle is not None:
            step = angle_difference(current, self.prev_pelvis_angle)
            if abs(step) <= self.MAX_STEP_DEG:
                self.total_rotation += step
        self.prev_pelvis_angle = current

        # hip alignment im vollen 0..180-Bereich, mit Sprung-Filter
        raw = hip_kick_angle(wl, kicking_side)
        if self.prev_alignment is not None and abs(raw - self.prev_alignment) > self.JUMP_FILTER_DEG:
            raw = self.prev_alignment
        self.hip_alignment = raw
        self.prev_alignment = raw

        # Über-Rotation: Weiterdrehung nach dem besten Alignment-Moment
        if self.phase in ("rotation", "chamber", "kick"):
            if self.hip_alignment > self._best_align:
                self._best_align = self.hip_alignment
                self._rotation_at_best_align = self.total_rotation
            if self._rotation_at_best_align is not None:
                self.max_over_rotation = max(
                    self.max_over_rotation,
                    abs(self.total_rotation - self._rotation_at_best_align),
                )

        # y-Position des Kickknies
        kicking_knee_idx = L_KNEE if kicking_side == "left" else R_KNEE
        self._knee_y = wl[kicking_knee_idx].y

        standing_ankle_idx = L_ANKLE if standing_side == "left" else R_ANKLE
        self._ankle_y = wl[standing_ankle_idx].y

        # Abstand Kickknöchel <-> Standknöchel
        kick_ankle_idx = L_ANKLE if kicking_side == "left" else R_ANKLE
        ka = np.array([wl[kick_ankle_idx].x, wl[kick_ankle_idx].y, wl[kick_ankle_idx].z])
        sa = np.array([wl[standing_ankle_idx].x, wl[standing_ankle_idx].y, wl[standing_ankle_idx].z])
        self._foot_gap = float(np.linalg.norm(ka - sa))

        # Fußhöhe und horizontaler Knöchelabstand, normiert auf die Beinlänge
        sh = landmark_to_array(wl[L_HIP if standing_side == "left" else R_HIP])
        sk = landmark_to_array(wl[L_KNEE if standing_side == "left" else R_KNEE])
        leg_len = float(np.linalg.norm(sh - sk) + np.linalg.norm(sk - sa)) or 1.0

        self._foot_height = float(wl[standing_ankle_idx].y - wl[kick_ankle_idx].y) / leg_len
        dx = wl[kick_ankle_idx].x - wl[standing_ankle_idx].x
        dz = wl[kick_ankle_idx].z - wl[standing_ankle_idx].z
        self._foot_gap_h = float(np.hypot(dx, dz)) / leg_len

        # Fußgeschwindigkeit
        foot_idx = L_ANKLE if kicking_side == "left" else R_ANKLE
        pos = np.array([wl[foot_idx].x, wl[foot_idx].y, wl[foot_idx].z])
        if self.prev_foot_pos is not None:
            dist = np.linalg.norm(pos - self.prev_foot_pos)
            velocity = (dist * self.fps) * 3.6  # km/h
            self.velocity_history.append(velocity)
            smoothed = round(statistics.median(self.velocity_history))
            if self.phase in ("rotation", "chamber", "kick"):
                self.max_foot_velocity = max(self.max_foot_velocity, smoothed)
        self.prev_foot_pos = pos

    def _update_idle(self, frame_index):
        self.idle_ankle_history.append(self._ankle_y)
        if len(self.idle_ankle_history) > self.BASELINE_WINDOW_FRAMES:
            self.idle_ankle_history.pop(0)
        if len(self.idle_ankle_history) >= self.BASELINE_WINDOW_FRAMES:
            self.baseline_ankle_y = np.mean(self.idle_ankle_history)

        if self.baseline_ankle_y is not None and self.kick_start_frame is None and (self.baseline_ankle_y - self._ankle_y) > self.LIFT_THRESHHOLD_M:
            self.kick_start_frame = frame_index

        # Drehung erkannt -> Rotationsphase, sonst direkt Chamber
        if abs(self.total_rotation) > self.ROTATION_MIN:
            self._transition_to("rotation", frame_index)
        elif self.knee_angle < self.CHAMBER_KNEE_MAX and self.hip_flexion < self.CHAMBER_HIP_MAX:
            self._transition_to("chamber", frame_index)

    def _update_rotation(self, frame_index):

        # sobald das Knie angezogen wird -> Chamber
        if self.knee_angle < self.CHAMBER_KNEE_MAX:
            self._transition_to("chamber", frame_index)

    def _update_chamber(self, frame_index):
        self.min_knee_in_chamber = min(self.min_knee_in_chamber, self.knee_angle)
        self.min_hip_in_chamber = min(self.min_hip_in_chamber, self.hip_flexion)
        self.min_knee_y = min(self.min_knee_y, self._knee_y)
        self.min_foot_gap = min(self.min_foot_gap, self._foot_gap)
        self.max_foot_height_in_chamber = max(self.max_foot_height_in_chamber, self._foot_height)

        # Abbruch zuerst: Beim Absetzen streckt sich das Knie auch, das ist kein Kick
        foot_was_up = self.max_foot_height_in_chamber > self.FOOT_HIGH_MIN
        if (foot_was_up and self._foot_height < self.FOOT_LOW_MAX):
            self._transition_to("idle", frame_index)
            self.reset()
            return

        # Kick: Fuß geht horizontal raus und bleibt oben
        if self._foot_gap_h > self.KICK_GAP_MIN and self._foot_height > self.FOOT_HIGH_MIN:
            self._transition_to("kick", frame_index)

    def _leg_is_up(self):
        return self._foot_height > self.RECHAMBER_FOOT_HIGH

    def _update_kick(self, frame_index):
        # Ausrichtung im Moment der max. Streckung festhalten (Treffer-Moment)
        if self.knee_angle > self.max_knee_in_kick:
            self.max_knee_in_kick = self.knee_angle
            self.min_knee_after_peak = self.knee_angle
            self.hip_alignment_at_impact = self.hip_alignment

        self.min_knee_y = min(self.min_knee_y, self._knee_y)
        self.max_knee_y_in_kick = max(self.max_knee_y_in_kick, self._knee_y)
        self.standing_knee_history.append(self.standing_knee_angle)
        self.min_foot_gap = min(self.min_foot_gap, self._foot_gap)
        self.max_foot_gap_in_kick = max(self.max_foot_gap_in_kick, self._foot_gap)
        self.max_hip_alignment = max(self.max_hip_alignment, self.hip_alignment)  # nur Debug

        leg_up = self._leg_is_up()
        if leg_up:
            self.min_knee_after_peak = min(self.min_knee_after_peak, self.knee_angle)

        if leg_up and (self.max_knee_in_kick - self.knee_angle) >= self.KNEE_RETURN_MIN:
            self._transition_to("rechamber", frame_index)
        elif self.knee_angle > self.IDLE_KNEE_MIN and self.hip_flexion > self.IDLE_HIP_MIN:
            self._transition_to("idle", frame_index)
            if self.kick_start_frame is not None:
                self.last_kick_duration_frames = frame_index - self.kick_start_frame
            self.reset()

    def _update_rechamber(self, frame_index):
        self.min_knee_in_rechamber = min(self.min_knee_in_rechamber, self.knee_angle)
        self.min_hip_in_rechamber = min(self.min_hip_in_rechamber, self.hip_flexion)

        if self._leg_is_up():
            self.min_knee_after_peak = min(self.min_knee_after_peak, self.knee_angle)

        # zurück in idle, sobald das Bein wieder abgesetzt wird
        if self.knee_angle > self.IDLE_KNEE_MIN:
            self._transition_to("idle", frame_index)
            if self.kick_start_frame is not None:
                self.last_kick_duration_frames = frame_index - self.kick_start_frame
            self.reset()

    def _transition_to(self, new_phase, frame_index):
        if new_phase == "kick":
            self.kick_detected = True
        self.phase = new_phase
        self.phases_log.append((new_phase, frame_index))

    def reset(self):
        self.kick_start_frame = None
        self.idle_ankle_history = []
        self.baseline_ankle_y = None
        self.total_rotation = 0.0
        self.prev_pelvis_angle = None
        self._best_align = 0.0
        self._rotation_at_best_align = None
        self.max_foot_gap_in_kick = 0.0
        self.max_foot_height_in_chamber = 0.0

    @staticmethod
    def _graded(name, label, value, fail_at, ideal_at, ok, fail, value_display=None):
        score = score_linear(value, fail_at, ideal_at)
        passed = score >= 50.0
        return {
            "name": name,
            "label": label,
            "score": round(score),
            "value": value_display if value_display is not None else round(value, 1),
            "passed": passed,
            "feedback": ok if passed else fail,
        }

    @staticmethod
    def _boolean_criterium(name, label, done, ok, fail):
        return {
            "name": name,
            "label": label,
            "score": 100 if done else 0,
            "value": None,
            "passed": done,
            "feedback": ok if done else fail,
        }

    def evaluate(self):
        results = []

        results.append(self._graded(
            "chamber_depth", "Chamber Winkel", self.min_knee_in_chamber,
            fail_at=self.CHAMBER_KNEE_MAX, ideal_at=self.CHAMBER_QUALITY_KNEE_MAX,
            ok="Knie angezogen.",
            fail="Knie nicht genug angezogen.",
        ))

        results.append(self._graded(
            "knee_extension", "Beinstreckung", self.max_knee_in_kick,
            fail_at=90, ideal_at=140,
            ok="Bein gut gestreckt.",
            fail="Bein nicht weit genug gestreckt — beim Dwit Chagi schiebt das Bein gerade nach hinten durch.",
        ))

        if not self.kick_detected:
            return results

        if self.max_over_rotation > self.OVER_ROTATION_FAIL:
            rotation_fail = "Zu weit durchgedreht. Das ist eher ein Spinning Side Kick"
        else:
            rotation_fail = "Zu wenig eingedreht — beim Treffer muss der Rücken zum Ziel zeigen."

        results.append(self._graded(
            "body_rotation", "Körperdrehung", self.hip_alignment_at_impact,
            fail_at=self.BODY_ROTATION_FAIL, ideal_at=self.BODY_ROTATION_IDEAL,
            ok="Körper korrekt eingedreht — Rücken zum Ziel.",
            fail=rotation_fail,
        ))

        # TODO Fußbahn-Kriterium: seitliche Abweichung von der Kicklinie,
        # auf Beinlänge normalisiert (min_foot_gap misst den Fehler nicht)

        standing_knee = statistics.median(self.standing_knee_history) if self.standing_knee_history else 180.0
        results.append(self._graded(
            "supporting_leg", "Standbein", standing_knee,
            fail_at=180, ideal_at=self.STANDING_KNEE_MAX,
            ok="Balance gut: Standbein leicht gebeugt",
            fail="Standbein durchgestreckt. Eine leichte Beugung verbessert Balance.",
        ))

        knee_return = max(0.0, self.max_knee_in_kick - self.min_knee_after_peak)
        results.append(self._graded(
            "rechamber", "Rechamber", knee_return,
            fail_at=10, ideal_at=60,
            ok="Rechamber durchgeführt.",
            fail="Knie nach dem Treffen weiter zurückziehen, bevor du absetzt.",
        ))

        return results
