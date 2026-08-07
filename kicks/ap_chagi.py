
from collections import deque
import statistics

from .geometry import L_KNEE, R_KNEE, hip_kick_angle, leg_angles, L_ANKLE, R_ANKLE, score_linear, thigh_elevation
import numpy as np


class ApChagi:

    NAME = "ap_chagi"

    # --- Detection ---
    CHAMBER_KNEE_MAX = 100
    CHAMBER_HIP_MAX = 120
    KICK_KNEE_MIN = 150
    IDLE_KNEE_MIN = 150
    IDLE_HIP_MIN = 150
    KNEE_RETURN_MIN = 80
    LEG_UP_MIN = 45
    LIFT_THRESHHOLD_M = 0.05
    BASELINE_WINDOW_FRAMES = 10

    # --- Quality ---
    CHAMBER_QUALITY_KNEE_MAX = 60
    THIGH_ELEVATION_FAIL = 50
    THIGH_ELEVATION_IDEAL = 90
    KNEE_DROP_TOLERANCE_M = 0.15
    STANDING_KNEE_MAX = 165

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
        self.thigh_elevation = 0.0
        self.standing_knee_angle = 0.0
        self.torso_angle = 0.0
        self._knee_y = 0.0
        self._ankle_y = 0.0

        # --- Min/Max for Chamber ---
        self.min_knee_in_chamber = 180.0
        self.min_hip_in_chamber = 180.0
        self.max_thigh_elevation = 0.0

        # --- Min/Max for Kick ---
        self.max_knee_in_kick = 0.0
        self.min_knee_after_peak = 180.0
        self.min_knee_y = float("inf")
        self.max_knee_y_in_kick = float("-inf")
        self.standing_knee_history = []

        # --- Min/Max for Rechamber ---
        self.min_knee_in_rechamber = 180.0
        self.min_hip_in_rechamber = 180.0

        # --- standing ankle for idle detection ---
        self.idle_ankle_history = []
        self.baseline_ankle_y = None

        # hip rotation/alignment
        self.hip_alignment = 0.0
        self.max_hip_alignment = 0.0

        # fps
        self.fps = 30
        self.prev_foot_pos = None
        self.max_foot_velocity = 0
        self.velocity_history = deque(maxlen=3)

    def update(self, wl, kicking_side, frame_index):
        self._compute_current_values(wl, kicking_side)

        if self.phase == "idle":
            self._update_idle(frame_index)

        elif self.phase == "chamber":
            self._update_chamber(frame_index)

        elif self.phase == "kick":
            self._update_kick(frame_index)

        elif self.phase == "rechamber":
            self._update_rechamber(frame_index)
    

    def _compute_current_values(self, wl, kicking_side):
        # compute knee and hip angles for the kicking leg
        self.knee_angle, self.hip_flexion = leg_angles(wl, kicking_side)

        self.thigh_elevation = thigh_elevation(wl, kicking_side)

        # compute knee angle for standing leg to check for overextension
        standing_side = "right" if kicking_side == "left" else "left"
        self.standing_knee_angle, _ = leg_angles(wl, standing_side)

        raw = hip_kick_angle(wl, kicking_side)
        self.hip_alignment = min(raw, 180 - raw)

        # check y position of kicking knee
        kicking_knee_idx = L_KNEE if kicking_side == "left" else R_KNEE
        self._knee_y = wl[kicking_knee_idx].y

        standing_ankle_idx = L_ANKLE if standing_side == "left" else R_ANKLE
        self._ankle_y = wl[standing_ankle_idx].y

        # compute foot velocity
        foot_idx = L_ANKLE if kicking_side == "left" else R_ANKLE
        pos = np.array([wl[foot_idx].x, wl[foot_idx].y, wl[foot_idx].z])

        if self.prev_foot_pos is not None:
            dist = np.linalg.norm(pos - self.prev_foot_pos) # Meter
            velocity = (dist * self.fps) * 3.6 # km/h

            self.velocity_history.append(velocity)
            smoothed = round(statistics.median(self.velocity_history))

            if self.phase == "chamber" or self.phase == "kick":
                self.max_foot_velocity = max(self.max_foot_velocity, smoothed)
        self.prev_foot_pos = pos


    def _update_idle(self, frame_index):
        self.idle_ankle_history.append(self._ankle_y)

        # detect beginning of kick
        if len(self.idle_ankle_history) > self.BASELINE_WINDOW_FRAMES:
            self.idle_ankle_history.pop(0)
        if len(self.idle_ankle_history) >= self.BASELINE_WINDOW_FRAMES:
            self.baseline_ankle_y = np.mean(self.idle_ankle_history)
        
        if self.baseline_ankle_y is not None and self.kick_start_frame is None and (self.baseline_ankle_y - self._ankle_y) > self.LIFT_THRESHHOLD_M:
            self.kick_start_frame = frame_index

        # change to chamber phase 
        if self.knee_angle < self.CHAMBER_KNEE_MAX and self.thigh_elevation > self.LEG_UP_MIN:
            self._transition_to("chamber", frame_index)


    def _update_chamber(self, frame_index):
        self.min_knee_in_chamber = min(self.min_knee_in_chamber, self.knee_angle)
        self.min_hip_in_chamber = min(self.min_hip_in_chamber, self.hip_flexion)
        self.max_thigh_elevation = max(self.max_thigh_elevation, self.thigh_elevation)
        self.min_knee_y = min(self.min_knee_y, self._knee_y)

        # change to kick phase
        if self.knee_angle > self.KICK_KNEE_MIN and self.thigh_elevation > self.LEG_UP_MIN:
            self._transition_to("kick", frame_index)
        elif self.knee_angle > self.IDLE_KNEE_MIN and self.thigh_elevation < self.THIGH_ELEVATION_FAIL: # Abbruch zurück in Idle, wenn die Person doch nicht kickt sondern z.B. nur das Knie hebt
            self._transition_to("idle", frame_index)
            self.reset()


    def _leg_is_up(self):
        return self.thigh_elevation > self.LEG_UP_MIN

    def _update_kick(self, frame_index):
        if self.knee_angle > self.max_knee_in_kick:
            self.max_knee_in_kick = self.knee_angle
            self.min_knee_after_peak = self.knee_angle

        self.min_knee_y = min(self.min_knee_y, self._knee_y)
        self.max_knee_y_in_kick = max(self.max_knee_y_in_kick, self._knee_y)
        self.standing_knee_history.append(self.standing_knee_angle)

        self.max_hip_alignment = max(self.max_hip_alignment, self.hip_alignment)

        leg_up = self._leg_is_up()
        if leg_up:
            self.min_knee_after_peak = min(self.min_knee_after_peak, self.knee_angle)

        if leg_up and (self.max_knee_in_kick - self.knee_angle) >= self.KNEE_RETURN_MIN:
            self._transition_to("rechamber", frame_index)
        elif self.knee_angle > self.IDLE_KNEE_MIN and self.thigh_elevation < self.THIGH_ELEVATION_FAIL: # Abbruch zurück in Idle, wenn die Person doch nicht kickt sondern z.B. nur das Knie hebt
            self._transition_to("idle", frame_index)
            if self.kick_start_frame is not None:
                self.last_kick_duration_frames = frame_index - self.kick_start_frame
            self.reset()


    def _update_rechamber(self, frame_index):
        self.min_knee_in_rechamber = min(self.min_knee_in_rechamber, self.knee_angle)
        self.min_hip_in_rechamber = min(self.min_hip_in_rechamber, self.hip_flexion)

        if self._leg_is_up():
            self.min_knee_after_peak = min(self.min_knee_after_peak, self.knee_angle)

        # transition back to idle, if the person lowers the leg again
        if self.knee_angle > self.IDLE_KNEE_MIN and self.thigh_elevation < self.THIGH_ELEVATION_FAIL:
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

    @staticmethod # for evaluation
    def _graded(name, label, value, fail_at, ideal_at, ok, fail, value_display=None):
        score = score_linear(value, fail_at, ideal_at)
        passed = score >= 50.0
        return {
            "name": name,
            "score": round(score),
            "label": label,
            "value": value_display if value_display is not None else round(value, 1),
            "passed": passed,
            "feedback": ok if passed else fail
        }
    
    @staticmethod
    def _boolean_criterium(name, label, done, ok, fail):
        return {
            "name": name,
            "label": label,
            "score": 100 if done else 0,
            "value": None,
            "passed": done,
            "feedback": ok if done else fail

        }

    def evaluate(self):
        results = []

        results.append(self._graded(
                    "knee_extension", "Beinstreckung", self.max_knee_in_kick,
                    fail_at=130, ideal_at=160,
                    ok="Bein voll gestreckt.",
                    fail="Bein nicht vollständig gestreckt.",
                ))
        
        if not self.kick_detected:
            return results

        results.append(self._graded(
            "chamber_depth",
            "Chamber Winkel",
            self.min_knee_in_chamber,
            fail_at=self.CHAMBER_KNEE_MAX,
            ideal_at=self.CHAMBER_QUALITY_KNEE_MAX,
            ok="Chamber eng genug.",
            fail="Knie nicht eng genug gezogen.",
        ))

        results.append(self._graded(
            "hip_lift",
            "Hüftbeugung",
            self.max_thigh_elevation,
            fail_at=self.THIGH_ELEVATION_FAIL,
            ideal_at=self.THIGH_ELEVATION_IDEAL,
            ok="Knie ausreichend hoch angezogen.",
            fail="Knie höher anziehen — der Oberschenkel sollte mindestens waagerecht sein.",
        ))


        results.append(self._graded(
            "hip_alignment", "Hüftrotation", self.max_hip_alignment,
            fail_at=90, ideal_at=30,
            ok="Hüfte nicht überdreht.",
            fail="Hüfte zu sehr rotiert. Weniger Hüftrotation für Ap Chagi.",
        ))

        knee_drop = self._knee_drop()
        results.append(self._graded(
            "knee_height", "Kniehöhe", knee_drop,
            fail_at=0.30, ideal_at=0.15, value_display=round(knee_drop, 3),
            ok="Knie bleibt auf Höhe während des Kicks.",
            fail=f"Knie sackt im Kick um {round(knee_drop * 100)} cm ab.",
        ))
        
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


    def _knee_drop(self):
        if self.max_knee_y_in_kick > float("-inf") and self.min_knee_y < float("inf"):
            return self.max_knee_y_in_kick - self.min_knee_y
        return 0.0