from collections import deque
import statistics

from .geometry import L_KNEE, R_KNEE, hip_orientation, angle_difference, leg_angles, L_ANKLE, R_ANKLE, score_linear, pelvis_facing, kick_direction
import numpy as np

# https://en.wikipedia.org/wiki/Roundhouse_kick
# Chamber without Hip Rotation


class BandalChagi:
    NAME = "bandal_chagi"

    # --- Detection ---
    CHAMBER_KNEE_MAX = 130
    CHAMBER_HIP_MAX = 130
    KICK_KNEE_MIN = 150 
    IDLE_KNEE_MIN = 160
    IDLE_HIP_MIN = 160
    RECHAMBER_TOLERANCE_DEG = 30
    LIFT_THRESHHOLD_M = 0.05
    BASELINE_WINDOW_FRAMES = 10

    # Bandal Chagi specific
    HIP_ROTATION_MIN = 100
    KNEE_LATERAL_MIN = 30

    # --- Quality ---
    CHAMBER_QUALITY_KNEE_MAX = 110
    CHAMBER_QUALITY_HIP_MAX = 110
    KNEE_DROP_TOLERANCE_M = 0.15
    STANDING_KNEE_MAX = 175

    def __init__(self):
        # --- phases state ---
        self.phase = "idle"
        self.phases_log = []
        self.kick_start_frame = None
        self.last_kick_duration_frames = None

        # --- active values for current frame ---
        self.knee_angle = 0.0
        self.hip_flexion = 0.0
        self.torso_angle = 0.0
        self.hip_rotation = 0.0
        self.hip_rotation_start = None
        self._knee_y = 0.0
        self._ankle_y = 0.0

        # --- Min/Max for Chamber ---
        self.min_knee_in_chamber = 180.0
        self.min_hip_in_chamber = 180.0

        self.standing_knee_angle = 0.0       # gesetzt in _compute_current_values, max in update
        self.max_abs_hip_rotation = 0.0       # max in _compute_current_values
        self.max_knee_in_kick = 0.0           # max in _update_kick
        self.min_knee_y = float("inf")                 # min in chamber + kick
        self.max_knee_y_in_kick = -float("inf")         # max in _update_kick
        self.max_standing_knee_angle = 0.0    # max in chamber + kick + rechamber
        self.min_knee_in_rechamber = 180.0     # min in _update_rechamber
        self.min_hip_in_rechamber = 180.0      # min in _update_rechamber
        self.idle_ankle_history = []          # genutzt in _update_idle
        self.baseline_ankle_y = None          # genutzt in _update_idle

        self.knee_lateral = 0.0
        self.max_knee_lateral = 0.0

        # hip rotation/alignment
        self.hip_alignment = 0.0
        self.max_hip_alignment = 0.0

        self.max_foot_velocity = 0.0
        self.fps = 30
        self.prev_foot_pos = None
        self.velocity_history = deque(maxlen=3)  # store last 5 foot velocities for smoothing

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

        # compute knee angle for standing leg to check for overextension
        standing_side = "right" if kicking_side == "left" else "left"
        self.standing_knee_angle, _ = leg_angles(wl, standing_side)

        # hip rotation
        if self.hip_rotation_start is None:
            self.hip_rotation_start = hip_orientation(wl)

        # compute hip rotation relative to start
        self.hip_rotation = angle_difference(hip_orientation(wl), self.hip_rotation_start)
        self.max_abs_hip_rotation = max(self.max_abs_hip_rotation, abs(self.hip_rotation))

        raw = abs(angle_difference(pelvis_facing(wl), kick_direction(wl, kicking_side)))
        self.hip_alignment = min(raw, 180 - raw)

        #self.knee_lateral = knee_lateral_angle(wl, kicking_side)
        #self.max_knee_lateral = max(self.max_knee_lateral, self.knee_lateral)

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
        if self.knee_angle < self.CHAMBER_KNEE_MAX and self.hip_flexion < self.CHAMBER_HIP_MAX:
            self._transition_to("chamber", frame_index)


    def _update_chamber(self, frame_index):
        self.min_knee_in_chamber = min(self.min_knee_in_chamber, self.knee_angle)
        self.min_hip_in_chamber = min(self.min_hip_in_chamber, self.hip_flexion)
        self.min_knee_y = min(self.min_knee_y, self._knee_y)
        #self.max_standing_knee_angle = max(self.max_standing_knee_angle, self.standing_knee_angle)

        # change to kick phase
        if self.knee_angle > self.KICK_KNEE_MIN and self.hip_flexion < self.CHAMBER_HIP_MAX and self.hip_alignment >= 30:
            self._transition_to("kick", frame_index)
        elif self.knee_angle > self.IDLE_KNEE_MIN and self.hip_flexion > self.IDLE_HIP_MIN: # Abbruch zurück in Idle, wenn die Person doch nicht kickt sondern z.B. nur das Knie hebt
            self._transition_to("idle", frame_index)
            self.reset()


    def _update_kick(self, frame_index):
        self.max_knee_in_kick = max(self.max_knee_in_kick, self.knee_angle)
        self.min_knee_y = min(self.min_knee_y, self._knee_y)
        self.max_knee_y_in_kick = max(self.max_knee_y_in_kick, self._knee_y)
        self.max_standing_knee_angle = max(self.max_standing_knee_angle, self.standing_knee_angle)

        self.max_hip_alignment = max(self.max_hip_alignment, self.hip_alignment)

        # transition back to chamber if knee and hip angles return to chamber range during kick
        if abs(self.knee_angle - self.min_knee_in_chamber) <= self.RECHAMBER_TOLERANCE_DEG and abs(self.hip_flexion - self.min_hip_in_chamber) <= self.RECHAMBER_TOLERANCE_DEG:
            self._transition_to("rechamber", frame_index)
        elif self.knee_angle > self.IDLE_KNEE_MIN and self.hip_flexion > self.IDLE_HIP_MIN:
            self._transition_to("idle", frame_index)
            if self.kick_start_frame is not None:
                self.last_kick_duration_frames = frame_index - self.kick_start_frame
            self.reset()


    def _update_rechamber(self, frame_index):
        self.min_knee_in_rechamber = min(self.min_knee_in_rechamber, self.knee_angle)
        self.min_hip_in_rechamber = min(self.min_hip_in_rechamber, self.hip_flexion)
        #self.max_standing_knee_angle = max(self.max_standing_knee_angle, self.standing_knee_angle)

        # transition back to idle, if the person lowers the leg again
        if self.knee_angle > self.IDLE_KNEE_MIN and self.hip_flexion > self.IDLE_HIP_MIN:
            self._transition_to("idle", frame_index)
            if self.kick_start_frame is not None:
                self.last_kick_duration_frames = frame_index - self.kick_start_frame
            self.reset()

    def _transition_to(self, new_phase, frame_index):
        self.phase = new_phase
        self.phases_log.append((new_phase, frame_index))

    def reset(self):
        self.kick_start_frame = None
        self.idle_ankle_history = []
        self.baseline_ankle_y = None

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
            self.min_hip_in_chamber,
            fail_at=self.CHAMBER_HIP_MAX,
            ideal_at=self.CHAMBER_QUALITY_HIP_MAX,
            ok="Hüfte ausreichend gehoben.",
            fail="Hüfte nicht weit genug gebeugt",
        ))

        results.append(self._graded(
            "hip_alignment", "Hüftrotation", self.max_hip_alignment,
            fail_at=30, ideal_at=70,
            ok="Hüfte gut rotiert.",
            fail="Hüfte nicht rotiert — beim Bandal muss die Hüfte rotieren.",
        ))

        if self.max_hip_alignment < 30:
            return results

        knee_extended = self.max_knee_in_kick >= self.KICK_KNEE_MIN
        results.append(self._graded(
            "knee_extension", "Beinstreckung", self.max_knee_in_kick,
            fail_at=130, ideal_at=170,
            ok="Bein voll gestreckt.",
            fail="Bein nicht vollständig gestreckt.",
        ))
        if not knee_extended:
            return results


        knee_drop = self._knee_drop()
        results.append(self._graded(
            "knee_height", "Kniehöhe", knee_drop,
            fail_at=0.30, ideal_at=0.15, value_display=round(knee_drop, 3),
            ok="Knie bleibt auf Höhe während des Kicks.",
            fail=f"Knie sackt im Kick um {round(knee_drop * 100)} cm ab.",
        ))

        results.append(self._graded(
            "supporting_leg", "Standbein", self.max_standing_knee_angle,
            fail_at=180, ideal_at=175,
            ok="Balance gut: Standbein leicht gebeugt",
            fail="Standbein durchgestreckt. Eine leichte Beugung verbessert Balance.",
        ))

        rechamber_done = any(p == "rechamber" for (p, _) in self.phases_log)
        results.append(self._boolean_criterium(
            "rechamber", "Rechamber", rechamber_done,
            ok="Rechamber durchgeführt.",
            fail="Rechamber vergessen: Knie nach dem Treffen zurückziehen.",
        ))

        return results


    def _knee_drop(self):
        if self.max_knee_y_in_kick > float("-inf") and self.min_knee_y < float("inf"):
            return self.max_knee_y_in_kick - self.min_knee_y
        return 0.0