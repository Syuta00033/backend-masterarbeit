from .geometry import calc_angle, leg_angles
import numpy as np


class ApChagi:

    NAME = "ap_chagi"

    CHAMBER_KNEE_MAX = 130
    CHAMBER_HIP_MAX = 130
    KICK_KNEE_MIN = 150
    IDLE_KNEE_MIN = 160
    IDLE_HIP_MIN = 160

    def __init__(self):
        self.phase = "idle"
        self.phases_log = []
        self._kick_start_frame = None
        self.last_kick_duration_frames = None
        self.knee_angle = 0.0
        self.hip_flexion = 0.0
        self.torso_angle = 0.0

    def update(self, wl, kicking_side, frame_index):
        self.knee_angle, self.hip_flexion = leg_angles(wl, kicking_side)
        print(wl[23])
        #self.torso_angle = calc_angle(wl[11], wl[23], wl[23] + np.array([0, 1, 0]))  # Schulter-Hüfte-Knie

        if self.phase == "idle":
            if self.knee_angle < self.CHAMBER_KNEE_MAX and self.hip_flexion < self.CHAMBER_HIP_MAX:
                self.phase = "chamber"
                self.phases_log.append(("chamber", frame_index))
                self._kick_start_frame = frame_index

        elif self.phase == "chamber":
            if self.knee_angle > self.KICK_KNEE_MIN and self.hip_flexion < self.CHAMBER_HIP_MAX:
                self.phase = "kick"
                self.phases_log.append(("kick", frame_index))
            elif self.knee_angle > self.IDLE_KNEE_MIN and self.hip_flexion > self.IDLE_HIP_MIN:
                self.phase = "idle"
                self.phases_log.append(("idle", frame_index))
                self._kick_start_frame = None

        elif self.phase == "kick":
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
        return []
