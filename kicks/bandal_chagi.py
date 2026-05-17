from .geometry import leg_angles


class BandalChagi:
    NAME = "bandal_chagi"

    def __init__(self):
        self.phase = "idle"
        self.phases_log = []
        self._kick_start_frame = None
        self.last_kick_duration_frames = None
        self.knee_angle = 0.0
        self.hip_flexion = 0.0

    def update(self, wl, kicking_side, frame_index):
        self.knee_angle, self.hip_flexion = leg_angles(wl, kicking_side)

    def evaluate(self):
        return []
