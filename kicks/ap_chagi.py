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

    def update(self, knee_angle, hip_flexion, frame_index):
        if self.phase == "idle":
            if knee_angle < self.CHAMBER_KNEE_MAX and hip_flexion < self.CHAMBER_HIP_MAX:
                self.phase = "chamber"
                self.phases_log.append(("chamber", frame_index))
                self._kick_start_frame = frame_index

        elif self.phase == "chamber":
            if knee_angle > self.KICK_KNEE_MIN and hip_flexion < self.CHAMBER_HIP_MAX:
                self.phase = "kick"
                self.phases_log.append(("kick", frame_index))
            elif knee_angle > self.IDLE_KNEE_MIN and hip_flexion > self.IDLE_HIP_MIN:
                self.phase = "idle"
                self.phases_log.append(("idle", frame_index))
                self._kick_start_frame = None

        elif self.phase == "kick":
            if knee_angle < self.CHAMBER_KNEE_MAX and hip_flexion < self.CHAMBER_HIP_MAX:
                self.phase = "rechamber"
                self.phases_log.append(("rechamber", frame_index))
            elif knee_angle > self.IDLE_KNEE_MIN and hip_flexion > self.IDLE_HIP_MIN:
                self.phase = "idle"
                self.phases_log.append(("idle", frame_index))
                if self._kick_start_frame is not None:
                    self.last_kick_duration_frames = frame_index - self._kick_start_frame
                    self._kick_start_frame = None

        elif self.phase == "rechamber":
            if knee_angle > self.IDLE_KNEE_MIN and hip_flexion > self.IDLE_HIP_MIN:
                self.phase = "idle"
                self.phases_log.append(("idle", frame_index))
                if self._kick_start_frame is not None:
                    self.last_kick_duration_frames = frame_index - self._kick_start_frame
                    self._kick_start_frame = None

    def evaluate(self):
        return []
