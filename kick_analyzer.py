import cv2


class KickAnalyzer:
    def __init__(self):
        self.frames_data = []
        self.phase = "idle"
        self.baseline_hip_y = None
        self.kick_label_frames = 0

    def process_frame(self, result, out_w, out_h, annotated):
        if not result.pose_landmarks:
            return annotated

        left_knee = result.pose_landmarks[0][25]
        left_hip = result.pose_landmarks[0][23]
        left_ankle = result.pose_landmarks[0][29]
        left_toe = result.pose_landmarks[0][31]

        self.frames_data.append({
            "knee": (left_knee.x, left_knee.y),
            "hip": (left_hip.x, left_hip.y),
            "ankle": (left_ankle.x, left_ankle.y),
            "toe": (left_toe.x, left_toe.y),
        })

        if left_knee.y < left_hip.y - 0.05 and self.kick_label_frames == 0:
            self.kick_label_frames = 30

        if self.kick_label_frames > 0:
            cv2.putText(annotated, "Kick erkannt!", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            self.kick_label_frames -= 1

        return annotated
    