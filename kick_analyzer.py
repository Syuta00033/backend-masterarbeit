import cv2

from kicks import KICK_CLASSES
from kicks.geometry import L_KNEE, R_KNEE


class KickAnalyzer:

    SIDE_LIFT_THRESHOLD = 0.05 # 5cm

    def __init__(self, kick_type="ap_chagi"):
        if kick_type not in KICK_CLASSES:
            raise ValueError(f"Unbekannter Kick-Typ: {kick_type}")
        self.kick = KICK_CLASSES[kick_type]()
        self.kicking_side = None

    def process_frame(self, result, annotated, frame_index):
        if not result.pose_world_landmarks:
            return annotated

        wl = result.pose_world_landmarks[0]

        self._select_side(wl)

        if self.kicking_side is not None:
            self.kick.update(wl, self.kicking_side, frame_index)

        cv2.putText(annotated, self.kick.phase, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Knee Angle: {int(self.kick.knee_angle)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Hip Flexion: {int(self.kick.hip_flexion)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Side: {self.kicking_side}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Torso Angle: {int(self.kick.torso_angle)}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        if self.kick.kick_start_frame is not None:
            cv2.putText(annotated, f"Start Frame: {self.kick.kick_start_frame}", (10, 180),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if self.kick.last_kick_duration_frames is not None:
            cv2.putText(annotated, f"Last Kick: {self.kick.last_kick_duration_frames} frames",
                        (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(annotated, f"Hip Rotation: {int(self.kick.hip_rotation)}", (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(annotated, f"Hip Alignment: {int(self.kick.hip_alignment)}", (10, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Max Alignment: {int(self.kick.max_hip_alignment)}", (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        return annotated

    def _select_side(self, wl):
        if self.kick.phase != "idle":
            return
        
        diff = wl[R_KNEE].y - wl[L_KNEE].y

        if abs(diff) < self.SIDE_LIFT_THRESHOLD:
            return

        self.kicking_side = "right" if diff < 0 else "left"
