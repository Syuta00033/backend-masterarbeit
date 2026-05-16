import cv2
import numpy as np

from kicks import KICK_CLASSES


class KickAnalyzer:

    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28

    def __init__(self, kick_type="ap_chagi"):
        if kick_type not in KICK_CLASSES:
            raise ValueError(f"Unbekannter Kick-Typ: {kick_type}")
        self.kick = KICK_CLASSES[kick_type]()
        self.kicking_side = None

    def process_frame(self, result, annotated, frame_index):
        if not result.pose_world_landmarks:
            return annotated

        wl = result.pose_world_landmarks[0]

        self.kicking_side = self._select_side(wl)
        knee_angle, hip_flexion = self._leg_angles(wl, self.kicking_side)

        self.kick.update(knee_angle, hip_flexion, frame_index)

        cv2.putText(annotated, self.kick.phase, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Knee Angle: {int(knee_angle)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Hip Flexion: {int(hip_flexion)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Side: {self.kicking_side}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if self.kick.last_kick_duration_frames is not None:
            cv2.putText(annotated, f"Last Kick: {self.kick.last_kick_duration_frames} frames",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        return annotated


    def _landmark(self, landmark):
        return np.array([landmark.x, landmark.y, landmark.z])

    def _select_side(self, wl):
        if self.kick.phase != "idle":
            return self.kicking_side

        left_knee_y = wl[self.L_KNEE].y
        right_knee_y = wl[self.R_KNEE].y

        if left_knee_y < right_knee_y:
            self.kicking_side = "left"
        else:
            self.kicking_side = "right"

        return self.kicking_side

    def _leg_angles(self, wl, side):
        if side == "left":
            hip = self._landmark(wl[self.L_HIP])
            knee = self._landmark(wl[self.L_KNEE])
            ankle = self._landmark(wl[self.L_ANKLE])
            shoulder = self._landmark(wl[self.L_SHOULDER])
        else:
            hip = self._landmark(wl[self.R_HIP])
            knee = self._landmark(wl[self.R_KNEE])
            ankle = self._landmark(wl[self.R_ANKLE])
            shoulder = self._landmark(wl[self.R_SHOULDER])

        knee_angle = self.calc_angle(hip, knee, ankle)
        hip_flexion = self.calc_angle(shoulder, hip, knee)

        return knee_angle, hip_flexion

    def calc_angle(self, a, b, c):
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle
