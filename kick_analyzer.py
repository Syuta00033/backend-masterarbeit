import cv2
import numpy as np


class KickAnalyzer:

    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28


    def __init__(self):
        self.phase = "idle"
        self.baseline_hip_y = None
        self.kick_label_frames = 0
        self.phases_log = []
        self.kicking_side = None  # "left" or "right"
        self._kick_start_frame = None
        self.last_kick_duration_frames = None

    def process_frame(self, result, out_w, out_h, annotated, frame_index):
        if not result.pose_world_landmarks:
            return annotated

        wl = result.pose_world_landmarks[0]

        self.kicking_side = self._select_side(wl)
        knee_angle, hip_flexion = self._leg_angles(wl, self.kicking_side)

        self.update(knee_angle, hip_flexion, frame_index)

        cv2.putText(annotated, self.phase, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(annotated, f"Knee Angle: {int(knee_angle)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Hip Flexion: {int(hip_flexion)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Side: {self.kicking_side}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if self.last_kick_duration_frames is not None:
            cv2.putText(annotated, f"Last Kick: {self.last_kick_duration_frames} frames",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        return annotated


    # helper methods

    def _landmark(self, landmark):
        return np.array([landmark.x, landmark.y, landmark.z])

    def _select_side(self, wl):
        if self.phase != "idle":
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

    def update(self, knee_angle, hip_flexion, frame_index):
        if self.phase =="idle":
            if knee_angle < 130 and hip_flexion < 130:
                self.phase = "chamber"
                self.phases_log.append(("chamber", frame_index))
                self._kick_start_frame = frame_index
        elif self.phase == "chamber":
            if knee_angle > 150 and hip_flexion < 130:
                self.phase = "kick"
                self.phases_log.append(("kick", frame_index))
            elif knee_angle > 160 and hip_flexion > 160:
                self.phase = "idle"
                self.phases_log.append(("idle", frame_index))
                self._kick_start_frame = None 
        elif self.phase == "kick":
            if knee_angle < 130 and hip_flexion < 130:
                self.phase = "rechamber"
                self.phases_log.append(("rechamber", frame_index))
        elif self.phase == "rechamber":
            if knee_angle > 160 and hip_flexion > 160:
                self.phase = "idle"
                self.phases_log.append(("idle", frame_index))
                if self._kick_start_frame is not None:
                    self.last_kick_duration_frames = frame_index - self._kick_start_frame
                    self._kick_start_frame = None
