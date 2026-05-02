import cv2
import numpy as np


class KickAnalyzer:
    def __init__(self):
        self.phase = "idle"
        self.baseline_hip_y = None
        self.kick_label_frames = 0
        self.phases_log = []

    def process_frame(self, result, out_w, out_h, annotated):
        if not result.pose_landmarks:
            return annotated

        hip = (result.pose_landmarks[0][24].x, result.pose_landmarks[0][24].y)
        knee = (result.pose_landmarks[0][26].x, result.pose_landmarks[0][26].y)
        ankle = (result.pose_landmarks[0][28].x, result.pose_landmarks[0][28].y)
        shoulder = (result.pose_landmarks[0][12].x, result.pose_landmarks[0][12].y)

        knee_angle = self.calc_angle(hip, knee, ankle)
        hip_flexion = self.calc_angle(shoulder, hip, knee)

        self.update(knee_angle, hip_flexion, frame_idx=0)

        cv2.putText(annotated, self.phase, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.putText(annotated, f"Knee Angle: {int(knee_angle)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Hip Flexion: {int(hip_flexion)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)




        # if left_knee.y < left_hip.y - 0.05 and self.kick_label_frames == 0:
        #     self.kick_label_frames = 30

        # if self.kick_label_frames > 0:
        #     cv2.putText(annotated, "Kick erkannt!", (10, 110),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        #     self.kick_label_frames -= 1

        return annotated
    

    def calc_angle(self, a, b, c):
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle

    def update(self, knee_angle, hip_flexion, frame_idx):
        if self.phase =="idle":
            if knee_angle < 130 and hip_flexion < 130:
                self.phase = "chamber"
                self.phases_log.append(("chamber", frame_idx))
        elif self.phase == "chamber":
            if knee_angle > 150 and hip_flexion < 130:
                self.phase = "kick"
                self.phases_log.append(("kick", frame_idx))
            elif knee_angle > 160 and hip_flexion > 160:
                self.phase = "idle"
                self.phases_log.append(("idle", frame_idx))
        elif self.phase == "kick":
            if knee_angle < 130 and hip_flexion < 130:
                self.phase = "rechamber"
                self.phases_log.append(("rechamber", frame_idx))
        elif self.phase == "rechamber":
            if knee_angle > 160 and hip_flexion > 160:
                self.phase = "idle"
                self.phases_log.append(("idle", frame_idx))