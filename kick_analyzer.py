import cv2

from kicks import KICK_CLASSES
from kicks.geometry import L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE


class KickAnalyzer:

    SIDE_LIFT_THRESHOLD = 0.05 # 5cm
    HUD_REFERENCE_SIZE = 540 

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


        s = min(annotated.shape[0], annotated.shape[1]) / self.HUD_REFERENCE_SIZE
        thick = max(1, round(2 * s))

        def text(msg, y, color=(255, 255, 255), size=0.8):
            cv2.putText(annotated, msg, (10, int(y * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, size * s, color, thick, cv2.LINE_AA)

        text(self.kick.phase, 30, (0, 255, 0), 1.0)
        text(f"Knee Angle: {int(self.kick.knee_angle)}", 60)
        text(f"Hip Flexion: {int(self.kick.hip_flexion)}", 90)
        text(f"Side: {self.kicking_side}", 120)

        if self.kick.NAME != "dwit_chagi":
            text(f"kick thigh angle: {int(self.kick.thigh_elevation)}", 150)

        if self.kick.kick_start_frame is not None:
            text(f"Start Frame: {self.kick.kick_start_frame}", 180)

        if self.kick.last_kick_duration_frames is not None:
            text(f"Last Kick: {self.kick.last_kick_duration_frames} frames", 210, (0, 255, 255))

        # Visibility der Bein-Landmarks
        leg_vis = [wl[i].visibility for i in (L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)]
        min_vis = min(leg_vis)
        avg_vis = sum(leg_vis) / len(leg_vis)
        if min_vis >= 0.7:
            vis_color = (0, 255, 0)      # gruen: ok
        elif min_vis >= 0.5:
            vis_color = (255, 255, 0)    # gelb: grenzwertig
        else:
            vis_color = (255, 0, 0)      # rot: unzuverlaessig
        text(f"Visibility: min {min_vis:.2f}  avg {avg_vis:.2f}", 240, vis_color)

        text(f"Hip Alignment: {int(self.kick.hip_alignment)}", 270, (0, 255, 255))
        text(f"Impact Align: {int(self.kick.hip_alignment_at_impact)}", 300, (0, 255, 255))

        if self.kick.NAME == "dwit_chagi":
            text(f"Over-Rot: {int(self.kick.max_over_rotation)}", 360, (0, 200, 255))
            text(f"Total-Rot: {int(self.kick.total_rotation)}", 390, (0, 200, 255))
            text(f"Foot Gap: {self.kick._foot_gap:.2f}  (max {self.kick.max_foot_gap_in_kick:.2f})", 420, (0, 200, 255), 0.7)
            text(f"Foot Height: {self.kick._foot_height:.2f}  Gap-H: {self.kick._foot_gap_h:.2f}", 450, (0, 200, 255), 0.7)

        return annotated

    def _select_side(self, wl):
        if self.kick.phase != "idle":
            return

        diff = wl[R_KNEE].y - wl[L_KNEE].y

        if abs(diff) < self.SIDE_LIFT_THRESHOLD:
            return

        self.kicking_side = "right" if diff < 0 else "left"
