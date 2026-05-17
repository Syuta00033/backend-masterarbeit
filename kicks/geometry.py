import numpy as np

L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


def landmark_to_array(landmark):
    return np.array([landmark.x, landmark.y, landmark.z])


def calc_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))


def leg_angles(wl, side):
    if side == "left":
        hip = landmark_to_array(wl[L_HIP])
        knee = landmark_to_array(wl[L_KNEE])
        ankle = landmark_to_array(wl[L_ANKLE])
        shoulder = landmark_to_array(wl[L_SHOULDER])
    else:
        hip = landmark_to_array(wl[R_HIP])
        knee = landmark_to_array(wl[R_KNEE])
        ankle = landmark_to_array(wl[R_ANKLE])
        shoulder = landmark_to_array(wl[R_SHOULDER])

    knee_angle = calc_angle(hip, knee, ankle)
    hip_flexion = calc_angle(shoulder, hip, knee)
    return knee_angle, hip_flexion
