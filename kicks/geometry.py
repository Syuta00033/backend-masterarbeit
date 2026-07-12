import numpy as np

L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# convert mediapipe landmark to numpy array
def landmark_to_array(landmark):
    return np.array([landmark.x, landmark.y, landmark.z])


# calculate angle between three points
def calc_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))


# calculate knee angle and hip flexion for given leg
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

def angle_difference(a, b):
    return (a - b + 180) % 360 - 180

def score_linear(value, fail_at, ideal_at):
    t = (value - fail_at) / (ideal_at - fail_at) # normalize to 0-1 range
    return float(max(0.0, min(1.0, t)) * 100 ) # convert to percentage


def pelvis_facing(wl):
    lh = np.array([wl[L_HIP].x, wl[L_HIP].z])
    rh = np.array([wl[R_HIP].x, wl[R_HIP].z])
    hip_vec = rh - lh
    facing = np.array([-hip_vec[1], hip_vec[0]])  # senkrecht zur Hüftlinie
    return float(np.degrees(np.arctan2(facing[1], facing[0])))


def kick_direction(wl, kicking_side):
    hip_idx = L_HIP if kicking_side == "left" else R_HIP
    ankle_idx = L_ANKLE if kicking_side == "left" else R_ANKLE
    dx = wl[ankle_idx].x - wl[hip_idx].x
    dz = wl[ankle_idx].z - wl[hip_idx].z
    return float(np.degrees(np.arctan2(dz, dx)))