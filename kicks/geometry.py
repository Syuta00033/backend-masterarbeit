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


# calculate hip rotation based on shoulder and hip landmarks
def hip_rotation(wl):
    ls = np.array([wl[11].x, wl[11].z]) # left shoulder
    rs = np.array([wl[12].x, wl[12].z]) # right shoulder
    lh = np.array([wl[23].x, wl[23].z]) # left hip
    rh = np.array([wl[24].x, wl[24].z]) # right hip

    shoulder_vec = rs - ls
    hip_vec = rh - lh

    # Calculate the angle between the shoulder vector and hip vector
    shoulder_angle = np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0]))
    hip_angle = np.degrees(np.arctan2(hip_vec[1], hip_vec[0]))

    # Positive value means right rotation, negative means left rotation
    return float(hip_angle - shoulder_angle)


def score_linear(value, fail_at, ideal_at):
    t = (value - fail_at) / (ideal_at - fail_at) # normalize to 0-1 range
    return float(max(0.0, min(1.0, t)) * 100 ) # convert to percentage