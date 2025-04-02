import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def calculate_angle_3d(a, b, c):
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])

    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_torso_angle_3d(hip, shoulder):
    hip = np.array([hip.x, hip.y, hip.z])
    shoulder = np.array([shoulder.x, shoulder.y, shoulder.z])
    vector = shoulder - hip  # 躯干向量（从髋部到肩膀）
    vertical = np.array([0, -1, 0])  # 垂直向上方向的单位向量

    # 计算夹角
    cosine_angle = np.dot(vector, vertical) / (np.linalg.norm(vector) * np.linalg.norm(vertical))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def draw_torso(image, landmarks):
    torso_connections = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    ]
    for connection in torso_connections:
        start = landmarks.landmark[connection[0].value]
        end = landmarks.landmark[connection[1].value]
        start_point = (int(start.x * image.shape[1]), int(start.y * image.shape[0]))
        end_point = (int(end.x * image.shape[1]), int(end.y * image.shape[0]))
        cv2.line(image, start_point, end_point, (0, 255, 0), 4)

def draw_arms(image, landmarks):
    arm_connections = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    ]
    for connection in arm_connections:
        start = landmarks.landmark[connection[0].value]
        end = landmarks.landmark[connection[1].value]
        start_point = (int(start.x * image.shape[1]), int(start.y * image.shape[0]))
        end_point = (int(end.x * image.shape[1]), int(end.y * image.shape[0]))
        cv2.line(image, start_point, end_point, (255, 0, 0), 4)

def draw_legs(image, landmarks):
    leg_connections = [
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
        (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
        (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    ]
    for connection in leg_connections:
        start = landmarks.landmark[connection[0].value]
        end = landmarks.landmark[connection[1].value]
        start_point = (int(start.x * image.shape[1]), int(start.y * image.shape[0]))
        end_point = (int(end.x * image.shape[1]), int(end.y * image.shape[0]))
        cv2.line(image, start_point, end_point, (0, 0, 255), 4)





