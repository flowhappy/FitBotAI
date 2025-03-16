import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

mp_pose = mp.solutions.pose

def detect_squats(video_path, output_video_path, display=False):

    action_state = 'rest'

    KNEE_ANGLE_THRESHOLD_START = 140  # 动作开始阈值
    KNEE_ANGLE_THRESHOLD_END = 160    # 动作结束阈值
    FRAME_COUNT_THRESHOLD = 5         # 连续帧数阈值
    frame_count = 0
    squat_count = 0  # 深蹲次数
    knee_angles = []  # 存储角度的列表，用于计算移动平均
    action_data = []  # 存储动作数据的列表

    # 存储每个阶段的数据
    stage_data = {
        'rest': {'knee_angle': [], 'arm_angles': [], 'leg_angles': [], 'torso_angle': []},
        'start': {'knee_angle': [], 'arm_angles': [], 'leg_angles': [], 'torso_angle': []},
        'moving': {'knee_angle': [], 'arm_angles': [], 'leg_angles': [], 'torso_angle': []},
        'end': {'knee_angle': [], 'arm_angles': [], 'leg_angles': [], 'torso_angle': []},
    }

    font_path = 'msyh.ttf'  # 请确保字体文件在指定路径中
    font = ImageFont.truetype(font_path, 24)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件。")
        return 0, []

    # 获取视频的帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 设置保存视频的参数
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        frame_number = 0  # 用于记录帧数

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("视频处理完毕。")
                break

            frame_number += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False  # 提升性能
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_world_landmarks and results.pose_landmarks:
                landmarks = results.pose_world_landmarks.landmark
                image_landmarks = results.pose_landmarks

                # 绘制不同部位的连线
                draw_torso(image, image_landmarks)
                draw_arms(image, image_landmarks)
                draw_legs(image, image_landmarks)

                # 获取关键点
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

                left_arm_angle = calculate_angle_3d(left_elbow, left_shoulder, left_hip)
                right_arm_angle = calculate_angle_3d(right_elbow, right_shoulder, right_hip)
                left_leg_angle = calculate_angle_3d(left_knee, left_hip, left_shoulder)
                right_leg_angle = calculate_angle_3d(right_knee, right_hip, right_shoulder)
                torso_angle = calculate_torso_angle_3d(left_hip, left_shoulder)

                left_knee_angle = calculate_angle_3d(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle_3d(right_hip, right_knee, right_ankle)
                knee_angle = (left_knee_angle + right_knee_angle) / 2

                knee_angles.append(knee_angle)
                if len(knee_angles) > 5:
                    knee_angles.pop(0)

                average_knee_angle = sum(knee_angles) / len(knee_angles)

                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)

                # 输出文本
                draw.text((50, 150), f'膝关节角度: {int(average_knee_angle)}°', font=font, fill=(255, 255, 0))
                draw.text((50, 200), f'左臂-躯干角度: {int(left_arm_angle)}°', font=font, fill=(255, 255, 255))
                draw.text((50, 230), f'右臂-躯干角度: {int(right_arm_angle)}°', font=font, fill=(255, 255, 255))
                draw.text((50, 260), f'左腿-躯干角度: {int(left_leg_angle)}°', font=font, fill=(255, 255, 255))
                draw.text((50, 290), f'右腿-躯干角度: {int(right_leg_angle)}°', font=font, fill=(255, 255, 255))
                draw.text((50, 320), f'躯干倾斜角度: {int(torso_angle)}°', font=font, fill=(255, 255, 0))
                draw.text((50, 100), f'深蹲次数: {squat_count}', font=font, fill=(0, 0, 255))

                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                # 动作检测逻辑
                if action_state == 'rest':
                    if average_knee_angle < KNEE_ANGLE_THRESHOLD_START:
                        frame_count += 1
                        if frame_count >= FRAME_COUNT_THRESHOLD:
                            action_state = 'start'
                            frame_count = 0
                            start_frame = frame_number
                            print(f"第 {squat_count + 1} 次深蹲开始于第 {start_frame} 帧")
                            current_action_data = []
                            stage_data['rest']['knee_angle'].append(average_knee_angle)
                            stage_data['rest']['arm_angles'].append((left_arm_angle, right_arm_angle))
                            stage_data['rest']['leg_angles'].append((left_leg_angle, right_leg_angle))
                            stage_data['rest']['torso_angle'].append(torso_angle)

                    else:
                        frame_count = 0

                elif action_state == 'start':
                    action_state = 'moving'
                    stage_data['start']['knee_angle'].append(average_knee_angle)
                    stage_data['start']['arm_angles'].append((left_arm_angle, right_arm_angle))
                    stage_data['start']['leg_angles'].append((left_leg_angle, right_leg_angle))
                    stage_data['start']['torso_angle'].append(torso_angle)

                elif action_state == 'moving':
                    frame_data = {
                        'left_arm_angle': left_arm_angle,
                        'right_arm_angle': right_arm_angle,
                        'left_leg_angle': left_leg_angle,
                        'right_leg_angle': right_leg_angle,
                        'torso_angle': torso_angle,
                        'average_knee_angle': average_knee_angle
                    }
                    current_action_data.append(frame_data)

                    # stage_data['moving']['knee_angle'].append(average_knee_angle)
                    # stage_data['moving']['arm_angles'].append((left_arm_angle, right_arm_angle))
                    # stage_data['moving']['leg_angles'].append((left_leg_angle, right_leg_angle))
                    # stage_data['moving']['torso_angle'].append(torso_angle)

                    if average_knee_angle > KNEE_ANGLE_THRESHOLD_END:
                        frame_count += 1
                        if frame_count >= FRAME_COUNT_THRESHOLD:
                            action_state = 'end'
                            frame_count = 0
                            end_frame = frame_number
                            squat_count += 1
                            print(f"第 {squat_count} 次深蹲结束于第 {end_frame} 帧")
                            action_data.extend(current_action_data)

                elif action_state == 'end':
                    action_state = 'rest'
                    stage_data['end']['knee_angle'].append(average_knee_angle)
                    stage_data['end']['arm_angles'].append((left_arm_angle, right_arm_angle))
                    stage_data['end']['leg_angles'].append((left_leg_angle, right_leg_angle))
                    stage_data['end']['torso_angle'].append(torso_angle)

                # 输出每个阶段的平均值
                if action_state != 'rest':
                    if len(stage_data[action_state]['knee_angle']) > 0:
                        avg_knee = np.mean(stage_data[action_state]['knee_angle'])
                        print(f"阶段 {action_state} 膝关节角度平均值: {avg_knee:.2f}°")
                    if len(stage_data[action_state]['arm_angles']) > 0:
                        avg_arm = np.mean(stage_data[action_state]['arm_angles'], axis=0)
                        print(f"阶段 {action_state} 左/右臂角度平均值: {avg_arm[0]:.2f}°, {avg_arm[1]:.2f}°")
                    if len(stage_data[action_state]['leg_angles']) > 0:
                        avg_leg = np.mean(stage_data[action_state]['leg_angles'], axis=0)
                        print(f"阶段 {action_state} 左/右腿角度平均值: {avg_leg[0]:.2f}°, {avg_leg[1]:.2f}°")
                    if len(stage_data[action_state]['torso_angle']) > 0:
                        avg_torso = np.mean(stage_data[action_state]['torso_angle'])
                        print(f"阶段 {action_state} 躯干角度平均值: {avg_torso:.2f}°")

                # 将帧写入输出视频文件
                out.write(image)

            else:
                print(f"在第 {frame_number} 帧未检测到关键点。")

        print(f"总共完成了 {squat_count} 次深蹲。")
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    return squat_count, action_data



def main():
    video_path = "testVideo/1.mp4"
    output_video_path = "output_video.mp4"  # 保存输出的视频文件路径
    total_squats, action_data = detect_squats(video_path, output_video_path)

    print(f"总共完成了 {total_squats} 次深蹲。")


if __name__ == "__main__":
    main()
