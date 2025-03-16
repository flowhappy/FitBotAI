import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class SquatDetector:
    def __init__(self,
                 knee_angle_threshold_start=140,
                 knee_angle_threshold_end=160,
                 frame_count_threshold=5,
                 font_path='msyh.ttc',
                 model_complexity=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        # 初始化参数
        self.KNEE_ANGLE_THRESHOLD_START = knee_angle_threshold_start
        self.KNEE_ANGLE_THRESHOLD_END = knee_angle_threshold_end
        self.FRAME_COUNT_THRESHOLD = frame_count_threshold
        self.font = ImageFont.truetype(font_path, 24) if font_path else None

        # 初始化MediaPipe模型
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # 状态变量
        self.action_state = 'rest'
        self.squat_count = 0
        self.frame_count = 0
        self.knee_angles = []
        self.action_data = []
        self.stage_data = {
            'rest': {'knee_angle': [], 'arm_angles': [], 'leg_angles': [], 'torso_angle': []},
            'start': {'knee_angle': [], 'arm_angles': [], 'leg_angles': [], 'torso_angle': []},
            'moving': {'knee_angle': [], 'arm_angles': [], 'leg_angles': [], 'torso_angle': []},
            'end': {'knee_angle': [], 'arm_angles': [], 'leg_angles': [], 'torso_angle': []},
        }

    def _calculate_angle_3d(self, a, b, c):
        """计算三维空间中三个点之间的角度"""
        a = np.array([a.x, a.y, a.z])
        b = np.array([b.x, b.y, b.z])
        c = np.array([c.x, c.y, c.z])

        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cosine_angle))

    def _calculate_torso_angle_3d(self, hip, shoulder):
        """计算躯干倾斜角度"""
        hip = np.array([hip.x, hip.y, hip.z])
        shoulder = np.array([shoulder.x, shoulder.y, shoulder.z])
        vector = shoulder - hip
        vertical = np.array([0, -1, 0])

        cosine_angle = np.dot(vector, vertical) / (np.linalg.norm(vector) * np.linalg.norm(vertical))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cosine_angle))

    def _draw_skeleton(self, image, landmarks):
        """绘制骨架连接线"""
        # 绘制躯干
        connections = [
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP),
        ]
        for conn in connections:
            self._draw_line(image, landmarks, conn, (0, 255, 0))

        # 绘制手臂
        arm_connections = [
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
            (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            (self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST),
        ]
        for conn in arm_connections:
            self._draw_line(image, landmarks, conn, (255, 0, 0))

        # 绘制腿部
        leg_connections = [
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
            (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE),
        ]
        for conn in leg_connections:
            self._draw_line(image, landmarks, conn, (0, 0, 255))

    def _draw_line(self, image, landmarks, connection, color):
        """辅助绘制单条连接线"""
        start = landmarks.landmark[connection[0].value]
        end = landmarks.landmark[connection[1].value]
        start_point = (int(start.x * image.shape[1]), int(start.y * image.shape[0]))
        end_point = (int(end.x * image.shape[1]), int(end.y * image.shape[0]))
        cv2.line(image, start_point, end_point, color, 4)

    def _update_display_info(self, image, **kwargs):
        """更新屏幕显示信息"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        y_position = 100
        for text in [
            f'深蹲次数: {self.squat_count}',
            f'膝关节角度: {int(kwargs.get("avg_knee_angle", 0))}°',
            f'左臂-躯干角度: {int(kwargs.get("left_arm_angle", 0))}°',
            f'右臂-躯干角度: {int(kwargs.get("right_arm_angle", 0))}°',
            f'左腿-躯干角度: {int(kwargs.get("left_leg_angle", 0))}°',
            f'右腿-躯干角度: {int(kwargs.get("right_leg_angle", 0))}°',
            f'躯干倾斜角度: {int(kwargs.get("torso_angle", 0))}°',
        ]:
            draw.text((50, y_position), text, font=self.font, fill=(255, 255, 0))
            y_position += 30

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def process_video(self, input_path, output_path):
        """处理视频的主方法"""
        print(input_path)
        cap = cv2.VideoCapture(input_path)
        print(cap.isOpened())
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件,{cap}")
        # 获取视频参数
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)

            if results.pose_world_landmarks:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self._draw_skeleton(image, results.pose_landmarks)

                # 计算关键点角度
                landmarks = results.pose_world_landmarks.landmark
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

                # 计算各类角度
                left_arm_angle = self._calculate_angle_3d(
                    landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                    left_shoulder,
                    left_hip
                )
                right_arm_angle = self._calculate_angle_3d(
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                    right_shoulder,
                    right_hip
                )
                left_knee_angle = self._calculate_angle_3d(
                    left_hip,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                    landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
                )
                right_knee_angle = self._calculate_angle_3d(
                    right_hip,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                )
                torso_angle = self._calculate_torso_angle_3d(left_hip, left_shoulder)
                avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

                # 更新角度数据
                self.knee_angles.append(avg_knee_angle)
                if len(self.knee_angles) > 5:
                    self.knee_angles.pop(0)
                smoothed_knee = sum(self.knee_angles) / len(self.knee_angles)

                # 更新显示信息
                image = self._update_display_info(
                    image,
                    avg_knee_angle=smoothed_knee,
                    left_arm_angle=left_arm_angle,
                    right_arm_angle=right_arm_angle,
                    left_leg_angle=self._calculate_angle_3d(
                        landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                        left_hip,
                        left_shoulder
                    ),
                    right_leg_angle=self._calculate_angle_3d(
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
                        right_hip,
                        right_shoulder
                    ),
                    torso_angle=torso_angle
                )

                # 状态机逻辑
                self._update_state_machine(smoothed_knee, frame_number)

            out.write(image)

        cap.release()
        out.release()
        return self.squat_count, self.action_data

    def _update_state_machine(self, knee_angle, frame_number):
        """更新状态机逻辑"""
        if self.action_state == 'rest':
            if knee_angle < self.KNEE_ANGLE_THRESHOLD_START:
                self.frame_count += 1
                if self.frame_count >= self.FRAME_COUNT_THRESHOLD:
                    self.action_state = 'start'
                    self.frame_count = 0
            else:
                self.frame_count = 0

        elif self.action_state == 'start':
            self.action_state = 'moving'

        elif self.action_state == 'moving':
            if knee_angle > self.KNEE_ANGLE_THRESHOLD_END:
                self.frame_count += 1
                if self.frame_count >= self.FRAME_COUNT_THRESHOLD:
                    self.action_state = 'end'
                    self.frame_count = 0
                    self.squat_count += 1

        elif self.action_state == 'end':
            self.action_state = 'rest'

    def get_detailed_statistics(self):
        """获取详细的统计信息"""
        return {
            'total_squats': self.squat_count,
            'stage_data': self.stage_data,
            'action_data': self.action_data
        }


if __name__ == "__main__":
    detector = SquatDetector()
    count, data = detector.process_video("testVideo/1.mp4", "output_video.mp4")
    print(f"检测到深蹲次数: {count}")
