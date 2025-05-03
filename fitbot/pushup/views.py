import datetime
import subprocess
import requests
from django.http import HttpResponse
from .BasicPoseFun import calculate_torso_angle_3d, calculate_angle_3d, draw_torso, draw_legs, draw_arms
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许所有源访问

mp_pose = mp.solutions.pose

# 标准角度定义
STANDARD_PUSHUP_ANGLES = {
    'start': {  # 俯卧撑起始姿势 (手臂伸直)
        'elbow': 135,    # 肘关节接近伸直 (例如 170° ± 10-15°)
        'torso': 90      # 躯干接近水平 (相对于垂直，例如 90° ± 5-10°)
    },
    'end': {    # 俯卧撑最低点 (胸部接近地面)
        'elbow': 85,     # 肘关节弯曲约90° (例如 90° ± 10-15°)
        'torso': 90      # 躯干仍接近水平 (相对于垂直，例如 90° ± 5-10°)
    }
}

# 对应俯卧撑的标准角度容差
TOLERANCES = {
    'start': {
        'elbow': 8,  # 起始姿势肘关节容差 (例如 170 +/- 8 度)
        'torso': 8   # 起始姿势躯干角度容差 (例如 90 +/- 8 度)
    },
    'end': {
        'elbow': 15, # 结束姿势肘关节容差 (例如 90 +/- 15 度)
        'torso': 8   # 结束姿势躯干角度容差 (例如 90 +/- 8 度)
    }
}


# 定义一个最大允许偏差的乘数，用于计算准确率得分
# 例如，如果偏差超过容差的 2.5 倍，准确率得分就为 0
MAX_DEVIATION_MULTIPLIER = 10

def calculate_score(deviation, tolerance, max_deviation_multiplier):
    """根据偏差计算单项得分 (0.0 到 1.0)"""
    if deviation <= tolerance:
        # 在容差范围内，满分
        return 1.0
    else:
        # 计算最大允许偏差
        max_allowed_deviation = tolerance * max_deviation_multiplier
        if deviation >= max_allowed_deviation:
            # 偏差过大，得分为 0
            return 0.0
        else:
            # 偏差在容差和最大允许偏差之间，得分线性下降
            # score = 1.0 - (当前超出容差的部分 / 容差到最大偏差的总范围)
            score = 1.0 - (deviation - tolerance) / (max_allowed_deviation - tolerance)
            # 确保得分不会低于0 (虽然理论上不会)
            return max(0.0, score)


def analyze_pushup_quality(user_data):
    """智能分析俯卧撑质量"""
    feedback = []
    total_score = 0.0  # 用于累加各项得分
    total_checks = 0  # 用于记录总共执行的检查项目数

    # --- 检查 1: 起始肘关节角度 (手臂是否伸直) ---
    # 检查 user_data['start'] 中是否有 'elbow_angle' 数据
    # 假设 detect_pushups 会在 stage_data['start'] 中存储 'elbow_angle' 列表
    if 'elbow_angle' in user_data.get('start', {}) and user_data['start']['elbow_angle']:
        user_elbow_start = np.mean(user_data['start']['elbow_angle'])
        std_elbow_start = STANDARD_PUSHUP_ANGLES['start']['elbow']
        tolerance_elbow_start = TOLERANCES['start']['elbow']
        total_checks += 1

        # 生成反馈 (基于起始肘关节角度与标准 170 度的偏差)
        # 小于标准减容差 -> 手臂未完全伸直
        if user_elbow_start < std_elbow_start - tolerance_elbow_start:
            feedback.append("起始姿势肘关节未完全伸直")
        # 大于标准加容差 -> 肘关节超伸 (不常见，但可以作为反馈)
        # elif user_elbow_start > std_elbow_start + tolerance_elbow_start:
        #     feedback.append("起始姿势肘关节超伸") # 可以根据需要保留或移除此反馈

        # 计算得分 (使用基于偏差的方法)
        deviation_elbow_start = abs(user_elbow_start - std_elbow_start)
        score_elbow_start = calculate_score(deviation_elbow_start, tolerance_elbow_start, MAX_DEVIATION_MULTIPLIER)
        total_score += score_elbow_start

    # --- 检查 2: 起始躯干角度 (身体是否平直 - 撅屁股/塌腰) ---
    # 检查 user_data['start'] 中是否有 'torso_angle' 数据
    # 假设 detect_pushups 会在 stage_data['start'] 中存储 'torso_angle' 列表
    if 'torso_angle' in user_data.get('start', {}) and user_data['start']['torso_angle']:
        user_torso_start = np.mean(user_data['start']['torso_angle'])
        std_torso_start = STANDARD_PUSHUP_ANGLES['start']['torso'] # 90 度
        tolerance_torso_start = TOLERANCES['start']['torso']
        total_checks += 1

        # 生成反馈 (基于起始躯干角度与标准 90 度的偏差)
        # 小于标准减容差 -> 角度小于 90，躯干更向上倾斜 -> 撅屁股
        if user_torso_start < std_torso_start - tolerance_torso_start:
            feedback.append("起始姿势躯干角度过小，注意保持身体平直（撅屁股）")
        # 大于标准加容差 -> 角度大于 90，躯干更向下倾斜 -> 塌腰
        elif user_torso_start > std_torso_start + tolerance_torso_start:
            feedback.append("起始姿势躯干角度过大，注意保持身体平直（塌腰）")

        # 计算得分 (使用基于偏差的方法)
        deviation_torso_start = abs(user_torso_start - std_torso_start)
        score_torso_start = calculate_score(deviation_torso_start, tolerance_torso_start, MAX_DEVIATION_MULTIPLIER)
        total_score += score_torso_start

    # --- 检查 3: 结束肘关节角度 (俯卧撑深度) ---
    # 检查 user_data['end'] 中是否有 'elbow_angle' 数据
    # 假设 detect_pushups 会在 stage_data['end'] 中存储 'elbow_angle' 列表
    if 'elbow_angle' in user_data.get('end', {}) and user_data['end']['elbow_angle']:
        user_elbow_end = np.mean(user_data['end']['elbow_angle'])
        std_elbow_end = STANDARD_PUSHUP_ANGLES['end']['elbow'] # 90 度
        tolerance_elbow_end = TOLERANCES['end']['elbow']
        total_checks += 1

        # 生成反馈 (基于结束肘关节角度与标准 90 度的偏差)
        # 大于标准加容差 -> 角度大于 90，肘关节弯曲不够 -> 深度不足
        if user_elbow_end > std_elbow_end + tolerance_elbow_end:
            feedback.append("俯卧撑深度不足，肘关节弯曲不够")
        # 小于标准减容差 -> 角度小于 90，可能下降过低
        elif user_elbow_end < std_elbow_end - tolerance_elbow_end:
             feedback.append("俯卧撑下降过低，注意控制幅度") # 可以根据需要保留或移除此反馈

        # 计算得分 (使用基于偏差的方法)
        deviation_elbow_end = abs(user_elbow_end - std_elbow_end)
        score_elbow_end = calculate_score(deviation_elbow_end, tolerance_elbow_end, MAX_DEVIATION_MULTIPLIER)
        total_score += score_elbow_end

    # --- 4: 结束躯干角度 (最低点身体是否平直 - 撅屁股/塌腰) ---
    # 检查 user_data['end'] 中是否有 'torso_angle' 数据
    # 假设 detect_pushups 会在 stage_data['end'] 中存储 'torso_angle' 列表
    if 'torso_angle' in user_data.get('end', {}) and user_data['end']['torso_angle']:
        user_torso_end = np.mean(user_data['end']['torso_angle'])
        std_torso_end = STANDARD_PUSHUP_ANGLES['end']['torso'] # 90 度
        tolerance_torso_end = TOLERANCES['end']['torso']
        total_checks += 1

        # 生成反馈 (基于结束躯干角度与标准 90 度的偏差)
        # 小于标准减容差 -> 角度小于 90，躯干更向上倾斜 -> 撅屁股
        if user_torso_end < std_torso_end - tolerance_torso_end:
            feedback.append("俯卧撑最低点躯干角度过小，注意保持身体平直（撅屁股）")
        # 大于标准加容差 -> 角度大于 90，躯干更向下倾斜 -> 塌腰
        elif user_torso_end > std_torso_end + tolerance_torso_end:
            feedback.append("俯卧撑最低点躯干角度过大，注意保持身体平直（塌腰）")

        # 计算得分 (使用基于偏差的方法)
        deviation_torso_end = abs(user_torso_end - std_torso_end)
        score_torso_end = calculate_score(deviation_torso_end, tolerance_torso_end, MAX_DEVIATION_MULTIPLIER)
        total_score += score_torso_end

    # --- 计算最终准确率 ---
    if total_checks > 0:
        # 最终准确率 = 总得分 / 总检查数 (得到一个 0.0 到 1.0 之间的值)
        accuracy = total_score / total_checks
    else:
        # 如果没有进行任何检查 (例如，stage_data 没有对应键或列表为空)
        # 返回一个指示无法分析的值，或者默认一个低分，这里沿用原代码的 100，但0更合适
        # 例如可以返回 None 或 0.0
        accuracy = 0.0 # 修改为 0.0 更合理，表示无法评估

    final_feedback = feedback if feedback else ["动作标准，继续保持！"]  # 如果没有反馈，给出默认鼓励
    accuracy_percentage = round(accuracy * 100, 2) # 将 0-1 的得分转换为 0-100 的百分比

    # 返回反馈列表和计算出的准确率百分比
    return final_feedback, accuracy_percentage



# 引用 BasicPoseFun 中的函数 (需要确保 BasicPoseFun.py 存在且路径正确)
# from .BasicPoseFun import calculate_torso_angle_3d, calculate_angle_3d, draw_torso, draw_legs, draw_arms

def detect_pushups(video_path, output_video_path, display=False):

    # 动作状态机：
    # rest: 休息或准备状态 (手臂伸直)
    # descending: 下降过程 (肘部弯曲)
    # bottom: 最低点 (肘部弯曲到阈值以下)
    # ascending: 上升过程 (肘部伸直)
    # top: 最高点 (手臂伸直到阈值以上)
    action_state = 'top' # 初始状态假设为俯卧撑准备好的最高点姿势

    # 俯卧撑动作检测阈值 (基于肘关节角度)
    ELBOW_ANGLE_THRESHOLD_TOP = 135  # 视为手臂伸直的最小肘关节角度
    ELBOW_ANGLE_THRESHOLD_BOTTOM = 85 # 视为到达最低点的最大肘关节角度
    FRAME_COUNT_THRESHOLD = 5         # 连续帧数阈值，防止误触
    frame_count = 0
    pushup_count = 0  # 俯卧撑次数

    # 用于平滑角度的列表 (以肘关节为例)
    elbow_angles_buffer = []
    buffer_size = 5 # 缓冲帧数

    # 存储每次完整俯卧撑的关键阶段数据 (最高点和最低点)
    # 注意：这里的结构与 analyze_pushup_quality 函数期望的 user_data 结构需要匹配
    # user_data 期望的是 {'start': {'elbow_angle': [avg1, avg2...], ...}, 'end': {...}}
    # 我们需要调整数据收集方式来匹配
    pushup_data = {
        'start': {'elbow_angle': [], 'torso_angle': []}, # 存储每次俯卧撑最高点的角度
        'end': {'elbow_angle': [], 'torso_angle': []},   # 存储每次俯卧撑最低点的角度
    }

    # 临时存储当前俯卧撑最高点和最低点找到的最佳角度 (例如角度最稳定或最符合阈值的帧)
    current_pushup_peak_angles = {'elbow': None, 'torso': None} # 最高点时的角度
    current_pushup_bottom_angles = {'elbow': None, 'torso': None} # 最低点时的角度


    font_path = 'msyh.ttf'  # 请确保字体文件在指定路径中
    font = ImageFont.truetype(font_path, 24)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件。")
        return 0, {}, [], 0.0 # 返回默认值

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

            current_elbow_angle = None
            current_torso_angle = None
            pose_detected = False # 标记是否成功检测到姿态

            if results.pose_world_landmarks and results.pose_landmarks:
                pose_detected = True
                landmarks = results.pose_world_landmarks.landmark
                image_landmarks = results.pose_landmarks # 用于绘制

                # 绘制不同部位的连线
                draw_torso(image, image_landmarks)
                draw_arms(image, image_landmarks)
                draw_legs(image, image_landmarks)

                # 获取关键点 (使用世界坐标系的 landmarks)
                # 注意：俯卧撑姿势可能更侧重于身体侧面或斜侧面，确保关键点可信
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

                # 计算核心角度
                # 肘关节角度 (肩-肘-腕)
                # 使用左右两侧的平均值，或者选择检测置信度高的一侧
                if left_elbow.visibility > 0.5 and left_shoulder.visibility > 0.5 and left_wrist.visibility > 0.5:
                     left_elbow_angle = calculate_angle_3d(left_shoulder, left_elbow, left_wrist)
                else:
                     left_elbow_angle = None

                if right_elbow.visibility > 0.5 and right_shoulder.visibility > 0.5 and right_wrist.visibility > 0.5:
                     right_elbow_angle = calculate_angle_3d(right_shoulder, right_elbow, right_wrist)
                else:
                     right_elbow_angle = None

                # 尝试取有效一侧或两侧平均
                if left_elbow_angle is not None and right_elbow_angle is not None:
                    current_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
                elif left_elbow_angle is not None:
                    current_elbow_angle = left_elbow_angle
                elif right_elbow_angle is not None:
                    current_elbow_angle = right_elbow_angle

                # 躯干角度 (髋-肩，相对于垂直)
                # 使用左右髋和肩的平均位置来计算躯干向量，提高鲁棒性
                if (left_hip.visibility > 0.5 and left_shoulder.visibility > 0.5) or \
                   (right_hip.visibility > 0.5 and right_shoulder.visibility > 0.5):
                    # 计算平均髋部和肩部位置
                    avg_hip_x = (left_hip.x + right_hip.x) / 2 if left_hip.visibility > 0.5 and right_hip.visibility > 0.5 else (left_hip.x if left_hip.visibility > 0.5 else right_hip.x)
                    avg_hip_y = (left_hip.y + right_hip.y) / 2 if left_hip.visibility > 0.5 and right_hip.visibility > 0.5 else (left_hip.y if left_hip.visibility > 0.5 else right_hip.y)
                    avg_hip_z = (left_hip.z + right_hip.z) / 2 if left_hip.visibility > 0.5 and right_hip.visibility > 0.5 else (left_hip.z if left_hip.visibility > 0.5 else right_hip.z)

                    avg_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2 if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5 else (left_shoulder.x if left_shoulder.visibility > 0.5 else right_shoulder.x)
                    avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5 else (left_shoulder.y if left_shoulder.visibility > 0.5 else right_shoulder.y)
                    avg_shoulder_z = (left_shoulder.z + right_shoulder.z) / 2 if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5 else (left_shoulder.z if left_shoulder.visibility > 0.5 else right_shoulder.z)

                    # 创建临时的 Landmark 对象 (只需要 xyz 属性)
                    class TempLandmark:
                        def __init__(self, x, y, z):
                            self.x = x
                            self.y = y
                            self.z = z

                    avg_hip = TempLandmark(avg_hip_x, avg_hip_y, avg_hip_z)
                    avg_shoulder = TempLandmark(avg_shoulder_x, avg_shoulder_y, avg_shoulder_z)

                    current_torso_angle = calculate_torso_angle_3d(avg_hip, avg_shoulder)
                else:
                     current_torso_angle = None


                # 更新肘关节角度缓冲
                if current_elbow_angle is not None:
                    elbow_angles_buffer.append(current_elbow_angle)
                    if len(elbow_angles_buffer) > buffer_size:
                        elbow_angles_buffer.pop(0)

                # 计算平滑后的平均肘关节角度
                average_elbow_angle = sum(elbow_angles_buffer) / len(elbow_angles_buffer) if elbow_angles_buffer else None

                # 将图像转换为PIL格式以绘制中文
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)

                # 输出文本
                if average_elbow_angle is not None:
                     draw.text((50, 150), f'平均肘关节角度: {int(average_elbow_angle)}°', font=font, fill=(255, 255, 0))
                if current_torso_angle is not None:
                     draw.text((50, 180), f'躯干倾斜角度: {int(current_torso_angle)}°', font=font, fill=(255, 255, 0))

                # 其他角度，如果需要显示
                # draw.text((50, 210), f'左臂-躯干角度: {int(left_arm_angle)}°', font=font, fill=(255, 255, 255))
                # draw.text((50, 240), f'右臂-躯干角度: {int(right_arm_angle)}°', font=font, fill=(255, 255, 255))
                # draw.text((50, 270), f'左腿-躯干角度: {int(left_leg_angle)}°', font=font, fill=(255, 255, 255))
                # draw.text((50, 300), f'右腿-躯干角度: {int(right_leg_angle)}°', font=font, fill=(255, 255, 255))
                # draw.text((50, 330), f'左膝关节角度: {int(left_knee_angle)}°', font=font, fill=(255, 255, 255))
                # draw.text((50, 360), f'右膝关节角度: {int(right_knee_angle)}°', font=font, fill=(255, 255, 255))


                draw.text((50, 100), f'俯卧撑次数: {pushup_count}', font=font, fill=(0, 0, 255))
                draw.text((50, 50), f'状态: {action_state}', font=font, fill=(0, 255, 0))


                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                # --- 动作检测和数据收集逻辑 (俯卧撑状态机) ---
                # 只在成功检测到姿态且关键角度有效时进行状态判断
                if pose_detected and average_elbow_angle is not None and current_torso_angle is not None:

                    if action_state == 'top':
                        # 处于最高点状态，等待下降
                        if average_elbow_angle < ELBOW_ANGLE_THRESHOLD_BOTTOM: # 直接从最高点到了最低点附近（快速下落或检测不稳定）
                            frame_count += 1
                            if frame_count >= FRAME_COUNT_THRESHOLD:
                                action_state = 'bottom' # 跳过下降阶段，直接认为到了最低点
                                frame_count = 0
                                print(f"第 {pushup_count + 1} 次俯卧撑 (可能) 下降太快/不稳定，直接进入最低点状态")
                                # 记录最低点角度
                                current_pushup_bottom_angles['elbow'] = average_elbow_angle
                                current_pushup_bottom_angles['torso'] = current_torso_angle
                        elif average_elbow_angle < ELBOW_ANGLE_THRESHOLD_TOP: # 肘关节开始弯曲，进入下降状态
                             frame_count += 1
                             if frame_count >= FRAME_COUNT_THRESHOLD:
                                action_state = 'descending'
                                frame_count = 0
                                print(f"第 {pushup_count + 1} 次俯卧撑开始下降")
                                # 记录最高点角度
                                current_pushup_peak_angles['elbow'] = average_elbow_angle # 此时是刚离开最高点或在最高点稳定状态
                                current_pushup_peak_angles['torso'] = current_torso_angle

                        else: # 保持在最高点
                            frame_count = 0
                            # 可以在保持最高点时更新记录的最佳最高点角度
                            if current_pushup_peak_angles['elbow'] is None or average_elbow_angle > current_pushup_peak_angles['elbow']:
                                 current_pushup_peak_angles['elbow'] = average_elbow_angle
                                 current_pushup_peak_angles['torso'] = current_torso_angle


                    elif action_state == 'descending':
                        # 处于下降状态，等待到达最低点
                        if average_elbow_angle < ELBOW_ANGLE_THRESHOLD_BOTTOM: # 到达或低于最低点阈值
                            frame_count += 1
                            if frame_count >= FRAME_COUNT_THRESHOLD:
                                action_state = 'bottom'
                                frame_count = 0
                                print(f"第 {pushup_count + 1} 次俯卧撑到达最低点")
                                # 记录最低点角度
                                current_pushup_bottom_angles['elbow'] = average_elbow_angle # 此时是刚进入最低点或在最低点状态
                                current_pushup_bottom_angles['torso'] = current_torso_angle
                        elif average_elbow_angle > ELBOW_ANGLE_THRESHOLD_TOP: # 下降过程中又伸直了 (可能是中断或错误)
                             action_state = 'top' # 回到最高点状态
                             frame_count = 0
                             print(f"第 {pushup_count + 1} 次俯卧撑下降中断，返回最高点")

                    elif action_state == 'bottom':
                         # 处于最低点状态，等待上升
                         if average_elbow_angle > ELBOW_ANGLE_THRESHOLD_BOTTOM: # 肘关节角度开始变大，开始上升
                              frame_count += 1
                              if frame_count >= FRAME_COUNT_THRESHOLD:
                                   action_state = 'ascending'
                                   frame_count = 0
                                   print(f"第 {pushup_count + 1} 次俯卧撑开始上升")
                                   # 再次确认并记录最低点角度 (取在该状态下找到的最小肘关节角度对应的帧)
                                   # 这里可以优化，例如找bottom状态下最小elbow角度的帧作为最低点
                                   # 简单起见，这里使用进入bottom状态时的角度
                                   if current_pushup_bottom_angles['elbow'] is None or average_elbow_angle < current_pushup_bottom_angles['elbow']:
                                       current_pushup_bottom_angles['elbow'] = average_elbow_angle
                                       current_pushup_bottom_angles['torso'] = current_torso_angle


                    elif action_state == 'ascending':
                        # 处于上升状态，等待回到最高点
                        if average_elbow_angle > ELBOW_ANGLE_THRESHOLD_TOP: # 到达或超过最高点阈值
                            frame_count += 1
                            if frame_count >= FRAME_COUNT_THRESHOLD:
                                action_state = 'top' # 完成一次俯卧撑
                                frame_count = 0
                                pushup_count += 1
                                print(f"完成第 {pushup_count} 次俯卧撑")

                                # --- 记录当前完成的俯卧撑数据 ---
                                # 将本次俯卧撑的最高点和最低点角度添加到 pushup_data 中
                                if current_pushup_peak_angles['elbow'] is not None:
                                     pushup_data['start']['elbow_angle'].append(current_pushup_peak_angles['elbow'])
                                     pushup_data['start']['torso_angle'].append(current_pushup_peak_angles['torso']) # 记录最高点时的躯干角度

                                if current_pushup_bottom_angles['elbow'] is not None:
                                     pushup_data['end']['elbow_angle'].append(current_pushup_bottom_angles['elbow'])
                                     pushup_data['end']['torso_angle'].append(current_pushup_bottom_angles['torso']) # 记录最低点时的躯干角度

                                # 重置本次俯卧撑的临时角度记录
                                current_pushup_peak_angles = {'elbow': None, 'torso': None}
                                current_pushup_bottom_angles = {'elbow': None, 'torso': None}

                        elif average_elbow_angle < ELBOW_ANGLE_THRESHOLD_BOTTOM: # 上升过程中又掉下去了 (可能是力竭或姿势不稳定)
                             action_state = 'bottom' # 回到最低点状态
                             frame_count = 0
                             print(f"第 {pushup_count + 1} 次俯卧撑上升中断，返回最低点")

                else:
                    # 如果姿态未检测到或关键角度无效，重置帧数计数，保持当前状态
                    frame_count = 0
                    # 可以在这里根据需要决定是否将状态强制重置为 'top' 或 'rest'
                    # 例如：如果连续很多帧都没有检测到有效姿态，可能用户离开了画面
                    # 暂时保持原状态，直到检测恢复

            else:
                # 未检测到关键点
                print(f"在第 {frame_number} 帧未检测到关键点。")
                # 在未检测到关键点时，保持当前状态，frame_count重置或递增取决于需求
                # 重置 frame_count 可以防止误触状态切换
                frame_count = 0
                # 可以考虑在长时间未检测到姿态时，将 state 强制设置为 'rest'/'top'

            # 将帧写入输出视频文件
            out.write(image)


        print(f"总共完成了 {pushup_count} 次俯卧撑。")
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # 在视频处理完成后，对收集到的数据进行分析
        # 注意：这里将 pushup_data 传递给 analyze_pushup_quality
        # pushup_data 的结构 {'start': {'elbow_angle': [...], 'torso_angle': [...]}, 'end': {'elbow_angle': [...], 'torso_angle': [...]}}
        # 这个结构与 analyze_pushup_quality 期望的 user_data 结构是一致的
        feedback, analysis_result = analyze_pushup_quality(pushup_data)
        print("\n动作分析报告:")
        # 打印反馈和结果
        print("反馈:", feedback)
        print("准确率:", analysis_result)

        # 返回结果
        return pushup_count, pushup_data, feedback, analysis_result





def main(video_name):
    # 下载视频
    subprocess.call(['you-get', 'https://www.lloydyide.cn/fitbotAPI/uploads/'+video_name])
    video_path = video_name #video_name
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps  # 计算视频时长（秒）
    cap.release()
    output_video_path = "output" + video_name # 保存输出的视频文件路径
    total_squats, action_data, feedback, analysis_result = detect_pushups(video_path, output_video_path)


    return total_squats,action_data, feedback, analysis_result, video_duration


# 确保 import requests 和 import os

def shipin(request):
    email = request.GET.get('email') if hasattr(request, 'GET') else request.args.get('email')

    if not email:
        return HttpResponse("Error: Email parameter missing.", status=400)

    try:
        # --- Step 1: 通过外部 API 获取实际的视频文件名 ---
        print(f"正在请求视频名称，email: {email}")
        video_name_response = requests.get('https://www.lloydyide.cn/fitbotAPI/downLoad.php', params={"email": email})

        # 检查 API 响应是否成功
        if video_name_response.status_code != 200:
             print(f"获取视频名称失败: {video_name_response.status_code}, {video_name_response.text}")
             return HttpResponse(f"Error: Could not retrieve video name from API. Status: {video_name_response.status_code}", status=video_name_response.status_code)

        # 获取响应文本，假设这就是视频文件名
        video_name_from_api = video_name_response.text.strip() # 获取视频名并去除首尾空白符

        if not video_name_from_api:
             print("从API获取的视频名称为空。")
             return HttpResponse("Error: Retrieved empty video name from API.", status=500)

        print(f"从API获取到视频名称: {video_name_from_api}")

        # --- Step 2: 调用 main 函数，并将获取到的视频文件名传递过去 ---
        # main 函数内部将使用这个文件名进行下载和处理
        pushup_count, pushup_data, feedback, analysis_result, video_duration = main(video_name_from_api)

        # 将分析结果上传到外部 API
        upload_data = {
            'email': email,
            'accuracy': analysis_result*0.01, # 使用分析结果百分比 (analyze_pushup_quality 返回 0-100)
            'time': round(video_duration, 2), # 时长保留两位小数
            'month': datetime.date.today().month # 当前月份
        }
        print(f"准备上传数据: {upload_data}")

        # todo 完成质量的上传
        try:
             upload_response = requests.post('https://www.lloydyide.cn/fitbotAPI/update_umi.php', data=upload_data)
             print(f"上传结果: {upload_response.status_code}, {upload_response.text}")
        except Exception as e:
             print(f"上传数据到API失败: {e}")
             # 可以选择在这里添加到反馈中，或者记录日志


    except Exception as e:
        # 如果在获取视频名或调用 main 函数过程中发生任何未捕获的错误
        print(f"俯卧撑分析流程中发生未预期的错误: {e}")
        return HttpResponse(f"An unexpected error occurred during analysis: {e}", status=500)


    # 构建完整的HTML响应 (这部分代码与你提供的最新版本一致，无需修改)
    html_content = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 600px;
                margin: 0 auto;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 12px;">

        <!-- 反馈建议部分 -->
        <div style="margin-bottom: 20px;">
            <h3 style="color: #333; margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 8px;">
                <span style="display: inline-block; width: 24px; height: 24px; background: #07C160;
                            color: white; text-align: center; border-radius: 50%; margin-right: 8px;">✓</span>
                俯卧撑动作分析报告
            </h3>
            <div>
                <div style="background: #f8f9fa;border-radius: 10px;flex: 1;box-shadow: 0 2px 5px rgba(0,0,0,0.1);padding:10px; margin-bottom: 10px;">
                    <span style="display: block;color: #6c757d;font-size: 14px;">动作准确率</span>
                    <span style="display: block;font-size: 16px;font-weight: bold;color: #343a40;margin-top: 5px;">{analysis_result:.2f}%</span>
                </div>
            <div style="background: #f8f9fa;border-radius: 10px;flex: 1;box-shadow: 0 2px 5px rgba(0,0,0,0.1);padding:10px; margin-bottom: 10px;">
                    <span style="display: block;color: #6c757d;font-size: 14px;">完成次数</span>
                    <span style="display: block;font-size: 16px;font-weight: bold;color: #343a40;margin-top: 5px;">{pushup_count} 次</span>
                </div>
                <div style="background: #f8f9fa;border-radius: 10px;flex: 1;box-shadow: 0 2px 5px rgba(0,0,0,0.1);padding:10px;">
                    <span style="display: block;color: #6c72d;font-size: 14px;">视频时长</span>
                    <span style="display: block;font-size: 16px;font-weight: bold;color: #343a40;margin-top: 5px;">{video_duration:.2f} 秒</span>
                </div>
            </div>

            {"".join(
                f'<div style="padding:12px; margin:8px 0; background:#fff; border-radius:8px;'
                f'box-shadow:0 1px 3px rgba(0,0,0,0.1); border-left:4px solid '
                f'{"#07C160" if "标准" in item or "继续保持" in item else "#DD524D"};'
                f'">{item}</div>'
                for item in feedback
            )}
        </div>

        <!-- 俯卧撑数据详情 -->
        <div style="background: white; border-radius: 10px; padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h3 style="color: #333; margin-top: 0; margin-bottom: 15px;">
                <span style="display: inline-block; width: 24px; height: 24px; background: #007AFF;
                            color: white; text-align: center; border-radius: 50%; margin-right: 8px;">🔍</span>
                详细运动数据 (最高点 / 最低点)
            </h3>
    """

    # 添加每次俯卧撑的数据详情
    # 遍历 pushup_data['start']['elbow_angle'] 的长度，因为它记录了每次俯卧撑的最高点数据
    # 假设 pushup_data['start'] 和 pushup_data['end'] 中的列表长度一致
    num_pushups_recorded = len(pushup_data.get('start', {}).get('elbow_angle', []))

    if num_pushups_recorded == 0 and pushup_count > 0:
         html_content += "<p>未能记录每次俯卧撑的详细角度数据。</p>"
    elif num_pushups_recorded > 0:
        for i in range(num_pushups_recorded):
            # 安全地获取数据，防止索引越界。如果索引超出范围，设置为 None 或适当的默认值
            start_elbow = pushup_data['start']['elbow_angle'][i] if i < len(pushup_data.get('start', {}).get('elbow_angle', [])) else None
            start_torso = pushup_data['start']['torso_angle'][i] if i < len(pushup_data.get('start', {}).get('torso_angle', [])) else None
            end_elbow = pushup_data['end']['elbow_angle'][i] if i < len(pushup_data.get('end', {}).get('elbow_angle', [])) else None
            end_torso = pushup_data['end']['torso_angle'][i] if i < len(pushup_data.get('end', {}).get('torso_angle', [])) else None

            # 在HTML中显示时，处理可能的 None 值
            start_elbow_str = f"{start_elbow:.1f}°" if start_elbow is not None else "N/A"
            start_torso_str = f"{start_torso:.1f}°" if start_torso is not None else "N/A"
            end_elbow_str = f"{end_elbow:.1f}°" if end_elbow is not None else "N/A"
            end_torso_str = f"{end_torso:.1f}°" if end_torso is not None else "N/A"


            html_content += f"""
            <div style="margin-bottom: 15px; padding: 12px; background: #f8f9fa; border-radius: 8px;">
                <div style="font-weight: 600; color: #007AFF; margin-bottom: 8px;">第 {i + 1} 次俯卧撑</div>
                <div style="display: flex; justify-content: space-between;">
                    <div style="flex: 1; padding: 8px; background: rgba(0,122,255,0.05); border-radius: 6px;">
                        <div style="font-weight: 500; margin-bottom: 4px; color: #333;">最高点</div>
                        <div style="margin-top: 4px;">
                            <span style="font-size: 12px; color: #888;">肘关节:</span>
                            <span style="font-weight: 500;">{start_elbow_str}</span>
                        </div>
                        <div>
                            <span style="font-size: 12px; color: #888;">躯干:</span>
                            <span style="font-weight: 500;">{start_torso_str}</span>
                        </div>
                    </div>
                    <div style="flex: 1; padding: 8px; background: rgba(0,122,255,0.05); border-radius: 6px; margin-left: 10px;">
                         <div style="font-weight: 500; margin-bottom: 4px; color: #333;">最低点</div>
                        <div style="margin-top: 4px;">
                            <span style="font-size: 12px; color: #888;">肘关节:</span>
                            <span style="font-weight: 500;">{end_elbow_str}</span>
                        </div>
                        <div>
                            <span style="font-size: 12px; color: #888;">躯干:</span>
                            <span style="font-weight: 500;">{end_torso_str}</span>
                        </div>
                    </div>
                </div>
            </div>
            """
    else:
        html_content += "<p>未能检测到完整的俯卧撑动作。</p>"

    html_content += """
        </div>
    </div>
    """

    return HttpResponse(html_content)


