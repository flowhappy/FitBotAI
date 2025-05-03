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
STANDARD_PULLUP_ANGLES = {
    'start': {  # 起始悬挂姿势
        'elbow': 150,
        'arm_torso': 150
    },
    'end': {    # 拉到最高点的姿势
        'elbow': 90,
        'arm_torso': 60
    }
}

PULLUP_TOLERANCES = {
    'start': {  # 起始悬挂姿势
        'elbow': 10,     # 肘关节角度容差 (允许 +/- 10°) - 手臂接近伸直
        'arm_torso': 10  # 手臂-躯干角度容差 (允许 +/- 10°) - 手臂充分举过头
    },
    'end': {    # 结束顶点姿势
        'elbow': 25,     # 肘关节角度容差 (允许 +/- 25°) - 考虑到顶点高度和发力差异
        'arm_torso': 20  # 手臂-躯干角度容差 (允许 +/- 20°) - 顶点时手臂下拉程度差异
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


def analyze_pullup_quality(user_data):
    """智能分析引体向上质量"""
    feedback = []
    total_score = 0.0  # 用于累加各项得分
    total_checks = 0  # 用于记录总共执行的检查项目数

    # --- 检查 1: 起始肘关节角度 (手臂伸直度) ---
    # 假设 user_data 结构类似，但键名已改为引体向上的角度
    # 例如 user_data['start']['elbow_angle']
    if 'elbow_angle' in user_data['start'] and user_data['start']['elbow_angle']:
        # 假设 user_data 存储的是每次动作的平均值列表，或者需要先计算平均值
        # 如果每次动作只有一个值，直接使用即可
        # 为保持与原代码一致，我们先假设需要计算平均值 (如果不是，需调整)
        user_elbow_start = np.mean(user_data['start']['elbow_angle'])
        std_elbow_start = STANDARD_PULLUP_ANGLES['start']['elbow']
        tolerance_elbow_start = PULLUP_TOLERANCES['start']['elbow']
        total_checks += 1

        # 生成反馈 (使用严格的容差)
        # 起始时，角度 *小于* 标准值意味着手臂弯曲过多
        if user_elbow_start < std_elbow_start - tolerance_elbow_start:
            feedback.append("起始悬挂时手臂不够伸直")
        # 理论上大于180不太可能，但保留以防数据异常
        elif user_elbow_start > std_elbow_start + tolerance_elbow_start:
            feedback.append("起始悬挂时手臂过度伸展 (检查数据是否准确)")

        # 计算得分 (使用基于偏差的方法)
        deviation_elbow_start = abs(user_elbow_start - std_elbow_start)
        score_elbow_start = calculate_score(deviation_elbow_start, tolerance_elbow_start, MAX_DEVIATION_MULTIPLIER)
        total_score += score_elbow_start
    elif 'elbow_angle' not in user_data['start']:
         print("警告: user_data['start'] 中缺少 'elbow_angle'")


    # --- 检查 2: 起始手臂-躯干角度 (手臂过头程度) ---
    if 'arm_torso_angle' in user_data['start'] and user_data['start']['arm_torso_angle']:
        user_arm_torso_start = np.mean(user_data['start']['arm_torso_angle'])
        std_arm_torso_start = STANDARD_PULLUP_ANGLES['start']['arm_torso']
        tolerance_arm_torso_start = PULLUP_TOLERANCES['start']['arm_torso']
        total_checks += 1

        # 生成反馈
        # 起始时，角度 *小于* 标准值意味着手臂没有完全举过头顶
        if user_arm_torso_start < std_arm_torso_start - tolerance_arm_torso_start:
            feedback.append("起始悬挂时肩关节未完全打开 (手臂未充分举过头)")
        elif user_arm_torso_start > std_arm_torso_start + tolerance_arm_torso_start:
             feedback.append("起始悬挂时手臂过度后伸 (检查数据是否准确)")

        # 计算得分
        deviation_arm_torso_start = abs(user_arm_torso_start - std_arm_torso_start)
        score_arm_torso_start = calculate_score(deviation_arm_torso_start, tolerance_arm_torso_start, MAX_DEVIATION_MULTIPLIER)
        total_score += score_arm_torso_start
    elif 'arm_torso_angle' not in user_data['start']:
        print("警告: user_data['start'] 中缺少 'arm_torso_angle'")


    # --- 检查 3: 结束肘关节角度 (拉起高度/手臂弯曲度) ---
    if 'elbow_angle' in user_data['end'] and user_data['end']['elbow_angle']:
        user_elbow_end = np.mean(user_data['end']['elbow_angle'])
        std_elbow_end = STANDARD_PULLUP_ANGLES['end']['elbow']
        tolerance_elbow_end = PULLUP_TOLERANCES['end']['elbow']
        total_checks += 1

        # 生成反馈
        # 结束时，角度 *大于* 标准值意味着手臂弯曲不够，即拉起高度不足
        if user_elbow_end > std_elbow_end + tolerance_elbow_end:
            feedback.append("引体向上拉起高度不足，手臂应更弯曲")
        # 结束时，角度 *小于* 标准值意味着手臂弯曲超过了标准，可能是拉得很高或数据有误
        elif user_elbow_end < std_elbow_end - tolerance_elbow_end:
            # 这个反馈可以根据需要调整，过度弯曲本身不一定是坏事
            feedback.append("引体向上手臂弯曲超过标准范围 (可能拉得非常高，或检查是否为目标)")

        # 计算得分
        deviation_elbow_end = abs(user_elbow_end - std_elbow_end)
        score_elbow_end = calculate_score(deviation_elbow_end, tolerance_elbow_end, MAX_DEVIATION_MULTIPLIER)
        total_score += score_elbow_end
    elif 'elbow_angle' not in user_data['end']:
        print("警告: user_data['end'] 中缺少 'elbow_angle'")


    # --- 检查 4: 结束手臂-躯干角度 (上臂下拉程度) ---
    if 'arm_torso_angle' in user_data['end'] and user_data['end']['arm_torso_angle']:
        user_arm_torso_end = np.mean(user_data['end']['arm_torso_angle'])
        std_arm_torso_end = STANDARD_PULLUP_ANGLES['end']['arm_torso']
        tolerance_arm_torso_end = PULLUP_TOLERANCES['end']['arm_torso']
        total_checks += 1

        # 生成反馈
        # 结束时，角度 *大于* 标准值意味着手臂没有充分下拉靠近身体
        if user_arm_torso_end > std_arm_torso_end + tolerance_arm_torso_end:
            feedback.append("引体向上顶点时，上臂下拉幅度不足 (尝试将手肘更靠近身体)")
        # 结束时，角度 *小于* 标准值可能意味着过度下拉或数据偏差
        elif user_arm_torso_end < std_arm_torso_end - tolerance_arm_torso_end:
            feedback.append("引体向上顶点时，上臂下拉超过标准范围 (检查动作形态)")

        # 计算得分
        deviation_arm_torso_end = abs(user_arm_torso_end - std_arm_torso_end)
        score_arm_torso_end = calculate_score(deviation_arm_torso_end, tolerance_arm_torso_end, MAX_DEVIATION_MULTIPLIER)
        total_score += score_arm_torso_end
    elif 'arm_torso_angle' not in user_data['end']:
        print("警告: user_data['end'] 中缺少 'arm_torso_angle'")


    # --- 计算最终准确率 ---
    if total_checks > 0:
        # 最终准确率 = 总得分 / 总检查数 (得到一个 0.0 到 1.0 之间的值)
        # 将准确率转换为百分比或其他形式可以在调用处处理
        accuracy = total_score / total_checks
    else:
        accuracy = 0.0 # 如果没有进行任何检查，准确率设为 0 (避免除零错误)
        feedback.append("未能获取足够的角度数据进行分析")


    final_feedback = feedback if feedback else ["动作标准，继续保持！"] # 保留默认反馈逻辑
    # accuracy = round(accuracy * 100) # 如果需要返回百分比整数
    accuracy = round(accuracy, 2) # 返回 0.0 到 1.0 之间的小数，保留两位

    # 返回反馈列表和计算出的准确率
    return final_feedback, accuracy


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def detect_pullups(video_path, output_video_path, display=False):
    """
    检测视频中的引体向上动作，记录关键角度，并调用分析函数。

    Args:
        video_path (str): 输入视频文件的路径。
        output_video_path (str): 处理后输出视频文件的路径。
        display (bool): 是否实时显示处理中的视频帧。

    Returns:
        tuple: 包含以下元素的元组:
            - pullup_count (int): 检测到的有效引体向上次数。
            - analysis_input_data (dict): 包含每次动作起止角度数据的字典，
                                          结构为 {'start': {'elbow_angle': [...], 'arm_torso_angle': [...]},
                                                 'end': {'elbow_angle': [...], 'arm_torso_angle': [...]}}。
            - feedback (list): 来自分析函数的反馈信息列表。
            - analysis_result (float): 来自分析函数的综合得分 (0.0 到 1.0)。
    """

    action_state = 'rest' # States: 'rest', 'pulling_up', 'lowering'

    # --- Thresholds for State Machine ---
    ELBOW_ANGLE_THRESHOLD_LOW = 100   # Angle considered "up" (e.g., less than 92 degrees)
    ELBOW_ANGLE_THRESHOLD_HIGH = 150  # Angle considered "down/hang" (e.g., greater than 150 degrees)
    CONSECUTIVE_FRAMES_THRESHOLD = 3 # Number of consecutive frames to confirm state change

    frame_count_down = 0 # Counter for detecting downward stable state (hang) or starting descent
    frame_count_up = 0   # Counter for detecting upward stable state (peak) or starting ascent

    pullup_count = 0  # 引体向上次数
    elbow_angles = [] # Store recent elbow angles for smoothing/averaging
    current_pullup_data = {} # Store data for the current pull-up rep (start/end angles)

    # Store data for each rep
    all_reps_data = {'start': [], 'end': [], 'start_arm_torso': [], 'end_arm_torso': []}

    # Font for drawing
    try:
        # 确保系统中有支持中文的字体文件，例如微软雅黑 'msyh.ttf' 或 'simhei.ttf'
        # 如果在 Linux 或 Mac 上，可能需要指定不同的路径或字体名称
        font_path = 'msyh.ttf' # Windows 常见路径
        # font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc' # Example for Linux with Noto
        font = ImageFont.truetype(font_path, 24)
    except IOError:
        print(f"警告: 无法加载字体 {font_path}. 将使用默认字体。")
        font = ImageFont.load_default() # Fallback

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        # Return default values indicating failure
        return 0, {'start': {'elbow_angle': [], 'arm_torso_angle': []}, 'end': {'elbow_angle': [], 'arm_torso_angle': []}}, ["无法打开视频文件"], 0.0

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 或者 'XVID', 'H264' 等，取决于你的系统支持
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1, # 可以根据需要调整 (0, 1, 2)
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        frame_number = 0
        min_elbow_angle_this_rep = 180.0 # Track minimum angle during ascent for the current rep
        peak_arm_torso_angle = 0.0     # Track arm-torso angle at the peak of the current rep

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("视频处理完毕或读取帧失败。")
                break

            frame_number += 1
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            landmarks_detected = results.pose_world_landmarks and results.pose_landmarks

            if landmarks_detected:
                landmarks = results.pose_world_landmarks.landmark
                image_landmarks = results.pose_landmarks.landmark # Use image landmarks for drawing

                # --- Draw Skeleton ---
                # 使用 MediaPipe 的绘图工具绘制基本骨架
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
                # 或者使用你的自定义绘图函数（如果它们使用 image_landmarks）
                # draw_torso(image, results.pose_landmarks)
                # draw_arms(image, results.pose_landmarks)
                # draw_legs(image, results.pose_landmarks)

                # --- Get Keypoints (World landmarks for 3D angles) ---
                try:
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                except IndexError:
                    print(f"帧 {frame_number}: 检测到的关键点不足，跳过此帧的角度计算。")
                    landmarks_detected = False # Treat as if no landmarks for angle calculation

            if landmarks_detected:
                 # --- Calculate Pull-up Angles ---
                left_elbow_angle = calculate_angle_3d(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle_3d(right_shoulder, right_elbow, right_wrist)
                avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2

                left_arm_torso_angle = calculate_angle_3d(left_elbow, left_shoulder, left_hip)
                right_arm_torso_angle = calculate_angle_3d(right_elbow, right_shoulder, right_hip)
                avg_arm_torso_angle = (left_arm_torso_angle + right_arm_torso_angle) / 2

                # --- Smoothing elbow angle ---
                elbow_angles.append(avg_elbow_angle)
                if len(elbow_angles) > 5: # Keep a moving average window of 5 frames
                    elbow_angles.pop(0)
                smoothed_elbow_angle = sum(elbow_angles) / len(elbow_angles)

                print(f"帧 {frame_number}: "
                      f"状态={action_state}, "
                      f"原始肘角={avg_elbow_angle:.1f}, "
                      f"平滑肘角={smoothed_elbow_angle:.1f}, "
                      f"臂躯角={avg_arm_torso_angle:.1f}, "
                      f"本轮最小肘角={min_elbow_angle_this_rep:.1f}, "
                      f"计数(下)={frame_count_down}, "
                      f"计数(上)={frame_count_up}")

                # --- State Machine Logic for Pull-ups ---

                # Current state: REST (hanging, waiting to pull)
                if action_state == 'rest':
                    # Check if starting to pull up (angle decreases significantly)
                    if smoothed_elbow_angle < ELBOW_ANGLE_THRESHOLD_HIGH - 10: # More robust start trigger
                        frame_count_up += 1
                        frame_count_down = 0 # Reset other counter
                        if frame_count_up >= CONSECUTIVE_FRAMES_THRESHOLD:
                            action_state = 'pulling_up'
                            print(f"***** 第 {pullup_count + 1} 次引体向上开始 (帧 {frame_number}) *****")
                            # Record START data for this rep
                            current_pullup_data['start'] = {
                                'elbow_angle': avg_elbow_angle, # Use current angle at start point
                                'arm_torso_angle': avg_arm_torso_angle
                            }
                            min_elbow_angle_this_rep = avg_elbow_angle # Initialize min angle tracker for this rep
                            peak_arm_torso_angle = avg_arm_torso_angle # Initialize peak angle tracker for this rep
                            frame_count_up = 0 # Reset counter for next state

                    else:
                        # If angle stays high or increases, reset the up counter
                        frame_count_up = 0

                # Current state: PULLING_UP
                elif action_state == 'pulling_up':
                    # Track the minimum elbow angle and corresponding arm-torso angle reached during this phase
                    if avg_elbow_angle < min_elbow_angle_this_rep:
                        min_elbow_angle_this_rep = avg_elbow_angle
                        peak_arm_torso_angle = avg_arm_torso_angle # Store arm-torso angle at the elbow minimum

                    # Check if user has reached the peak (lowest elbow angle <= threshold)
                    # AND starts lowering (angle increases consistently)
                    # We use frame_count_down to confirm the upward trend (lowering)
                    if min_elbow_angle_this_rep <= ELBOW_ANGLE_THRESHOLD_LOW and smoothed_elbow_angle > min_elbow_angle_this_rep + 5: # Start moving up from the bottom
                        frame_count_down += 1
                        frame_count_up = 0 # Reset other counter
                        if frame_count_down >= CONSECUTIVE_FRAMES_THRESHOLD:
                            action_state = 'lowering'
                            print(f"***** 第 {pullup_count + 1} 次引体向上达到顶点/开始下降 (帧 {frame_number}) *****")
                            # Record END data (peak position) for this rep
                            current_pullup_data['end'] = {
                                'elbow_angle': min_elbow_angle_this_rep, # Record the minimum elbow angle as peak
                                'arm_torso_angle': peak_arm_torso_angle # Record arm-torso angle at that peak
                            }

                            # Store the completed rep's data and increment count
                            if 'start' in current_pullup_data and 'end' in current_pullup_data:
                                all_reps_data['start'].append(current_pullup_data['start']['elbow_angle'])
                                all_reps_data['end'].append(current_pullup_data['end']['elbow_angle'])
                                # Also store arm-torso angles, checking if they exist
                                if 'arm_torso_angle' in current_pullup_data['start']:
                                    all_reps_data['start_arm_torso'].append(current_pullup_data['start']['arm_torso_angle'])
                                if 'arm_torso_angle' in current_pullup_data['end']:
                                    all_reps_data['end_arm_torso'].append(current_pullup_data['end']['arm_torso_angle'])

                                pullup_count += 1 # Increment count ONLY after a valid peak is detected and descent starts
                            else:
                                print(f"警告: 第 {pullup_count + 1} 次动作数据不完整（缺少起始或结束点），未计数。")

                            current_pullup_data = {} # Clear data for the next potential rep
                            min_elbow_angle_this_rep = 180.0 # Reset min angle tracker for the next rep
                            frame_count_down = 0 # Reset counter for next state

                    elif smoothed_elbow_angle <= min_elbow_angle_this_rep + 5 : # Still at the bottom or going lower
                         frame_count_down = 0 # Reset descent counter if angle stops increasing or decreases


                # Current state: LOWERING
                elif action_state == 'lowering':
                    # Check if user has returned to the hanging position (angle high again consistently)
                    # We use frame_count_up to confirm the stable high angle (rest)
                    if smoothed_elbow_angle > ELBOW_ANGLE_THRESHOLD_HIGH:
                        frame_count_up += 1
                        frame_count_down = 0 # Reset other counter
                        if frame_count_up >= CONSECUTIVE_FRAMES_THRESHOLD:
                            action_state = 'rest'
                            print(f"***** 第 {pullup_count} 次引体向上完成下降/回到悬挂 (帧 {frame_number}) *****")
                            frame_count_up = 0 # Reset counter for next state
                    else:
                        # If angle drops significantly again before reaching rest, reset the rest counter
                        if smoothed_elbow_angle < ELBOW_ANGLE_THRESHOLD_HIGH - 10:
                            frame_count_up = 0
                        # Keep frame_count_down at 0 while lowering


            # --- Draw information on frame ---
            # Use PIL to draw text with the loaded font
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            # Display relevant angles and count if landmarks were detected
            if landmarks_detected:
                draw.text((50, 150), f'平滑肘关节角度: {int(smoothed_elbow_angle)}°', font=font, fill=(255, 255, 0)) # Yellow
                draw.text((50, 180), f'平均臂-躯干角度: {int(avg_arm_torso_angle)}°', font=font, fill=(0, 255, 255)) # Cyan
            else:
                # Handle frame with no detected landmarks
                draw.text((50, 50), "未检测到人体", font=font, fill=(255, 0, 0)) # Red

            # Always display state and count
            draw.text((50, 100), f'有效引体向上次数: {pullup_count}', font=font, fill=(0, 0, 255)) # Blue
            draw.text((50, 50), f'状态: {action_state}', font=font, fill=(0, 255, 0)) # Green

            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


            # Write frame to output video
            out.write(image)

            # Optional: Display frame if needed
            if display:
                cv2.imshow('MediaPipe Pose - Pull-up Detection', image)
                # Adjust waitKey for desired playback speed (e.g., 1ms for max speed, higher for slower)
                if cv2.waitKey(1) & 0xFF == 27: # Press ESC to exit
                    break

        # --- Cleanup ---
        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()

        print(f"\n总共完成了 {pullup_count} 次有效引体向上。")

        # --- Prepare data for analysis ---
        # Structure the data as expected by analyze_pullup_quality
        analysis_input_data = {
            'start': {
                'elbow_angle': all_reps_data.get('start', []),
                'arm_torso_angle': all_reps_data.get('start_arm_torso', [])
            },
            'end': { # 'end' here corresponds to the peak contraction point
                'elbow_angle': all_reps_data.get('end', []),
                'arm_torso_angle': all_reps_data.get('end_arm_torso', [])
            }
        }

        # --- Call the analysis function (ensure it's defined/imported) ---
        try:
            # Make sure analyze_pullup_quality is defined and imported correctly
            # This function should return feedback (list) and analysis_result (float)
            feedback, analysis_result = analyze_pullup_quality(analysis_input_data)
            print("\n动作分析报告:")
            if not feedback: feedback = ["无具体反馈。"] # Default feedback if empty
            for msg in feedback:
                print(f"- {msg}")
            print(f"综合准确率: {analysis_result:.2f}")
        except NameError:
            print("错误: analyze_pullup_quality 函数未定义或不可用。无法进行质量分析。")
            feedback = ["分析函数不可用"]
            analysis_result = 0.0
        except Exception as e:
             print(f"分析过程中发生错误: {e}")
             feedback = [f"分析时出错: {e}"]
             analysis_result = 0.0


    # Return count, structured data, feedback, and score
    return pullup_count, analysis_input_data, feedback, analysis_result

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
    total_squats, action_data, feedback, analysis_result = detect_pullups(video_path, output_video_path)


    return total_squats,action_data, feedback, analysis_result, video_duration


def shipin(request):
    email = request.GET.get('email')
    # 从 API 获取视频文件名
    video_name_response = requests.get('https://www.lloydyide.cn/fitbotAPI/downLoad.php', params={"email": email})
    video_name = video_name_response.text # 假设 API 返回的是纯文件名，如 "user_video.mp4"

    # 调用你的 main 函数，它处理下载、时长计算、调用 detect_pullups
    # main 函数返回 5 个值
    count, data, feedback, analysis_result, video_duration = main(video_name)

    # 对 main 返回的结果进行后续处理
    video_duration = round(video_duration, 2)
    # 将准确率从 0.0-1.0 转换为 0-100
    api_accuracy = round(analysis_result * 100)

    # 上传结果到 API
    requests.post('https://www.lloydyide.cn/fitbotAPI/update_umi.php',
                             data={'email': email, 'accuracy': api_accuracy*0.01, 'time': video_duration,'month':datetime.date.today().month})

    # 构建完整的HTML响应 (这部分和上次修改一致，已经是引体向上的内容了)
    html_content = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 600px;
                margin: 0 auto;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 12px;">

        <div style="margin-bottom: 20px;">
            <h3 style="color: #333; margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 8px;">
                <span style="display: inline-block; width: 24px; height: 24px; background: #07C160;
                            color: white; text-align: center; border-radius: 50%; margin-right: 8px;">✓</span>
                动作分析报告
            </h3>
                <div>
        <div style="background: #f8f9fa;border-radius: 10px;flex: 1;box-shadow: 0 2px 5px rgba(0,0,0,0.1);padding:5px">
            <span style="display: block;color: #6c757d;font-size: 12px;">动作准确率</span>
            <span style="display: block;font-size: 14px;font-weight: bold;color: #343a40;margin-top: 5px;">{api_accuracy}%</span>
        </div>
        <div style="background: #f8f9fa;border-radius: 10px;flex: 1;box-shadow: 0 2px 5px rgba(0,0,0,0.1);padding:5px;margin-top:5px">
            <span style="display: block;color: #6c757d;font-size: 12px;">视频时长</span>
            <span style="display: block;font-size: 14px;font-weight: bold;color: #343a40;margin-top: 5px;">{video_duration} 秒</span>
        </div>
         <div style="background: #f8f9fa;border-radius: 10px;flex: 1;box-shadow: 0 2px 5px rgba(0,0,0,0.1);padding:5px;margin-top:5px">
             <span style="display: block;color: #6c757d;font-size: 12px;">完成次数</span>
             <span style="display: block;font-size: 14px;font-weight: bold;color: #343a40;margin-top: 5px;">{count} 次</span>
        </div>
    </div>
            {"".join(
        f'<div style="padding:12px; margin:8px 0; background:#fff; border-radius:8px;'
        f'box-shadow:0 1px 3px rgba(0,0,0,0.1); border-left:4px solid '
        f'{"#07C160" if "标准" in item else "#F0AD4E" if "建议" in item or "不足" in item or "不够" in item else "#DD524D"};'
        f'"><span style="font-size: 14px;">{item}</span></div>'
        for item in feedback
    )}
        </div>

        <div style="background: white; border-radius: 10px; padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h3 style="color: #333; margin-top: 0; margin-bottom: 15px;">
                <span style="display: inline-block; width: 24px; height: 24px; background: #007AFF;
                            color: white; text-align: center; border-radius: 50%; margin-right: 8px;">🔍</span>
                详细运动数据
            </h3>
    """

    # 添加每个引体向上的数据 (这部分逻辑也和上次一样)
    start_elbow_angles = data.get('start', {}).get('elbow_angle', [])
    start_arm_torso_angles = data.get('start', {}).get('arm_torso_angle', [])
    end_elbow_angles = data.get('end', {}).get('elbow_angle', [])
    end_arm_torso_angles = data.get('end', {}).get('arm_torso_angle', [])

    for i in range(count):
        # 稍微安全地访问列表，以防万一 count 和列表长度不匹配
        start_elbow_display = f"{start_elbow_angles[i]:.1f}°" if i < len(start_elbow_angles) else "N/A"
        start_arm_torso_display = f"{start_arm_torso_angles[i]:.1f}°" if i < len(start_arm_torso_angles) else "N/A"
        end_elbow_display = f"{end_elbow_angles[i]:.1f}°" if i < len(end_elbow_angles) else "N/A"
        end_arm_torso_display = f"{end_arm_torso_angles[i]:.1f}°" if i < len(end_arm_torso_angles) else "N/A"

        html_content += f"""
        <div style="margin-bottom: 15px; padding: 12px; background: #f8f9fa; border-radius: 8px;">
            <div style="font-weight: 600; color: #007AFF; margin-bottom: 8px;">第 {i + 1} 次引体向上</div>
            <div style="display: flex; justify-content: space-between; font-size: 13px;">
                <div style="flex: 1; padding: 8px; background: rgba(0,122,255,0.05); border-radius: 6px; text-align: center;">
                    <div style="font-weight: 500; color: #555; margin-bottom: 5px;">起始姿势</div>
                    <div>
                        <span style="font-size: 12px; color: #888;">肘关节:</span>
                        <span style="font-weight: 500;">{start_elbow_display}</span>
                    </div>
                    <div>
                        <span style="font-size: 12px; color: #888;">臂-躯干:</span>
                        <span style="font-weight: 500;">{start_arm_torso_display}</span>
                    </div>
                </div>
                <div style="flex: 1; padding: 8px; background: rgba(76,217,100,0.1); border-radius: 6px; margin-left: 10px; text-align: center;">
                     <div style="font-weight: 500; color: #555; margin-bottom: 5px;">顶点姿势</div>
                    <div>
                        <span style="font-size: 12px; color: #888;">肘关节:</span>
                        <span style="font-weight: 500;">{end_elbow_display}</span>
                    </div>
                    <div>
                        <span style="font-size: 12px; color: #888;">臂-躯干:</span>
                        <span style="font-weight: 500;">{end_arm_torso_display}</span>
                    </div>
                </div>
            </div>
        </div>
        """

    html_content += """
        </div>
    </div>
    """

    return HttpResponse(html_content)