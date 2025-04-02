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
    'start': {  # 高点位置（起始姿势）
        'elbow': 140,    # 肘关节应接近完全伸直（170-180°）
        'torso': 80     # 躯干应完全挺直（175-180°）
    },
    'end': {    # 低点位置（最低点）
        'elbow': 80,     # 肘关节标准90°（80-100°可接受）
        'torso': 90     # 躯干保持挺直（165-175°）
    }
}



def analyze_pushup_quality(user_data):
    """智能分析俯卧撑质量"""
    feedback = []

    # 肘关节分析
    user_elbow_start = np.mean(user_data['start']['elbow_angle'])
    std_elbow_start = STANDARD_PUSHUP_ANGLES['start']['elbow']

    if user_elbow_start < std_elbow_start - 5:
        feedback.append("起始姿势肘关节未完全伸直")
    elif user_elbow_start > std_elbow_start + 5:
        feedback.append("起始姿势肘关节过度伸展")

    # 躯干稳定性分析
    user_torso_start = np.mean(user_data['start']['torso_angle'])
    std_torso_start = STANDARD_PUSHUP_ANGLES['start']['torso']

    if abs(user_torso_start - std_torso_start) > 10:
        feedback.append("身体未保持直线，注意核心收紧")

    # 俯卧撑深度分析
    user_elbow_end = np.mean(user_data['end']['elbow_angle'])
    std_elbow_end = STANDARD_PUSHUP_ANGLES['end']['elbow']

    if user_elbow_end > std_elbow_end + 15:
        feedback.append("俯卧撑深度不足，胸部应更接近地面")
    elif user_elbow_end < std_elbow_end - 15:
        feedback.append("俯卧撑过深，可能对肩关节造成压力")

    # 躯干稳定性分析
    user_torso_end = np.mean(user_data['end']['torso_angle'])
    std_torso_end = STANDARD_PUSHUP_ANGLES['end']['torso']

    if abs(user_torso_end - std_torso_end) > 10:
        feedback.append("动作过程中身体晃动，注意保持核心稳定")

    accuracy = 1  # 待实现准确率计算

    return feedback if feedback else ["动作标准，继续保持！"], accuracy


def detect_pushups(video_path, output_video_path, display=False):
    action_state = 'rest'
    ELBOW_ANGLE_THRESHOLD_START = 160  # 动作开始阈值
    ELBOW_ANGLE_THRESHOLD_END = 140  # 动作结束阈值
    FRAME_COUNT_THRESHOLD = 5
    frame_count = 0
    pushup_count = 0
    elbow_angles = []
    action_data = []  # 存储动作数据的列表

    stage_data = {
        'rest': {'elbow_angle': [], 'torso_angle': []},
        'start': {'elbow_angle': [], 'torso_angle': []},
        'moving': {'elbow_angle': [], 'torso_angle': []},
        'end': {'elbow_angle': [], 'torso_angle': []},
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
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                left_arm_angle = calculate_angle_3d(left_elbow, left_shoulder, left_hip)
                right_arm_angle = calculate_angle_3d(right_elbow, right_shoulder, right_hip)
                left_leg_angle = calculate_angle_3d(left_knee, left_hip, left_shoulder)
                right_leg_angle = calculate_angle_3d(right_knee, right_hip, right_shoulder)
                torso_angle = calculate_torso_angle_3d(left_hip, left_shoulder)

                left_elbow_angle = calculate_angle_3d(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle_3d(right_shoulder, right_elbow, right_wrist)

                elbow_angle = (left_elbow_angle + right_elbow_angle) / 2  # 改为计算双肘平均角度
                elbow_angles.append(elbow_angle)  # 改为存储肘关节角度
                if len(elbow_angles) > 5:  # 保持5帧滑动窗口
                    elbow_angles.pop(0)
                average_elbow_angle = sum(elbow_angles) / len(elbow_angles)  # 计算平均肘关节角度

                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)

                # 输出文本
                draw.text((50, 150), f'肘关节角度: {int(average_elbow_angle)}°', font=font, fill=(255, 255, 0))
                draw.text((50, 200), f'左臂-躯干角度: {int(left_arm_angle)}°', font=font, fill=(255, 255, 255))
                draw.text((50, 230), f'右臂-躯干角度: {int(right_arm_angle)}°', font=font, fill=(255, 255, 255))
                draw.text((50, 260), f'左腿-躯干角度: {int(left_leg_angle)}°', font=font, fill=(255, 255, 255))
                draw.text((50, 290), f'右腿-躯干角度: {int(right_leg_angle)}°', font=font, fill=(255, 255, 255))
                draw.text((50, 320), f'躯干倾斜角度: {int(torso_angle)}°', font=font, fill=(255, 255, 0))
                draw.text((50, 100), f'深蹲次数: {pushup_count}', font=font, fill=(0, 0, 255))

                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                # 动作检测逻辑
                if action_state == 'rest':
                    if average_elbow_angle < ELBOW_ANGLE_THRESHOLD_START:
                        frame_count += 1
                        if frame_count >= FRAME_COUNT_THRESHOLD:
                            action_state = 'start'
                            frame_count = 0
                            start_frame = frame_number
                            print(f"第 {pushup_count + 1} 次深蹲开始于第 {start_frame} 帧")
                            current_action_data = []

                            # 新增：立即记录start阶段数据（原代码缺少这部分）
                            stage_data['start']['elbow_angle'].append(average_elbow_angle)
                            stage_data['start']['torso_angle'].append(torso_angle)

                            stage_data['rest']['elbow_angle'].append(average_elbow_angle)
                            stage_data['rest']['torso_angle'].append(torso_angle)

                    else:
                        frame_count = 0

                elif action_state == 'start':
                    action_state = 'moving'
                    stage_data['start']['elbow_angle'].append(average_elbow_angle)
                    stage_data['start']['torso_angle'].append(torso_angle)

                elif action_state == 'moving':
                    frame_data = {
                        'left_arm_angle': left_arm_angle,
                        'right_arm_angle': right_arm_angle,
                        'left_leg_angle': left_leg_angle,
                        'right_leg_angle': right_leg_angle,
                        'torso_angle': torso_angle,
                        'average_elbow_angle': average_elbow_angle
                    }
                    current_action_data.append(frame_data)

                    if average_elbow_angle > ELBOW_ANGLE_THRESHOLD_END:
                        frame_count += 1
                        if frame_count >= FRAME_COUNT_THRESHOLD:
                            # 记录end阶段数据（确保每次结束都执行）
                            stage_data['end']['elbow_angle'].append(average_elbow_angle)
                            stage_data['end']['torso_angle'].append(torso_angle)

                            action_state = 'end'
                            frame_count = 0
                            end_frame = frame_number
                            pushup_count += 1
                            print(f"第 {pushup_count} 次深蹲结束于第 {end_frame} 帧")
                            action_data.extend(current_action_data)

                elif action_state == 'end':
                    action_state = 'rest'
                    stage_data['end']['elbow_angle'].append(average_elbow_angle)
                    stage_data['end']['torso_angle'].append(torso_angle)

                # 输出每个阶段的平均值
                if action_state != 'rest':
                    if len(stage_data[action_state]['elbow_angle']) > 0:
                        avg_knee = np.mean(stage_data[action_state]['elbow_angle'])
                        print(f"阶段 {action_state} 肘关节角度平均值: {avg_knee:.2f}°")
                    if len(stage_data[action_state]['torso_angle']) > 0:
                        avg_torso = np.mean(stage_data[action_state]['torso_angle'])
                        print(f"阶段 {action_state} 躯干角度平均值: {avg_torso:.2f}°")

                # 将帧写入输出视频文件
                out.write(image)

            else:
                print(f"在第 {frame_number} 帧未检测到关键点。")

        print(f"总共完成了 {pushup_count} 次深蹲。")
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        feedback,analysis_result = analyze_pushup_quality(stage_data)
        print("\n动作分析报告:")

    return pushup_count, stage_data, feedback, analysis_result


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


def shipin(request):
    email = request.GET.get('email')
    video_name = requests.get('https://www.lloydyide.cn/fitbotAPI/downLoad.php', params={"email": email})
    count, data, feedback, analysis_result, video_duration = main(video_name.text)

    # 构建完整的HTML响应
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
                动作分析报告
            </h3>
                <div>
        <div style="background: #f8f9fa;border-radius: 10px;flex: 1;box-shadow: 0 2px 5px rgba(0,0,0,0.1);padding:5rpx">
            <span style="display: block;color: #6c757d;font-size: 10rpx;">动作准确率</span>
            <span style="display: block;font-size: 10rpx;font-weight: bold;color: #343a40;margin-top: 5px;">{analysis_result}</span>
        </div>
        <div style="background: #f8f9fa;border-radius: 10px;flex: 1;box-shadow: 0 2px 5px rgba(0,0,0,0.1);padding:5rpx;margin-top:5rpx">
            <span style="display: block;color: #6c757d;font-size: 10rpx;">视频时长</span>
            <span style="display: block;font-size: 10rpx;font-weight: bold;color: #343a40;margin-top: 5px;">{video_duration}</span>
        </div>
    </div>
            {"".join(
        f'<div style="padding:12px; margin:8px 0; background:#fff; border-radius:8px;'
        f'box-shadow:0 1px 3px rgba(0,0,0,0.1); border-left:4px solid '
        f'{"#07C160" if "标准" in item else "#F0AD4E" if "建议" in item else "#DD524D"};'
        f'">{item}</div>'
        for item in feedback
    )}
        </div>

        <!-- 深蹲数据详情 -->
        <div style="background: white; border-radius: 10px; padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h3 style="color: #333; margin-top: 0; margin-bottom: 15px;">
                <span style="display: inline-block; width: 24px; height: 24px; background: #007AFF; 
                            color: white; text-align: center; border-radius: 50%; margin-right: 8px;">🔍</span>
                详细运动数据
            </h3>
    """

    # 添加每个深蹲的数据
    for i in range(count):
        start_data = {
            'knee': np.mean(data['start']['elbow_angle']),
            'torso': np.mean(data['start']['torso_angle'])
        }
        end_data = {
            'knee': np.mean(data['end']['elbow_angle']),
            'torso': np.mean(data['end']['torso_angle'])
        }

        html_content += f"""
        <div style="margin-bottom: 15px; padding: 12px; background: #f8f9fa; border-radius: 8px;">
            <div style="font-weight: 600; color: #007AFF; margin-bottom: 8px;">第 {i + 1} 次俯卧撑</div>
            <div style="display: flex; justify-content: space-between;">
                <div style="flex: 1; padding: 8px; background: rgba(0,122,255,0.05); border-radius: 6px;">
                    <div style="margin-top: 4px;">
                        <span style="font-size: 12px; color: #888;">肘关节:</span> 
                        <span style="font-weight: 500;">{data['start']['elbow_angle'][i]:.1f}°</span>
                    </div>
                    <div>
                        <span style="font-size: 12px; color: #888;">躯干:</span> 
                        <span style="font-weight: 500;">{data['start']['torso_angle'][i]:.1f}°</span>
                    </div>
                </div>
                <div style="flex: 1; padding: 8px; background: rgba(0,122,255,0.05); border-radius: 6px; margin-left: 10px;">
                    <div style="margin-top: 4px;">
                        <span style="font-size: 12px; color: #888;">肘关节:</span> 
                        <span style="font-weight: 500;">{data['end']['elbow_angle'][i]:.1f}°</span>
                    </div>
                    <div>
                        <span style="font-size: 12px; color: #888;">躯干:</span> 
                        <span style="font-weight: 500;">{data['end']['torso_angle'][i]:.1f}°</span>
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
