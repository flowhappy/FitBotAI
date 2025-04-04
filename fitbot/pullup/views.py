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
    'arm_angle': {
        'start': 170,  # 起始时手臂接近伸直
        'end': 90,     # 最高点时手臂弯曲90度
        'threshold': 15,  # 允许的误差范围
        'description': "手臂弯曲角度应在90°-170°之间变化"
    },
    'torso_angle': {
        'vertical': 180,  # 理想躯干垂直角度
        'threshold': 10,  # 允许的前后摆动范围
        'description': "躯干应保持垂直，避免过度摆动"
    }
}


def analyze_pullup_quality(data, standard_angles=STANDARD_PULLUP_ANGLES):
    """
    分析引体向上动作质量
    返回格式保持与原始analyze_squat_quality一致:
    {
        'start': { 'arm_angle': [], 'torso_angle': [], 'feedback': [] },
        'end': { 'arm_angle': [], 'torso_angle': [], 'feedback': [] },
        'accuracy': 0,
        'feedback': [],
        'score': 0
    }
    """
    result = {
        'start': {
            'arm_angle': [],
            'torso_angle': [],
            'feedback': []
        },
        'end': {
            'arm_angle': [],
            'torso_angle': [],
            'feedback': []
        },
        'accuracy': 0,
        'feedback': [],
        'score': 0
    }

    # 提取关键角度数据（假设data中包含手臂和躯干角度）
    arm_angles = data.get('arm_angle', [])
    torso_angles = data.get('torso_angle', [])

    if not arm_angles:
        result['feedback'].append("未检测到有效的手臂角度数据")
        return result

    # 1. 分析起始和结束姿势
    start_arm = arm_angles[0] if arm_angles else 0
    end_arm = arm_angles[-1] if arm_angles else 0

    # 起始姿势评估（手臂应接近伸直）
    result['start']['arm_angle'] = [start_arm]
    if start_arm < standard_angles['arm_angle']['start'] - standard_angles['arm_angle']['threshold']:
        result['start']['feedback'].append("起始姿势手臂未充分伸直")
    else:
        result['start']['feedback'].append("起始姿势良好")

    # 结束姿势评估（手臂应充分弯曲）
    result['end']['arm_angle'] = [end_arm]
    if end_arm > standard_angles['arm_angle']['end'] + standard_angles['arm_angle']['threshold']:
        result['end']['feedback'].append("动作未完成：下巴未超过单杠")
    else:
        result['end']['feedback'].append("动作完成度良好")

    # 2. 躯干稳定性分析（可选）
    if torso_angles:
        start_torso = torso_angles[0] if torso_angles else 0
        end_torso = torso_angles[-1] if torso_angles else 0

        result['start']['torso_angle'] = [start_torso]
        result['end']['torso_angle'] = [end_torso]

        torso_deviation = abs(start_torso - standard_angles['torso_angle']['vertical'])
        if torso_deviation > standard_angles['torso_angle']['threshold']:
            result['start']['feedback'].append(f"躯干前倾/后仰({start_torso:.1f}°)")

    # 3. 计算动作准确度（在标准范围内的帧比例）
    valid_frames = 0
    for angle in arm_angles:
        if (angle <= standard_angles['arm_angle']['start'] + standard_angles['arm_angle']['threshold'] and
                angle >= standard_angles['arm_angle']['end'] - standard_angles['arm_angle']['threshold']):
            valid_frames += 1

    result['accuracy'] = (valid_frames / len(arm_angles)) * 100 if arm_angles else 0

    # 4. 综合评分（保持与原始代码相同的0-100范围）
    # 评分逻辑：
    # - 50% 动作完成度（是否达到起始和结束标准）
    # - 30% 动作准确度
    # - 20% 躯干稳定性

    completion_score = 50 if (
            start_arm >= standard_angles['arm_angle']['start'] - standard_angles['arm_angle']['threshold'] and
            end_arm <= standard_angles['arm_angle']['end'] + standard_angles['arm_angle']['threshold']
    ) else 20

    accuracy_score = min(30, result['accuracy'] * 0.3)

    stability_score = 0
    if torso_angles:
        avg_deviation = np.mean([abs(a - standard_angles['torso_angle']['vertical']) for a in torso_angles])
        stability_score = max(0, 20 - (avg_deviation * 0.5))

    result['score'] = min(100, completion_score + accuracy_score + stability_score)

    # 5. 综合反馈
    if result['score'] >= 85:
        result['feedback'].append("优秀的引体向上动作！")
    elif result['score'] >= 60:
        result['feedback'].append("动作基本规范，还有改进空间")
    else:
        result['feedback'].append("需要更多练习，注意动作规范")

    return result


def detect_pullups(landmarks, frame_width, frame_height, prev_state=None):
    """
    检测引体向上动作（保持与原始detect_squats相同的返回结构）
    返回:
    {
        'pose_landmarks': landmarks,
        'state': 'up'/'down',
        'counter': int,
        'feedback': str,
        'angles': {
            'arm_angle': float,
            'torso_angle': float
        }
    }
    """
    if not landmarks or not hasattr(landmarks, 'pose_landmarks'):
        return {
            'pose_landmarks': None,
            'state': None,
            'counter': 0,
            'feedback': "未检测到人体",
            'angles': {}
        }

    mp_pose = mp.solutions.pose
    pose_landmarks = landmarks.pose_landmarks

    # 获取关键点索引（根据MediaPipe Pose定义）
    shoulder_left = [pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame_width,
                     pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height]
    elbow_left = [pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * frame_width,
                  pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * frame_height]
    wrist_left = [pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * frame_width,
                  pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * frame_height]

    shoulder_right = [pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame_width,
                      pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height]
    elbow_right = [pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * frame_width,
                   pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * frame_height]
    wrist_right = [pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame_width,
                   pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame_height]

    # 计算手臂角度（肘关节角度）
    left_arm_angle = calculate_angle_3d(shoulder_left, elbow_left, wrist_left)
    right_arm_angle = calculate_angle_3d(shoulder_right, elbow_right, wrist_right)
    arm_angle = (left_arm_angle + right_arm_angle) / 2  # 取平均值

    # 计算躯干角度（相对于垂直线的倾斜）
    hip_left = [pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * frame_width,
                pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * frame_height]
    hip_right = [pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * frame_width,
                 pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * frame_height]
    shoulder_center = [(shoulder_left[0] + shoulder_right[0]) / 2,
                       (shoulder_left[1] + shoulder_right[1]) / 2]
    hip_center = [(hip_left[0] + hip_right[0]) / 2,
                  (hip_left[1] + hip_right[1]) / 2]

    # 躯干垂直线（从肩膀到髋部）
    torso_angle = calculate_angle_3d(
        [shoulder_center[0], shoulder_center[1] - 50],  # 虚拟上延长点
        shoulder_center,
        hip_center
    )

    # 状态判断（基于手臂角度）
    current_state = 'up' if arm_angle > 100 else 'down'  # 阈值可调整

    # 初始化返回结果（保持与原始函数相同的结构）
    result = {
        'pose_landmarks': pose_landmarks,
        'state': current_state,
        'counter': prev_state['counter'] if prev_state else 0,
        'feedback': "",
        'angles': {
            'arm_angle': arm_angle,
            'torso_angle': torso_angle
        }
    }

    # 状态转换计数（从伸直->弯曲->伸直算1次）
    if prev_state and prev_state['state'] == 'up' and current_state == 'down':
        result['counter'] = prev_state['counter'] + 1
        result['feedback'] = f"计数: {result['counter']}"
    elif prev_state and prev_state['state'] == 'down' and current_state == 'up':
        result['feedback'] = "完成一次引体向上"

    # 姿势质量反馈
    if torso_angle > 15:  # 躯干摆动阈值
        result['feedback'] += " | 注意保持躯干稳定"

    if current_state == 'up' and arm_angle < 170:
        result['feedback'] += " | 手臂未完全伸直"

    return result


def main(video_name):
    # 下载视频
    subprocess.call(['you-get', 'https://www.lloydyide.cn/fitbotAPI/uploads/' + video_name])
    cap = cv2.VideoCapture(video_name)

    # 获取视频参数
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps

    # 初始化MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)

    # 初始化变量
    action_data = {
        'start': {'time': [], 'elbow_angle': []},
        'end': {'time': [], 'elbow_angle': []},
        'counter': 0
    }
    prev_state = None
    feedback = []
    analysis_result = []

    # 视频输出设置
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"analyzed_{video_name}", fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # 调用detect_pullups处理每一帧
            landmarks = results.pose_landmarks.landmark
            counter, state, angle = detect_pullups(
                landmarks=landmarks,
                frame_width=frame_width,
                frame_height=frame_height,
                prev_state=prev_state
            )

            # 更新状态和数据
            prev_state = state
            action_data['counter'] = counter

            # 记录动作数据（示例：仅记录下拉阶段）
            if state == "down":
                action_data['start']['time'].append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                action_data['start']['elbow_angle'].append(angle)
            elif state == "up":
                action_data['end']['time'].append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                action_data['end']['elbow_angle'].append(angle)


        # 写入输出视频
        out.write(frame)

    # 分析动作质量
    feedback, analysis_result = analyze_pullup_quality(action_data)

    cap.release()
    out.release()
    pose.close()
    return action_data['counter'], action_data, feedback, analysis_result, video_duration


def shipin(request):
    email = request.GET.get('email')
    video_name = requests.get('https://www.lloydyide.cn/fitbotAPI/downLoad.php', params={"email": email})

    # 获取分析结果（现在返回5个值）
    total_pullups, action_data, feedback, analysis_result, video_duration = main(video_name.text)

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
                引体向上分析报告
            </h3>
            <div>
                <div style="background: #f8f9fa;border-radius: 10px;flex: 1;box-shadow: 0 2px 5px rgba(0,0,0,0.1);padding:5px">
                    <span style="display: block;color: #6c757d;font-size: 12px;">完成次数</span>
                    <span style="display: block;font-size: 16px;font-weight: bold;color: #343a40;margin-top: 5px;">{total_pullups} 次</span>
                </div>
                <div style="background: #f8f9fa;border-radius: 10px;flex: 1;box-shadow: 0 2px 5px rgba(0,0,0,0.1);padding:5px;margin-top:5px">
                    <span style="display: block;color: #6c757d;font-size: 12px;">动作完成度</span>
                    <span style="display: block;font-size: 16px;font-weight: bold;color: #343a40;margin-top: 5px;">{analysis_result.get('completion_rate', 0):.1f}%</span>
                </div>
                <div style="background: #f8f9fa;border-radius: 10px;flex: 1;box-shadow: 0 2px 5px rgba(0,0,0,0.1);padding:5px;margin-top:5px">
                    <span style="display: block;color: #6c757d;font-size: 12px;">视频时长</span>
                    <span style="display: block;font-size: 16px;font-weight: bold;color: #343a40;margin-top: 5px;">{video_duration:.1f} 秒</span>
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

        <!-- 引体向上数据详情 -->
        <div style="background: white; border-radius: 10px; padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h3 style="color: #333; margin-top: 0; margin-bottom: 15px;">
                <span style="display: inline-block; width: 24px; height: 24px; background: #007AFF; 
                            color: white; text-align: center; border-radius: 50%; margin-right: 8px;">🔍</span>
                详细运动数据
            </h3>
    """

    # 添加每个引体向上的数据
    for i in range(min(total_pullups, len(action_data['start']['arm_angle']))):
        html_content += f"""
        <div style="margin-bottom: 15px; padding: 12px; background: #f8f9fa; border-radius: 8px;">
            <div style="font-weight: 600; color: #007AFF; margin-bottom: 8px;">第 {i + 1} 次引体向上</div>
            <div style="display: flex; justify-content: space-between;">
                <div style="flex: 1; padding: 8px; background: rgba(0,122,255,0.05); border-radius: 6px;">
                    <div style="margin-top: 4px;">
                        <span style="font-size: 12px; color: #888;">手臂(起始):</span> 
                        <span style="font-weight: 500;">{action_data['start']['arm_angle'][i]:.1f}°</span>
                    </div>
                    <div>
                        <span style="font-size: 12px; color: #888;">躯干(起始):</span> 
                        <span style="font-weight: 500;">{action_data['start']['torso_angle'][i]:.1f}°</span>
                    </div>
                </div>
                <div style="flex: 1; padding: 8px; background: rgba(0,122,255,0.05); border-radius: 6px; margin-left: 10px;">
                    <div style="margin-top: 4px;">
                        <span style="font-size: 12px; color: #888;">手臂(顶峰):</span> 
                        <span style="font-weight: 500;">{action_data['end']['arm_angle'][i]:.1f}°</span>
                    </div>
                    <div>
                        <span style="font-size: 12px; color: #888;">躯干(顶峰):</span> 
                        <span style="font-weight: 500;">{action_data['end']['torso_angle'][i]:.1f}°</span>
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

