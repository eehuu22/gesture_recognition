import cv2
import mediapipe as mp
import pandas as pd
import math
import time

def get_vectors(data):
    def get(a,b):
        vector=[]
        vector.append(data.loc[a]['x']-data.loc[b]['x'])
        vector.append(data.loc[a]['y'] - data.loc[b]['y'])
        vector.append(data.loc[a]['z'] - data.loc[b]['z'])
        return vector

    v=[]
    v.append(get(13,15))
    v.append(get(14,16))
    v.append(get(11, 13))
    v.append(get(12, 14))
    v.append(get(23, 11))
    v.append(get(24, 12))
    v.append(get(25, 23))
    v.append(get(26, 24))
    v.append(get(27, 25))
    v.append(get(28, 26))
    return v

# 由向量计算角度
def get_angle(vector1, vector2):
    # 计算点积
    dot_product = sum(a * b for a, b in zip(vector1, vector2))

    # 计算两个向量的模
    magnitude1 = math.sqrt(sum(a ** 2 for a in vector1))
    magnitude2 = math.sqrt(sum(b ** 2 for b in vector2))
    # 计算余弦值
    cos_theta = dot_product / (magnitude1 * magnitude2)
    # 避免因为浮点数精度问题，cos_theta 超出 [-1, 1] 范围
    cos_theta = max(-1.0, min(1.0, cos_theta))
    # 计算角度（单位：弧度）
    angle_radians = math.acos(cos_theta)
    # 将弧度转换为度数
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

# 获取每条肢体的角度，保存为angles
def get_angles(vectors):
    angles=[]
    angles.append(get_angle(vectors[0],vectors[2]))
    angles.append(get_angle(vectors[1], vectors[3]))
    angles.append(get_angle(vectors[2], vectors[4]))
    angles.append(get_angle(vectors[3], vectors[5]))
    angles.append(get_angle(vectors[4], vectors[6]))
    angles.append(get_angle(vectors[5], vectors[7]))
    angles.append(get_angle(vectors[6], vectors[8]))
    angles.append(get_angle(vectors[7], vectors[9]))
    return angles

# 读取静态图像
static_image_path = "test2.jpg"
static_image = cv2.imread(static_image_path)

if static_image is None:
    raise FileNotFoundError(f"Image not found at {static_image_path}")

# 调整静态图像大小
static_image_height = 480  # 静态图像目标高度
static_image_width = 200   # 静态图像目标宽度
static_image = cv2.resize(static_image, (static_image_width, static_image_height))

# 另一张图片，假设当 score 达到 70 时切换
new_image_path = "test.jpg"
new_image = cv2.imread(new_image_path)

if new_image is None:
    raise FileNotFoundError(f"Image not found at {new_image_path}")

new_image = cv2.resize(new_image, (static_image_width, static_image_height))

# 标准角度
standard = [26.333088377451883, 66.92274698892035, 69.40410287641619, 88.46159285835151,
            94.32846798052309, 95.03122396457627, 104.61976182588778, 119.26570565187514]

# 定义向量和角度计算函数（略）

# 初始化 MediaPipe Holistic 模型
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

last_time = time.time()
interval = 1
current_score = 0
image_shown = static_image  # 控制显示的图像

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 翻转图像，纠正镜像
        frame = cv2.flip(frame, 1)

        # 转换颜色并进行处理
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # 有时会跟丢人，导致没有数据而报错，这个if可以跳过那个时候
        if not results.pose_landmarks:
            print("No pose landmarks detected, skipping this frame.")
            continue
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            landmarks_data = []
            for landmark in results.pose_landmarks.landmark:
                landmarks_data.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            data = pd.DataFrame(landmarks_data).drop(['visibility'], axis=1)

            current_time = time.time()
            if current_time - last_time >= interval:
                last_time = current_time

                data_vectors = get_vectors(data)
                angles = get_angles(data_vectors)
                difference = [abs(angles[i] - standard[i]) for i in range(len(standard))]

                current_score = 0
                for diff in difference:
                    if diff < 20:
                        current_score += 12.5
                    elif 20 <= diff < 40:
                        current_score += 5
                    elif diff >= 40:
                        current_score -= 3

                print("difference:", difference)
                print("score:", current_score)

        # 显示分数
        cv2.putText(
            image, f"Score: {current_score}", (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
        )

        # 如果分数超过 70，切换图像
        if current_score >= 20 and image_shown is static_image:
            image_shown = new_image  # 切换到新图片
            print("Switching to new image")

        # 调整视频帧大小
        video_frame_resized = cv2.resize(image, (640, 480))

        # 在新窗口中显示视频流
        cv2.imshow('Real-time Video Stream', video_frame_resized)

        # 在新窗口中显示当前的静态图像
        cv2.imshow('Static Image', image_shown)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
