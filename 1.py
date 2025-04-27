import cv2
import mediapipe as mp
import pandas as pd
import math
import time  # 导入时间模块

# 初始化计时器
last_time = time.time()
interval = 1  # 时间间隔（秒）
index=0
good_score=20 #如果达到这个分数，换成下一个动作
# 读取静态图像
static_image_path = "test2.jpg"
static_image = cv2.imread(static_image_path)

if static_image is None:
    raise FileNotFoundError(f"Image not found at {static_image_path}")
# 调整静态图像大小
static_image_height = 480  # 静态图像目标高度
static_image_width =300   # 静态图像目标宽度
static_image = cv2.resize(static_image, (static_image_width, static_image_height))

# 另一张图片，假设当 score 达到 70 时切换
new_image_path = "test.jpg"
new_image = cv2.imread(new_image_path)
if new_image is None:
    raise FileNotFoundError(f"Image not found at {new_image_path}")

new_image = cv2.resize(new_image, (static_image_width, static_image_height))

# 标准动作数据，你可以选择其他图片的数据，这里仅用两张作为参考
standard=[26.333088377451883, 66.92274698892035, 69.40410287641619, 88.46159285835151, 94.32846798052309, 95.03122396457627, 104.61976182588778, 119.26570565187514]
standard2=[15.741107355697773, 20.342408975036385, 54.83069013433787, 90.24464608079013, 33.17064773438967, 131.82305965076074, 63.32851594712265, 115.18073942143322]
# 由每个点的位置计算向量
def get_vectors(data):
    def get(a,b):
        vector=[]
        vector.append(data.loc[a]['x']-data.loc[b]['x'])
        vector.append(data.loc[a]['y'] - data.loc[b]['y'])
        vector.append(data.loc[a]['z'] - data.loc[b]['z'])
        return vector
# 关节位置参考图示
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


# 初始化 MediaPipe Holistic 模型
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
image_shown = static_image  # 控制显示的图像

# 打开摄像头
cap = cv2.VideoCapture(0)

# 使用 Holistic 模型进行人体关键点识别
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        # 转换为 RGB 图像，因为 MediaPipe 使用 RGB 输入
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # 处理图像并获取结果
        results = holistic.process(image)

        # 转回 BGR 图像以便在 OpenCV 中显示
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not results.pose_landmarks:
            print("No pose landmarks detected, skipping this frame.")
            continue
        # 绘制身体骨架（包括手部、脸部和身体关键点）
        datas = results.pose_landmarks
        # 绘制身体骨架（包括手部、脸部和身体关键点）
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
        data = pd.DataFrame(landmarks_data)
        data = data.drop(['visibility'], axis=1)

        # 获取当前时间
        current_time = time.time()
        if current_time - last_time >= interval:
            # 更新计时器
            last_time = current_time
            # 输出一些动作参数，先得到四肢关节的位置，再将一些特定关节连成向量
            # 然后再计算一些向量的角度，来判断是否做对
            # 输出关节向量
            # print(get_vectors(data))
            data_vectors = get_vectors(data)
            data_vectors_df = pd.DataFrame(data_vectors)
            # 为了便于观察,这里转换成了dataframe
            print(data_vectors_df)
            angles = get_angles(data_vectors)
            print(angles)
            difference=[]
            for i in range(len(standard)):
                difference.append(abs(angles[i]-standard[i]))
                # 获取的角度列表与标准的角度列表每个元素分别作差得到difference列表
            print("difference:")
            print(difference)
            score=0
            # 100分为满分，角度小于20，加12.5分，小于40大于20，加5分，大于40，扣3分
            for i in difference:
                if i<20:
                    score+=12.5
                elif i>=20 and i<40:
                    score+=5
                elif i>=40:
                    score-=3
            print("score:")
            print(score)
            if score >= 90:  # 根据分数段，打印不同的评价结果
                cv2.putText(
                    image, "Unbelievable!!!", (400, 400),  # 5显示位置
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5, cv2.LINE_AA  # 字体、大小、颜色和粗细
                )
            elif score >= 80 and score < 90:
                cv2.putText(
                    image, "Great!!", (400, 400),  # 5显示位置
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5, cv2.LINE_AA  # 字体、大小、颜色和粗细
                )
            elif score < 80 and score > 70:
                cv2.putText(
                    image, "Very Good!", (370, 400),  # 5显示位置
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 165, 0), 5, cv2.LINE_AA  # 字体、大小、颜色和粗细
                )
            elif score <= 70 and score > 60:
                cv2.putText(
                    image, "Not Bad", (370, 400),  # 5显示位置
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 100, 0), 5, cv2.LINE_AA  # 字体、大小、颜色和粗细
                )
            else:
                cv2.putText(
                    image, "Try Again", (350, 400),  # 5显示位置
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 200), 5, cv2.LINE_AA  # 字体、大小、颜色和粗细
                )
        # 将分数显示在视频帧上
        cv2.putText(
            image, f"Score: {score}", (50, 50),  # 显示位置
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5, cv2.LINE_AA  # 字体、大小、颜色和粗细
        )
        # 如果分数超过 70，切换图像
        # 这里可以修改切换下一个动作的标准
        if score >= good_score and image_shown is static_image:
            # 切换到另一张图的标准信息
            standard=standard2
            # 后续如果图片动作很多，可以靠这个index切换不同序号的动作
            index=index+1
            # 关闭当前显示图片的窗口
            cv2.destroyWindow('Static Image')
            # 切换到新图片
            image_shown = new_image  # 切换到新图片
            print("Switching to new image")


        # combined_frame = cv2.hconcat([video_frame_resized, static_image])

        # 调整视频帧大小
        video_frame_resized = cv2.resize(image, (640, 480))

        # 在新窗口中显示视频流
        cv2.imshow('Real-time Video Stream', video_frame_resized)

        # 在新窗口中显示静态图像
        cv2.imshow('Static Image', image_shown)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 释放资源
cap.release()
cv2.destroyAllWindows()
