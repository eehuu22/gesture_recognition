# gesture_recognition
基于MediaPipe的动作识别（The action posture recognition based on MediaPipe）
## 用到的库
openCV, mediapipe
## 描述
基于MediaPipe 的 33 个人体关键点模型，人工分析，得出最有特征的8
个角度（图中（vector index.png）橙色的0-7）， 通过计算人体各关节之间的角度，评估用户的姿势准
确性。首先，通过静态图像提取标准姿势的关键点数据，并计算出这些关键点之
间的角度作为参考标准。在实时视频流中，通过MediaPipe进行人体关键点检测，
提取每一帧中的关节位置并计算出对应的角度。然后，将实时计算的角度与标准
角度进行比较，根据差异来计算姿势得分。如果分数超过设定阈值（如70分），
则切换到新的标准姿势图像。

有啥需要解释的可以联系作者 1095893534@qq.com
