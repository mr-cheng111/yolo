import cv2
import numpy as np
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO('./models/best.pt')

# 相机参数
camera_params = {
    "image_width": 640,
    "image_height": 480,
    "camera_matrix": np.array([[657.6599073397491, 0, 297.2104358756092],
                               [0, 660.4187664212818, 315.258308139506],
                               [0, 0, 1]]),
    "distortion_coefficients": np.array([-0.641008058598746, 0.4915926910386066, 0.01285490678706567, -0.006399165615193381, -0.1774630528991807]),
    "rectification_matrix": np.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]]),
    "projection_matrix": np.array([[541.4896621146351, 0, 281.6524049179836, 0],
                                   [0, 563.5689544350256, 334.7746185533754, 0],
                                   [0, 0, 1, 0]])
}
# 目标物体的3D坐标（假设为正方形的四个角）
object_points = np.array([
    [-0.0125, -0.0125, 0],
    [ 0.0125, -0.0125, 0],
    [ 0.0125,  0.0125, 0],
    [-0.0125,  0.0125, 0]
], dtype=np.float32)

camera_matrix = camera_params["camera_matrix"]
dist_coeffs = camera_params["distortion_coefficients"]
dist_coeffs = camera_params["distortion_coefficients"]
rectification_matrix = camera_params["rectification_matrix"]
projection_matrix = camera_params["projection_matrix"]

# 计算矫正映射
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, rectification_matrix, projection_matrix, 
                                             (camera_params["image_width"], camera_params["image_height"]), cv2.CV_16SC2)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 进行图像矫正
    frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    # YOLOv8 推理
    results = model(frame)

    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 获取检测框区域
            roi = frame[y1:y2, x1:x2]

            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # 定义红色的HSV范围
            lower_red = np.array([0, 120, 70])
            upper_red = np.array([10, 255, 255])

            # 创建颜色掩码
            mask = cv2.inRange(hsv, lower_red, upper_red)

            # 使用形态学操作清理掩码
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # 找到轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # 过滤掉较小的轮廓
                    # 获取最小外接矩形
                    rect = cv2.minAreaRect(contour)
                    box_points = cv2.boxPoints(rect)
                    box_points = np.int0(box_points)

                    # 绘制矩形的四个顶点
                    for i in range(4):
                        cv2.circle(frame, (box_points[i][0] + x1, box_points[i][1] + y1), 5, (0, 255, 0), -1)

                    # PNP计算
                    image_points = np.array([
                        [box_points[0][0] + x1, box_points[0][1] + y1],
                        [box_points[1][0] + x1, box_points[1][1] + y1],
                        [box_points[2][0] + x1, box_points[2][1] + y1],
                        [box_points[3][0] + x1, box_points[3][1] + y1]
                    ], dtype=np.float32)
                    
                    _, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

                    # 绘制检测框和标签
                    label = f"{model.names[cls]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    # 显示旋转向量和平移向量
                    rotation_text = f"Rvec: {rvec.flatten()}"
                    translation_text = f"Tvec: {tvec.flatten()}"
                    cv2.putText(frame, rotation_text, (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(frame, translation_text, (x1, y2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 显示处理结果
    cv2.imshow("YOLOv8 Detection with PnP", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()