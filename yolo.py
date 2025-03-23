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
# 目标物体真实宽度（米）
REAL_WIDTH = 0.025  # 10 cm

camera_matrix = camera_params["camera_matrix"]
dist_coeffs = camera_params["distortion_coefficients"]
rectification_matrix = camera_params["rectification_matrix"]
projection_matrix = camera_params["projection_matrix"]

# 计算矫正映射
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, rectification_matrix, projection_matrix, 
                                             (camera_params["image_width"], camera_params["image_height"]), cv2.CV_16SC2)

# 距离估计
def estimate_distance(pixel_width):
    if pixel_width > 0:
        distance = (REAL_WIDTH * camera_matrix[0][0]) / pixel_width
        return distance
    return None

# 像素坐标转换为相机坐标
def pixel_to_camera_coords(u, v, depth):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return x, y, z

cap = cv2.VideoCapture(0)

# 逐帧处理
frame_count = 0
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
            if box.conf[0].item() > 0.85:
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1
                
                # 获取中心点
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # 估计距离
                distance = estimate_distance(width)

                # 转换为相机坐标系
                if distance is not None:
                    x, y, z = pixel_to_camera_coords(cx, cy, distance)
                    position_text = f"Position: X={x:.2f}m Y={y:.2f}m Z={z:.2f}m"
                    cv2.putText(frame, position_text, (x1, y2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # 获取框区域颜色
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    avg_color = cv2.mean(roi)[:3]  # BGR格式

                # 绘制检测框和标签
                label = f"{model.names[cls]} {conf:.2f} ({distance:.6f}m)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # 显示颜色
                color_text = f"Color: BGR({int(avg_color[0])}, {int(avg_color[1])}, {int(avg_color[2])})"
                cv2.putText(frame, color_text, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 显示处理结果（调试用）
    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()