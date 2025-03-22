import cv2
import numpy as np
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO('yolov8n.pt')

# 相机内参（通过相机标定获取）
fx = 913.1  # 焦距（像素）
fy = 913.1
ppx = 646.8
ppy = 378.4

# 目标物体真实宽度（米）
REAL_WIDTH = 0.035  # 10 cm

# 距离估计
def estimate_distance(pixel_width):
    if pixel_width > 0:
        distance = (REAL_WIDTH * fx) / pixel_width
        return distance
    return None

# 打开视频文件
video_path = './data/rv.mp4'
cap = cv2.VideoCapture(video_path)

# 定义输出（可选）
output_path = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

# 逐帧处理
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}...")

    # YOLOv8 推理
    results = model(frame)

    for result in results:
        for box in result.boxes:
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

            # 获取框区域颜色
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                avg_color = cv2.mean(roi)[:3]  # BGR格式

            # 绘制检测框和标签
            label = f"{model.names[cls]} {conf:.2f} ({distance:.2f}m)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # 显示颜色
            color_text = f"Color: BGR({int(avg_color[0])}, {int(avg_color[1])}, {int(avg_color[2])})"
            cv2.putText(frame, color_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 写入输出视频（可选）
    out.write(frame)

    # 显示处理结果（调试用）
    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
