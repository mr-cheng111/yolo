from ultralytics import YOLO

# 加载预训练模型（推荐yolov8n-pose.pt）
model = YOLO("./models/yolov8n.pt")

# 训练配置（GPU加速）
model.train(
    data="./datasets/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,  # 根据GPU显存调整
    device=0,  # 使用GPU
    project="cube_detection",
    name="train_v1"
)