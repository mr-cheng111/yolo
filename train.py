from ultralytics import YOLO

# 加载预训练模型（推荐yolov8n.pt）
model = YOLO("./models/yolo11n.pt")

# 训练配置（GPU加速）
model.train(
    data="./datasets/data.yaml",
    epochs=256,
    imgsz=(640, 480),
    batch=16,  # 根据GPU显存调整
    device=0,  # 使用GPU
    project="cube_detection",
    name="train_v3"
)

# 导出模型为 ncnn 格式
model.export(format="ncnn", path="./models/yolo11n.ncnn")

