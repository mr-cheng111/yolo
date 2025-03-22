import cv2
import numpy as np

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

def main():
    # 打开默认摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    img_counter = 0

    # 获取相机参数
    camera_matrix = camera_params["camera_matrix"]
    dist_coeffs = camera_params["distortion_coefficients"]
    rectification_matrix = camera_params["rectification_matrix"]
    projection_matrix = camera_params["projection_matrix"]

    # 计算矫正映射
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, rectification_matrix, projection_matrix, 
                                             (camera_params["image_width"], camera_params["image_height"]), cv2.CV_16SC2)

    while True:
        # 读取一帧
        ret, frame = cap.read()

        if not ret:
            print("无法接收帧，退出")
            break

        # 进行图像矫正
        undistorted_frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        # 显示矫正后的帧
        cv2.imshow('Undistorted Camera', undistorted_frame)

        key = cv2.waitKey(1) & 0xFF

        # 按下'q'键退出
        if key == ord('q'):
            break
        # 按下's'键拍照
        elif key == ord('s'):
            img_name = f"photo_{img_counter}.png"
            cv2.imwrite(img_name, undistorted_frame)
            print(f"{img_name} 已保存")
            img_counter += 1

    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()