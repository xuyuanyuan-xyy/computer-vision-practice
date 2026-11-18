import cv2
import numpy as np
import os

base_path = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_path, "results")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

img_path = os.path.join(base_path, "building.jpg")

# 读取
img_buf = np.fromfile(img_path, dtype=np.uint8)
img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
if img is None:
    print("图片读取失败！")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯平滑：3×3，sigmaX=0 sigmaY=0，BORDER_REFLECT_101
blur = cv2.GaussianBlur(gray, (3,3), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_REFLECT_101)

# 三组阈值
groups = [
    (7, 21, "03_canny_7_21.png"),
    (25, 75, "04_canny_25_75.png"),
    (60, 180, "05_canny_60_180.png"),
]

H, W = gray.shape
total_pixel = H * W

for low, high, fname in groups:
    edge = cv2.Canny(blur, threshold1=low, threshold2=high, apertureSize=3, L2gradient=False)
    non_zero = cv2.countNonZero(edge)
    ratio = non_zero / total_pixel * 100
    print(f"阈值({low},{high}) | 非零像素：{non_zero} | 占比：{ratio:.2f}%")

    # 保存图片
    save_full = os.path.join(save_dir, fname)
    _, buf = cv2.imencode(".png", edge)
    buf.tofile(save_full)

cv2.waitKey(0)
cv2.destroyAllWindows()
