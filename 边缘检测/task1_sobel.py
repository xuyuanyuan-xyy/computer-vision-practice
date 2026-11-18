import cv2
import numpy as np
import os

base_path = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_path, "results")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

img_path = os.path.join(base_path, "blox.jpg")

# 兼容中文路径读取
try:
    img_buf = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
except Exception as e:
    img = None

if img is None:
    print(f"❌ 读取失败！路径：{img_path}")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ksize = 3
scale = 1
delta = 0
ddepth = cv2.CV_16S
border = cv2.BORDER_REFLECT_101

sobel_x = cv2.Sobel(gray, ddepth, dx=1, dy=0, ksize=ksize, scale=scale, delta=delta, borderType=border)
sobel_y = cv2.Sobel(gray, ddepth, dx=0, dy=1, ksize=ksize, scale=scale, delta=delta, borderType=border)

min_x, max_x = cv2.minMaxLoc(sobel_x)[0], cv2.minMaxLoc(sobel_x)[1]
min_y, max_y = cv2.minMaxLoc(sobel_y)[0], cv2.minMaxLoc(sobel_y)[1]
dtype_x = sobel_x.dtype
dtype_y = sobel_y.dtype

abs_x = cv2.convertScaleAbs(sobel_x)
abs_y = cv2.convertScaleAbs(sobel_y)


save_x = os.path.join(save_dir, "01_sobel_x.png")
save_y = os.path.join(save_dir, "02_sobel_y.png")

_, buf_x = cv2.imencode(".png", abs_x)
buf_x.tofile(save_x)

_, buf_y = cv2.imencode(".png", abs_y)
buf_y.tofile(save_y)

print("==== Sobel X方向 dx=1 dy=0 ====")
print(f"数据类型: {dtype_x}, 最小值: {min_x:.2f}, 最大值: {max_x:.2f}")
print("==== Sobel Y方向 dx=0 dy=1 ====")
print(f"数据类型: {dtype_y}, 最小值: {min_y:.2f}, 最大值: {max_y:.2f}")

cv2.waitKey(0)
cv2.destroyAllWindows()
