import cv2
import numpy as np
import os

base_path = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_path, "results")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 读取两张图
img1_path = os.path.join(base_path, "box.png")
img2_path = os.path.join(base_path, "box_in_scene.png")

def read_img(path):
    buf = np.fromfile(path, dtype=np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if im is None:
        print(f"图片 {path} 读取失败！")
        exit()
    return im

img_query = read_img(img1_path)
img_scene = read_img(img2_path)

# SIFT 默认参数创建
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img_query, None)
kp2, des2 = sift.detectAndCompute(img_scene, None)

print("==== SIFT特征信息 ====")
print(f"box.png 关键点数量: {len(kp1)}")
print(f"box.png 描述子 shape:{des1.shape}, dtype:{des1.dtype}")
print(f"box_in_scene.png 关键点数量: {len(kp2)}")
print(f"box_in_scene.png 描述子 shape:{des2.shape}, dtype:{des2.dtype}")

# 绘制关键点图
kp_img1 = cv2.drawKeypoints(img_query, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kp_img2 = cv2.drawKeypoints(img_scene, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 保存关键点图
def save_img(out_img, name):
    full = os.path.join(save_dir, name)
    _, b = cv2.imencode(".png", out_img)
    b.tofile(full)

save_img(kp_img1, "06_keypoints_query.png")
save_img(kp_img2, "07_keypoints_scene.png")

# BF匹配器，NORM_L2，crossCheck=False
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
raw_matches = bf.knnMatch(des1, des2, k=2)

# 只保留有两个近邻的完整候选C
candidates_C = [pair for pair in raw_matches if len(pair) == 2]
C = len(candidates_C)
print(f"\n完整候选C（拥有2个近邻）数量：{C}")

r_list = [
    (0.60, "08_matches_060.png"),
    (0.75, "09_matches_075.png")
]

for r, fname in r_list:
    good = []
    for m, n in candidates_C:
        if m.distance < r * n.distance:
            good.append(m)
    G = len(good)
    if C == 0:
        ratio_percent = "不适用"
    else:
        ratio_percent = f"{G / C * 100:.2f}%"
    print(f"r={r:.2f} | 筛选保留G={G}, 筛选保留比例G/C={ratio_percent}")

    # 绘图
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    match_img = cv2.drawMatchesKnn(img_query, kp1, img_scene, kp2, [[g] for g in good], None,** draw_params)
    save_img(match_img, fname)

cv2.waitKey(0)
cv2.destroyAllWindows()
