import cv2 as cv
import numpy as np

def main():
    # 输入图像
    im = cv.imread('building.jpg')
    im_grey = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    # 预处理参数：5×5高斯，σ=1.2
    blur = cv.GaussianBlur(im_grey, (5,5), sigmaX=1.2, sigmaY=1.2, borderType=cv.BORDER_REFLECT_101)

    # ========== 1. Sobel 梯度 ==========
    sobel_x = cv.Sobel(blur, cv.CV_32F, dx=1, dy=0, ksize=3, borderType=cv.BORDER_REFLECT_101)
    sobel_y = cv.Sobel(blur, cv.CV_32F, dx=0, dy=1, ksize=3, borderType=cv.BORDER_REFLECT_101)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # ========== 2. Scharr 梯度 ==========
    scharr_x = cv.Scharr(blur, cv.CV_32F, dx=1, dy=0, borderType=cv.BORDER_REFLECT_101)
    scharr_y = cv.Scharr(blur, cv.CV_32F, dx=0, dy=1, borderType=cv.BORDER_REFLECT_101)
    scharr_mag = np.sqrt(scharr_x**2 + scharr_y**2)

    # ========== 3. Laplacian ==========
    lap = cv.Laplacian(blur, cv.CV_32F, ksize=3, borderType=cv.BORDER_REFLECT_101)
    lap_abs = np.abs(lap)

    # -------- 保存原始带符号浮点 .npy --------
    np.save("sobel_x.npy", sobel_x)
    np.save("sobel_y.npy", sobel_y)
    np.save("scharr_x.npy", scharr_x)
    np.save("scharr_y.npy", scharr_y)
    np.save("laplacian.npy", lap)

    # -------- 打印最值 --------
    print("==== Sobel_x  min, max：", sobel_x.min(), sobel_x.max())
    print("==== Sobel_y  min, max：", sobel_y.min(), sobel_y.max())
    print("==== Scharr_x min, max：", scharr_x.min(), scharr_x.max())
    print("==== Scharr_y min, max：", scharr_y.min(), scharr_y.max())
    print("==== Laplacian min, max：", lap.min(), lap.max())

    # 归一化函数：缩放到0~255，用于可视化
    def norm255(src):
        dst = cv.normalize(src, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        return dst

    # 保存幅值可视化图片
    cv.imwrite("t5_sobel_mag.png", norm255(sobel_mag))
    cv.imwrite("t5_scharr_mag.png", norm255(scharr_mag))
    cv.imwrite("t5_lap_abs.png", norm255(lap_abs))
    print("第5题运行完成，npy文件和图像已保存")

if __name__ == "__main__":
    main()
