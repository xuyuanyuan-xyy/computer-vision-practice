import cv2 as cv
import numpy as np

def main():
    im = cv.imread('building.jpg')
    im_grey = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(im_grey, (5,5), sigmaX=1.2, sigmaY=1.2, borderType=cv.BORDER_REFLECT_101)

    sobel_x = cv.Sobel(blur, cv.CV_32F, dx=1, dy=0, ksize=3, borderType=cv.BORDER_REFLECT_101)
    sobel_y = cv.Sobel(blur, cv.CV_32F, dx=0, dy=1, ksize=3, borderType=cv.BORDER_REFLECT_101)

    scharr_x = cv.Scharr(blur, cv.CV_32F, dx=1, dy=0, borderType=cv.BORDER_REFLECT_101)
    scharr_y = cv.Scharr(blur, cv.CV_32F, dx=0, dy=1, borderType=cv.BORDER_REFLECT_101)

    lap = cv.Laplacian(blur, cv.CV_32F, ksize=3, borderType=cv.BORDER_REFLECT_101)

    cv.imwrite("t4_sobel_x.png", cv.convertScaleAbs(sobel_x))
    cv.imwrite("t4_sobel_y.png", cv.convertScaleAbs(sobel_y))
    cv.imwrite("t4_scharr_x.png", cv.convertScaleAbs(scharr_x))
    cv.imwrite("t4_scharr_y.png", cv.convertScaleAbs(scharr_y))
    cv.imwrite("t4_laplacian.png", cv.convertScaleAbs(lap))
    print("第4题运行完成")

if __name__ == "__main__":
    main()
