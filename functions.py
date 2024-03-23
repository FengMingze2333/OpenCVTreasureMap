import cv2
import numpy as np


# 通过定位点特殊的面积比来判断找到的边缘是否为定位点
def is_locating_point(con1, con2, con3):
    area1 = cv2.contourArea(con1) / 49
    area2 = cv2.contourArea(con2) / 25
    area3 = cv2.contourArea(con3) / 9
    area = min(area1, area2, area3)
    error = 0.8
    if area1 - area <= area * error and area2 - area <= area * error and area3 - area <= area * error:
        return True
    return False


# 用边缘检测寻找定位点
def find_locating_point(img):
    # cons中存储了边缘的像素， hies中存储了对应的边缘与其他边缘的层级关系
    # level用来给边缘的层级计数，寻找3层及以上边缘，并把对应边缘的像素存入cons_locating_points中
    # locations用来存储定位点中心像素
    cons, hies = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    levels = 0
    cons_locating_points = []
    locations = []

    # 遍历存储有所有边缘的层级关系的hies数组，并根据其与其他层级的关系来计数该边缘的层级
    if hies is None:
        return None
    for i in range(len(hies[0])):
        hie = hies[0][i]
        if hie[2] != -1:
            levels += 1
        elif hie[2] == -1 and levels != 0:
            levels += 1
            # 添加3层边缘及以上的像素
            if levels >= 3:
                if is_locating_point(cons[i - 2], cons[i - 1], cons[i]):
                    cons_locating_points.append(cons[i])
            levels = 0
        else:
            levels = 0

    # 求出定位点的中心点的像素
    for con_locating_point in cons_locating_points:
        x_sum, y_sum = 0, 0
        for x, y in con_locating_point[:, 0]:
            x_sum += x
            y_sum += y
        x_avg = x_sum / con_locating_point.shape[0]
        y_avg = y_sum / con_locating_point.shape[0]
        locations.append([int(x_avg), int(y_avg)])

    # 判断是否检测到四个定位点
    if len(locations) == 4:
        return locations
    else:
        return None


# 透视变换图像
def get_trans(img, locating_points):
    # 找到距离左上角和右下角最近的定位点并求出连线的直线方程系数k、b
    distance = [locating_points[0][0]+locating_points[0][1], locating_points[1][0]+locating_points[1][1],
                locating_points[2][0]+locating_points[2][1], locating_points[3][0]+locating_points[3][1]]
    argsort = np.argsort(distance)
    locating_point0 = locating_points[argsort[0]]
    locating_point3 = locating_points[argsort[3]]
    mat_x = np.array([[locating_point0[0], 1], [locating_point3[0], 1]])
    mat_y = np.array([locating_point0[1], locating_point3[1]])
    k, b = np.linalg.solve(mat_x, mat_y)

    # 找到直线上方的点即距离右上角最近的点，直线下方的点即距离左下角最近的点
    if locating_points[argsort[1]][1] - k*locating_points[argsort[1]][0] > b:
        locating_point1 = locating_points[argsort[1]]
        locating_point2 = locating_points[argsort[2]]
    else:
        locating_point1 = locating_points[argsort[2]]
        locating_point2 = locating_points[argsort[1]]

    # 将四个定位点就近透视变换到标准位置
    mat = cv2.getPerspectiveTransform(np.float32([locating_point0, locating_point1, locating_point2, locating_point3]),
                                      np.float32([(0, 0), (0, 707), (707, 0), (707, 707)]))
    img_trans = cv2.warpPerspective(img, mat, (707, 707))

    return img_trans


# 旋转图像
def get_roi(img):
    # 旋转变换矩阵
    m1 = cv2.getRotationMatrix2D((707/2, 707/2), 90, 1)
    m2 = cv2.getRotationMatrix2D((707/2, 707/2), 180, 1)
    m3 = cv2.getRotationMatrix2D((707/2, 707/2), 270, 1)

    # 对比四个角格子的HSV值来判断旋转图像
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if 100 <= img_hsv[112][595][0] <= 124 and 120 <= img_hsv[112][595][1]:
        img_spin = cv2.warpAffine(img, m1, (707, 707))
    elif 100 <= img_hsv[595][595][0] <= 124 and 120 <= img_hsv[595][595][1]:
        img_spin = cv2.warpAffine(img, m2, (707, 707))
    elif 100 <= img_hsv[595][112][0] <= 124 and 120 <= img_hsv[595][112][1]:
        img_spin = cv2.warpAffine(img, m3, (707, 707))
    else:
        img_spin = img

    return img_spin


# 用霍夫圆寻找宝藏
def find_circles(img):
    locations = []
    img_blur = cv2.medianBlur(img, 3)
    hough_circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT_ALT, 5, 20,
                                     param1=1000, param2=0.70, minRadius=10, maxRadius=20)
    if hough_circles is None:
        return None
    for hough_circle in hough_circles[0, :]:
        locations.append((int(hough_circle[0]), int(hough_circle[1])))
    return locations
