import numpy as np
import cv2


# Beyond the filter. See how you want
def prep_image(img):
    temp = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # off color
    temp = cv2.GaussianBlur(temp, (3, 3), 0)  # blur
    temp_canny = cv2.Canny(temp, 30, 50)
    return temp, temp_canny


def lines(img):
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    edges = prep_image(img)[1]

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    return img


def boxes(img):
    hsv_min = np.array((0, 54, 5), np.uint8)
    hsv_max = np.array((187, 255, 253), np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # меняем цветовую модель с BGR на HSV
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)  # применяем цветовой фильтр
    contours0, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # перебираем все найденные контуры в цикле
    for cnt in contours0:
        rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат
        temp = box.tolist()
        t1 = (box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2
        t2 = (box[1][0] - box[2][0])**2 + (box[1][1] - box[2][1])**2
        t3 = (box[2][0] - box[3][0])**2 + (box[2][1] - box[3][1])**2
        t4 = (box[3][0] - box[0][0])**2 + (box[3][1] - box[0][1])**2
        prec = 50
        if t1 < prec**2 or t2 < prec**2 or t3 < prec**2 or t4 < prec**2:
            continue
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)  # рисуем прямоугольник

    # cv2.imshow('contours', img)  # вывод обработанного кадра в окно
    # img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
    # img = stretch_near = cv2.resize(img, (780, 540), interpolation=cv2.INTER_LINEAR)
    # img = cv2.resize(img, (1200, 800))
    return img
