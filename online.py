import cv2
import numpy as np
import math
from main import prep_image, lines, boxes


if __name__ == '__main__':
    cv2.namedWindow("online")
    cap = cv2.VideoCapture(0)

    last_img = prep_image(np.zeros((480, 640, 3), dtype=np.uint8))[1]
    mode = 1
    modes = {0: "undefined", 1: "normal", 2: "gray", 3: "canny", 4: "lines", 5: "boxes"}
    while True:
        flag, img = cap.read()
        img = cv2.flip(img, 1)

        if mode == 1:
            pass
        elif mode == 2:
            img = prep_image(img)[0]
        elif mode == 3:
            img = prep_image(img)[1]
        elif mode == 4:
            img = lines(img)
        elif mode == 5:
            img = boxes(img)
        else:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        if mode in modes:
            cv2.putText(img, modes[mode], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, 2)
        else:
            cv2.putText(img, modes[0], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, 2)

        try:
            cv2.imshow('online', img)
        except IndexError:  # TODO
            cap.release()
            raise
        ch = cv2.waitKey(5)
        if ch == 27:
            break
        elif 49 <= ch <= 57:
            mode = ch - 48

    cap.release()
    cv2.destroyAllWindows()
