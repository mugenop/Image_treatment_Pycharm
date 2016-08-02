__author__ = 'Adel'
from scipy.optimize import fsolve
import math
import numpy as np
from cv2 import *
EDGE = 20
W = 1920
L= 1080

DISTANCE = 50
def make_squares():
    img = np.zeros((L, W), np.uint8)
    y = 0
    white = 0
    black = 0
    img [img == 0] = 255
    while y< L:
        y = y + DISTANCE
        x = 0
        while x < W:
            x = x + DISTANCE
            rectangle(img, (x, y), (x+EDGE, y+EDGE), black, cv.CV_FILLED)
    return img




if __name__ == '__main__':
    img = make_squares()
    imshow('image', img)
    imwrite("grid.png", img)
    0xFF & waitKey()
    destroyAllWindows()