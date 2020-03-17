from __future__ import annotations
import core #type: ignore
from core import ImageFeatures
import numpy as np #type: ignore
import cv2 #type: ignore
import math
from itertools import count



def gray_SIFT(img: np.ndarray, step:int=5, spacing:int=5)->ImageFeatures:
    kp = []
    for y in range(0, img.shape[1], spacing):
        for x in range(0, img.shape[0], spacing):
          kp.append(cv2.KeyPoint(x,y, _octave=0,_size=4 ))  
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.compute(img, kp)
    print(kp)
    print(des)


def __grey_SIFT(img: np.ndarray, step:int=5, spacing:int=5)->ImageFeatures:
    pass

if __name__ == "__main__":
    tImg = cv2.imread("thin_gray_scale.jpg",  cv2.COLOR_BGR2GRAY)

    gray_SIFT(tImg, step=5, spacing=5)
