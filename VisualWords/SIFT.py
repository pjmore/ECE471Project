from __future__ import annotations
import core #type: ignore
from core import ImageFeatures, FeatureExtractor
import numpy as np #type: ignore
import cv2 #type: ignore
import math
from itertools import count


def MakeGraySIFTExtractor(step:int=5, size:int=5)->FeatureExtractor:
    def GraySIFTExtractor(img:np.ndarray)->ImageFeatures:
        return gray_SIFT(img, size=size, step=step)
    return GraySIFTExtractor
    


def MakeColourSIFTExtractor(step:int=5, size:int=5)->FeatureExtractor:
    def ColourSIFTExtractor(img:np.ndarray)->ImageFeatures:
        return colour_SIFT(img, size=size, step=step)
    return ColourSIFTExtractor

def MakeSparseGraySIFTExtractor(step:int, size:int=5)->FeatureExtractor:
    def SparseGraySIFTExtractor(img:np.ndarray)->ImageFeatures:
        return sparse_gray_SIFT(img)
    return SparseGraySIFTExtractor




def gray_SIFT(img: np.ndarray, step:int=5, size:int=5)->ImageFeatures:
    return ImageFeatures(1,1,np.array([0]))


def colour_SIFT(img: np.ndarray, step:int=5, size:int=5)->ImageFeatures:
    return ImageFeatures(1,1,np.array([0]))

# see references , [7, 14, 15].    
def sparse_gray_SIFT(img: np.ndarray, step:int=5, size:int=5)->ImageFeatures:
    return ImageFeatures(1,1,np.array([0]))
