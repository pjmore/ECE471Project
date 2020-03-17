from __future__ import annotations
from typing import Tuple
import core #type: ignore
from core import ImageFeatures
import numpy as np #type: ignore
import cv2 #type: ignore
import math
import numba  #type: ignore
from numba import prange
from itertools import count
import timeit


def grey_patches(img: np.ndarray, step:int=5, spacing:int=5)->ImageFeatures:
    if step % 2 == 0:
        raise ValueError(f"The step parameter to Visualwords.Patches.grey_patches must be odd, got {step}")
    if spacing % 2 == 0:
        raise ValueError(f"The spacing parameter to Visualwords.Patches.grey_patches must be odd, got {spacing}")
    if len(img.shape) == 3:
        grey_img:np.ndarray = core.toGrayscale(img)
    else:
        grey_img = img
    return __get_patch_feature(grey_img, kernel_region_size=step, size=spacing)



def __patch_extract_par(img: np.ndarray, kernel_region_size=5, size=5, feature_size_multiple=1)->Tuple[int, int, np.ndarray]:
    # Number of indexes on either side of center
    N:int = int((size-1)/2)
    # Number of points to step forward
    step: int = int((kernel_region_size-1)/1) + 1
    #width and height
    w:int
    h:int
    w,h = img.shape[0:2]
    num_row_iter:int = math.ceil((w-2*N)/(step))
    num_col_iter:int = math.ceil((h-2*N)/step)
    numFeatures:int = num_row_iter*num_col_iter
    sizeFeature:int = size*size*feature_size_multiple
    features = np.zeros((numFeatures, sizeFeature))
    for i in range(num_row_iter):
        for j in range(num_col_iter):
            row_idx = N*i
            col_idx = N*j
            cidx = int((col_idx/step) - 2)
            ridx = int((row_idx/step)-2)
            f_idx = ridx*(num_col_iter -1) + cidx
            #print(f"Indexing from {row_idx-N}:{row_idx+N+1} and {col_idx-N}:{col_idx+N+1}")
            features[f_idx,:] = img[row_idx-N:row_idx+N+1, col_idx-N:col_idx+N+1].copy().flatten()

    return (numFeatures, sizeFeature, features)



def __patch2feature(numFeatures:int, sizeFeature:int, raw_features:np.ndarray)->ImageFeatures:
    features: ImageFeatures = ImageFeatures(sizeFeature, numFeatures, raw_features)
    return features

    


def __get_patch_feature(img: np.ndarray, kernel_region_size=5, size=5, feature_size_multiple=1)->ImageFeatures:
    numFeatures, sizeFeature, raw_feature = __patch_extract_par(img, kernel_region_size=kernel_region_size, size=size, feature_size_multiple=feature_size_multiple)
    return __patch2feature(numFeatures, sizeFeature, raw_feature)


def colour_patches(img: np.ndarray, step:int=5, spacing:int=5)->ImageFeatures:
    if step % 2 == 0:
        raise ValueError(f"The step parameter to Visualwords.Patches.colour_patches must be odd, got {step}")
    if spacing % 2 == 0:
        raise ValueError(f"The spacing parameter to Visualwords.Patches.colour_patches must be odd, got {spacing}")
    return __get_patch_feature(img, kernel_region_size=step, size=spacing, feature_size_multiple=3)

def run():
    test = cv2.imread("ct.jpg", cv2.IMREAD_UNCHANGED)
    run_func = lambda : colour_patches(test, step=1, spacing=5)
    num_runs = 1000
    total_time = timeit.timeit(run_func, number=num_runs)
    print(f"Each run of the function took {total_time}")
    #print(len(features))
    #for f in features:
    #   print(f)

if __name__ == "__main__":
    run()