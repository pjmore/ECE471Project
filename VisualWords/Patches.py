from __future__ import annotations
from typing import Tuple
import VisualWords.core #type: ignore
from VisualWords.core import ImageFeatures, FeatureExtractor
import numpy as np #type: ignore
import cv2 #type: ignore
import math
from itertools import count
import timeit


def MakeGrayPatchExtractor(size:int=5, step:int=5)->FeatureExtractor:
    #Check for valid step and size parameters
    if step % 2 == 0:
        raise ValueError(f"The step parameter to Visualwords.Patches.grey_patches must be odd, got {step}")
    if size % 2 == 0:
        raise ValueError(f"The size parameter to Visualwords.Patches.grey_patches must be odd, got {size}")
    #Closure for extracting gray patches
    def GrayPatchExtractor(img:np.ndarray)->ImageFeatures:
        numFeatures, sizeFeature, raw_feature = __patch_extract(img, size=size, step=step)
        return ImageFeatures(sizeFeature, numFeatures, raw_feature)
    return GrayPatchExtractor
     


def MakeColourPatchExtractor(size:int=5, step:int=5)->FeatureExtractor:
    if step % 2 == 0:
        raise ValueError(f"The step parameter to Visualwords.Patches.colour_patches must be odd, got {step}")
    if size % 2 == 0:
        raise ValueError(f"The size parameter to Visualwords.Patches.colour_patches must be odd, got {size}")
    def ColourPatchExtractor(img:np.ndarray)->ImageFeatures:
        if len(img.shape) == 2:
            raise ValueError(f"Must only pass colour images to colour image extractor")
        numFeatures, sizeFeature, raw_feature = __patch_extract(img, size=size, step=step, feature_size_multiple=3)
        return ImageFeatures(sizeFeature, numFeatures, raw_feature)
    return ColourPatchExtractor



def __patch_extract(img: np.ndarray, size=5, step=5, feature_size_multiple=1)->Tuple[int, int, np.ndarray]:
    #print(f"The shape of the image is {img.shape}")
    #print(f"kernel size:{size}\nstep size:{step}\nfeature size multiple:{feature_size_multiple}")
    
    # Number of indexes on either side of center
    N:int = int((size-1)/2)
    
    #width and height
    w:int
    h:int
    w,h = img.shape[0:2]
    num_row_iter:int = math.ceil((w-2*N)/step)
    num_col_iter:int = math.ceil((h-2*N)/step)
    #print(f"There will be {num_row_iter} row iterations and {num_col_iter} column iterations")
    
    numFeatures:int = num_row_iter*num_col_iter
    sizeFeature:int = size*size*feature_size_multiple
    features = np.zeros((numFeatures, sizeFeature))
    #print(f"There will be {numFeatures} features\nEach feature will be {sizeFeature} long")

    f_idx = 0
    for y in range(N, h-N, step):
        l_y = y-N
        u_y = y+N+1 
        for x in range(N, w-N, step):
            l_x = x-N       
            u_x = x+N+1      
            #print(f"(x,y):{x},{y}")
            #print(f"{l_x}->{u_x} - {l_y}->{u_y}")
            #print(f"The shape of the new feature: {img[x-N:x+N+1, y-N:y+N+1].shape}")
            features[f_idx,:] = img[x-N:x+N+1, y-N:y+N+1].copy().flatten()
            f_idx += 1
    return (numFeatures, sizeFeature, features)
