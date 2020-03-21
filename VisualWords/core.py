from __future__ import annotations
from typing import *
import numpy as np #type: ignore
from enum import IntEnum, auto


FeatureExtractor = Callable[[np.ndarray], ImageFeatures]

#Features are always arranged row major
#This way accessing a single feature is faster
class ImageFeatures: #type: ignore
    __slots__=['FeatureDim', 'NumFeatureVecs','Features','IterIndex']
    def __init__(self, FeatureDim:int, NumFeatureVecs: int, Features:np.ndarray):
        self.FeatureDim: int = FeatureDim
        self.NumFeatureVecs: int = NumFeatureVecs
        self.Features: np.ndarray = Features 
        self.IterIndex:int=0

    #Resets the iteration index of the feature vector
    def __iter__(self)->ImageFeatures:
        self.IterIndex = 0
        return self

    #Returns the next feature vector
    def __next__(self)->np.ndarray:
        if self.IterIndex >= self.NumFeatureVecs:
            raise StopIteration
        if len(self.Features.shape) <= 1:
            self.IterIndex+=1
            return self.Features
        res = self.Features[self.IterIndex,:]
        self.IterIndex+=1
        return res


    #Returns how many feature vectors are contained 
    def __len__(self)->int:
        return self.NumFeatureVecs
    

    #Allows indexing into the features, always returns a ImageFeatures type
    def __getitem__(self, key: Union[int, slice])->np.ndarray:
        if not(isinstance(key, slice) or isinstance(key, int)):
            raise KeyError("ImageFeatures only accepts integers and slices as subscripts")
        features = self.Features[key,:]
        return ImageFeatures(self.FeatureDim, features.shape[0], features)

            

    def __setitem__(self, key, item):
        if not(isinstance(key, slice) or isinstance(key, int)):
            raise KeyError("ImageFeatures only accepts integers and slices as subscripts when setting values")
        self.Features[key, :] = item


    def __str__(self)->str:
        return f"ImageClass(FeatureDim={self.FeatureDim}, NumFeatureVecs={self.NumFeatureVecs}, IterIndex={self.IterIndex}, Features={self.Features})"

