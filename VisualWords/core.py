from __future__ import annotations
from typing import *
import numpy as np #type: ignore
#from recordclass import RecordClass #type: ignore
#from recordclass.typing import RecordClass #type: ignore
from enum import IntEnum, auto


class RGBOrder(IntEnum):
    OpenCV = auto()
    RGB = auto()

def toGrayscale(img:np.ndarray, img_format:RGBOrder= RGBOrder.OpenCV)->np.ndarray:
    out:np.ndarray = np.zeros_like(img)
    if img_format == RGBOrder.OpenCV:
        out = img[:,:,0]*0.1140 + img[:,:,1]*0.2989 + img[:,:,0]*0.5870
    elif img_format == RGBOrder.RGB:
        out = img[:,:,0]*0.2989 + img[:,:,1]*0.1440 + img[:,:,0]*0.5870
    return out




#Features are always arranged row major
#This way accessing a single feature is faster
class ImageFeatures: #type: ignore
    __slots__=['FeatureDim', 'NumFeatureVecs','Features','IterIndex']
    def __init__(self, FeatureDim:int, NumFeatureVecs: int, Features:np.ndarray):
        self.FeatureDim: int = FeatureDim
        self.NumFeatureVecs: int = NumFeatureVecs
        self.Features: np.ndarray = Features 
        #data_slices: List[slice]
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
        if isinstance(key, slice):
            res = self.Features[key,:]
            numVecs = res.shape[0]
            featureDim = self.FeatureDim
            return_val: ImageFeatures =  ImageFeatures(featureDim, numVecs, res)
            return return_val
        elif isinstance(key, int):
            return_val =  ImageFeatures(self.FeatureDim, 1, self.Features[key,:])
            return return_val
        else:
            raise KeyError("ImageFeatures only accepts integers and slices as subscripts")

    def __setitem__(self, key, item):
        if isinstance(key, slice):
            self.Features[key, :] = item
        elif isinstance(key, int):
            self.Features[key, :] = item
        else:
            raise KeyError("ImageFeatures only accepts integers and slices as subscripts when setting values")
       

    def __str__(self)->str:
        return f"ImageClass(FeatureDim={self.FeatureDim}, NumFeatureVecs={self.NumFeatureVecs}, IterIndex={self.IterIndex}, Features={self.Features})"

tcount = 0
def t(item: ImageFeatures)->None:
    global tcount
    print(f"examining {tcount}th test")
    for feature in item:
        print(f"\t{feature}")
    tcount += 1

if __name__ == "__main__":
    test = np.eye(6)
    num_features:int = test.shape[0]
    featureDim:int = test.shape[1]
    t_if = ImageFeatures(featureDim, num_features, test)

    t(t_if[0])
    t(t_if[0:4:2])
    test_subslicing = t_if[0::1]
    t(t_if[-1])