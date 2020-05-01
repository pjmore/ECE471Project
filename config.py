from __future__ import annotations
from typing import *
from DataLoader import SamplingMethod, Colour

# Path to the list of files to pull images from
# line format is:
#<Relative file path>  <class>
BaseImageListPath:str = "./dataset/filelist/places365_train_standard.txt"

#prepended to the relative filepath from the filelist file
BaseDatasetPath:str="./dataset/data"
#Number of visual words to use 
NumVisualWords: int=1500
#Number of topics to use
NumTopics: int=25
#How many neighbors are considered in KNN
NumNeighbors:int = 10
#How many categories there are in the dataset
NumCategories:int = 365
#How many images from the filelist to use for training
TrainSize:int = 1000
#how many images from the filelist to use for testing
TestSize: int = 1000
#How to sample images from the filelist
SampleMethod: SamplingMethod = SamplingMethod.Deterministic
#What colour to load the files as
ImageColour = Colour.GRAY
#Apply a transform to the image, if it is None no transform is applied
ImageTransform: Any = None
#The epsilon used to test if the EM loop should terminate
Eps = 0.000001
#The maximum number of iteratiosn that the EM loop will run
MaxIter = 1000