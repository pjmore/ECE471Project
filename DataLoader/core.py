from __future__ import annotations
from typing import *
from enum import Enum


class Colour(Enum]:
    RGB = 0
    HSV = 1
    GRAY = 2


class SamplingMethod(Enum):
    PsuedoRandom = 0
    Random = 1
    Deterministic = 2   




class ImageListDataset:
    def __init__(self, output_color: Colour = Colour.GRAY, transform:Optional[Callable]=None):
        self.transform:Optional[Callable] = transform
        self.output_color: Colour = output_color
        self.image_list: List[np.ndarray] = []
        self.class_list: List[int] = []
        self.iter_index: int = 0


    def __iter__(self)->ImageListDataset:
        self.iter_index
        return self

    def __len__(self)->int:
        return len(self.image_list)

    def __next__(self)->ImageListDataset:
        self.iter_index = 0
        return self

    def iteritems(self)->TrainDataset:
        return 


        