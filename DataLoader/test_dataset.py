from __future__ import annotations
from typing import *
from enum import Enum
import numpy as np #type: ignore
import cv2 #type: ignore
import os

from .core import Colour
from .train_dataset import TrainDataset


class TestDataset(TrainDataset):
    def __init__(self,base_path:str, line_numbered_file_list:List[Tuple[str, int]], output_color: Colour = Colour.GRAY, transform:Optional[Callable]=None):
        super().__init__(base_path, line_numbered_file_list, output_color = output_color, transform=transform)


    def __iter__(self)->TrainDataset:
        self.iter_index = 0
        return self


    def __next__(self)->np.ndarray:
        img_path:str
        if self.iter_index >= len(self.metadata_list):
            raise StopIteration
        img_path, _ = self.metadata_list[self.iter_index]
        img:np.ndarray = self.__load_img(img_path)
        self.iter_index += 1
        return img
    

    def __getitem__(self,key: Union[int, slice])->Union[List[np.ndarray],np.ndarray]:
        res = self.__getitem__Internals(key)
        if isinstance(res, tuple):
            return res[0]
        else:
            return [img for (img,_) in res]
