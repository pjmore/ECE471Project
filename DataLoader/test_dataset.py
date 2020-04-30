from __future__ import annotations
from typing import *
from enum import Enum
import numpy as np #type: ignore
import cv2 #type: ignore
import os

from .core import Colour
from .train_dataset import TrainDataset, OpenImageError


class TestDataset(TrainDataset):
    def __init__(self,base_path:str, line_numbered_file_list:List[Tuple[str, int]], output_color: Colour = Colour.GRAY, transform:Optional[Callable]=None):
        super().__init__(base_path, line_numbered_file_list, output_color = output_color, transform=transform)


    def __iter__(self)->TrainDataset:
        self.iter_index = 0
        return self


    def __next__(self)->Tuple[np.ndarray,int]:
        img_path:str
        if self.iter_index >= len(self.metadata_list):
            raise StopIteration
        img_path, label = self.metadata_list[self.iter_index]
        img:np.ndarray = self.__load_img(img_path)
        self.iter_index += 1
        return img, label
    

    def __getitem__(self,key: Union[int, slice])->Union[List[np.ndarray],np.ndarray]:
        res = self.__getitem__Internals(key)
        if isinstance(res, tuple):
            return res[0],res[1]
        else:
            return [(img, label) for (img,label) in res]

    
    def __load_img(self, img_rel_path:str)-> np.ndarray:
        img_path_start_idx = 0
        if img_rel_path[0] == "/" or  img_rel_path[0] == "\\":
            img_path_start_idx = 1
        img_path = os.path.join(self.base_path, img_rel_path[img_path_start_idx:])
        cv2_output_color = cv2.IMREAD_COLOR
        if self.output_color == Colour.GRAY:
            cv2_output_color = cv2.IMREAD_GRAYSCALE
        img = cv2.imread(img_path, cv2_output_color)
        if img is None:
            raise OpenImageError(img_path)
        if self.output_color == Colour.HSV:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.output_color == Colour.RGB:
            buffer = np.zeros((img.shape[0], img.shape[1]))

            #put the blue layer in the buffer
            buffer = img[:,:,0]

            #overwrite the blue layer in the image with the red layer
            img[:,:,0] = img[:,:,2]
            #overwrite the redlayer in the buffer
            img[:,:,2] = buffer 
        elif self.output_color == Colour.BGR or self.output_color == Colour.GRAY:
            pass
        else:
            raise ValueError(f"Did not recognize the color code {self.output_color}")
        if self.transform is not None:
            img = self.transform(img)
        return img
