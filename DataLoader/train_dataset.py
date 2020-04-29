from __future__ import annotations
from typing import *
from enum import Enum
import numpy as np #type:ignore
import cv2  #type: ignore
import os

from .core import SamplingMethod, Colour, OpenImageError, ExtractImageMetaDataFromLines


class TrainDataset:
    def __init__(self, base_path:str, line_numbered_file_list:List[Tuple[str, int]], output_color: Colour = Colour.GRAY, transform:Optional[Callable]=None):
        #Image property properties
        self.transform:Optional[Callable] = transform
        self.output_color: Colour = output_color
        
        #iterator properties
        self.iter_index: int = 0

        #core private properties
        self.base_path: str = base_path
        self.metadata_list = ExtractImageMetaDataFromLines(line_numbered_file_list)

        
        

    def __iter__(self)->TrainDataset:
        self.iter_index = 0
        return self

    def __len__(self)->int:
        return len(self.metadata_list)

    def __next__(self)->Tuple[np.ndarray,int]:
        img_path: str
        img_class: int
        if self.iter_index >= len(self.metadata_list):
            raise StopIteration
        img_path, img_class = self.metadata_list[self.iter_index]
        img = self.__load_img(img_path)
        self.iter_index +=1
        return img, img_class


    def __getitem__(self,key: Union[int, slice])->Union[List[np.ndarray],np.ndarray]:
        return self.__getitem__Internals(key)


    def __getitem__Internals(self, key: Union[int, slice])->Union[List[Tuple[np.ndarray,int]],Tuple[np.ndarray, int]]:
        if isinstance(key, slice):
            indexed_metadata_list: List[Tuple[str, int]] = self.metadata_list[key]
            return [(self.__load_img(img_path), img_class) for (img_path, img_class) in indexed_metadata_list]
        elif isinstance(key, int):
            img_path, img_class = self.metadata_list[key]
            return self.__load_img(img_path), img_class
        else:
            raise KeyError("Only integers and slices may be used as keys")


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


