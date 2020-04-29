from __future__ import annotations
from typing import *
from enum import Enum
import re
import numpy as np #type: ignore
import cv2  #type: ignore
import os

class Colour(Enum):
    RGB = 0
    HSV = 1
    GRAY = 2
    BGR = 3


class SamplingMethod(Enum):
    PsuedoRandom = 0
    Random = 1
    Deterministic = 2 


  
file_list_line_extraction_regex = re.compile(r'^(?P<RelativePath>.*?\d+\.jpg)\s*(?P<ImageClass>\d+)$')


class MalformedSpecFileError(Exception):
    def __init__(self, line:str, line_number:int, message:str):
        #line that caused the failure
        self.line:str = line
        #The line number of the problem line
        self.line_number: int = line_number
        #User defined message which should give more context to error
        self.message:str = message
    
    def __str__(self)->str:
        return f"There was an error reading line {self.line_number}, which contained '{self.line}', in the specfile: {self.message}"




def PrepareFilterEnums(highest_enum:int)->Callable[[Tuple[str,int]],bool]:
    def FilterHigherEnums(numbered_line: Tuple[str,int])->bool:
        m  = file_list_line_extraction_regex.match(numbered_line[0])
        if m is None:
            raise MalformedSpecFileError(numbered_line[0], numbered_line[1], "line did not not matche the metadata extraction regex")
        img_class: int = int(m.group('ImageClass'))
        if img_class > highest_enum:
            return False
        return True
    return FilterHigherEnums



def ExtractImageMetaDataFromLines(file_lines:List[Tuple[str, int]])->List[Tuple[str, int]]:
    metadata_list: List[Tuple[str, int]] = []
    for line, line_number in file_lines:
        m  = file_list_line_extraction_regex.match(line)
        if m is None:
            raise MalformedSpecFileError(line, line_number, "line did not not matche the metadata extraction regex")
        rel_path:str = m.group('RelativePath')
        img_class: int = int(m.group('ImageClass'))
        metadata_list.append((rel_path, img_class))
    return metadata_list
        



    
    



class OpenImageError(Exception):
    def __init__(self, path:str):
        #Path of the image that failed to open
        self.path: str = path
        #User message
        self.message: str = "The file could not be opened by cv2 properly for an unknown reason because cv2 doesn't have the best error messsages. Please check that it is a valid jpg file."
        if os.path.exists(path):
            if os.path.isfile(path):
                if not os.access(path, os.R_OK):
                    self.message = "The file exists but you dont have the proper permissions to open it."
                else:
                    self.message = "The file exists but could not be opened."
            elif os.path.isdir(path):
                self.message = "The path exists but it is a directory"
        else:
            self.message = "The path does not exist"

    def __str__(self)->str:
        return f"There was an error trying to open {self.path}: {self.message}"










