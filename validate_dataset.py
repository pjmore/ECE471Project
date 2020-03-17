from __future__ import annotations
from typing import *
import os
import sys
from DataLoader import ImageClass, max_enum_idx, prettystr
import cv2 #type: ignore
import time
import multiprocessing
import tqdm #type:ignore



root_dataset_file = "./dataset"

class StringIndex(NamedTuple):
    RelPath: str
    Class: ImageClass

class Result(NamedTuple):
    BadFile: bool
    Class: ImageClass
    Name: str




def is_bad_file(idx: StringIndex)->Result:
    img_path = os.path.join(root_dataset_file,"data", idx.RelPath[1:])
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    isBadFile = img is None
    return Result(isBadFile, idx.Class, idx.RelPath)






def line_to_idx(line:str, line_number: int)->StringIndex:
    s_line = line.strip().split()
    if len(s_line) != 2:
        print(f"Found a line split something besides two items on line[{line_number}]: '{s_line}'")
        sys.exit(1)
    str_index = StringIndex(s_line[0], ImageClass(int(s_line[1])))
    return str_index



if __name__ == "__main__":
    specFile_path = os.path.join(root_dataset_file, "filelist","places365_train_standard.txt")
    with open(os.path.join(root_dataset_file, "filelist","places365_train_standard.txt")) as f:
        all_lines: List[str] = f.readlines()
    input_list = [line_to_idx(x, line_no) for line_no, x in enumerate(all_lines)]
    error_count = [0]*(max_enum_idx() + 1)
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        results= list( tqdm.tqdm( p.imap_unordered( is_bad_file, input_list ), total=len(all_lines) ) )

    results.sort(key=lambda x: int(x.Class))
    curr_class:int = 0
    print(f"In the class {prettystr(ImageClass(curr_class))} the following images were unable to be opened:")
    for res in results:
        if curr_class != int(res.Class):
            curr_class+=1
            if curr_class > max_enum_idx():
                break
            print(f"The class {prettystr(ImageClass(curr_class))} had {error_count[curr_class]} errors:")
        if res.BadFile:
            print(f"\t{res.Name}")
    #with open("./BrokenImageList.txt", 'w') as w:
    #    w.wri

    
    