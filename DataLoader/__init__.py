from __future__ import annotations
from typing import *

from itertools import count

from DataLoader.core import Colour, SamplingMethod, MalformedSpecFileError, OpenImageError, ExtractImageMetaDataFromLines, PrepareFilterEnums
from DataLoader.test_dataset import TestDataset
from DataLoader.train_dataset import TrainDataset

from random import uniform, shuffle
import os

def LoadDatasets(filelist_path:str, base_path:str, training_set_size: int, testing_set_size: int, num_classes:int, sampling_method: SamplingMethod = SamplingMethod.Random ,output_color: Colour = Colour.GRAY, transform:Optional[Callable]=None)->Tuple[TrainDataset, TestDataset]:
    with open(filelist_path,'r') as f:
        line_count = count(start=0, step=1)
        numbered_lines = [(line,line_count.__next__()) for line in f.readlines()]
    max_enum: int = num_classes -1
    num_available_images = len(numbered_lines)

    if training_set_size + testing_set_size > num_available_images:
        raise ValueError("The combined size of the test and training sets was larger than the total size of the dataset")


    filter_higher_enums = PrepareFilterEnums(max_enum)

    numbered_lines = list(filter(filter_higher_enums, numbered_lines))





    training_line_list : List[Tuple[str, int]] = []
    testing_line_list : List[Tuple[str,int]] = []

    training_metadata_list: List[Tuple[str, int]] = []
    testing_metadata_list: List[Tuple[str, int]] = []


    #flags for full dataset
    test_all_data: bool = False
    train_all_data: bool = False
    #marks if done iterating over file lines
    done_fetching: bool = False

    

    if sampling_method == SamplingMethod.Random:
        #Randomly shuffle the lines
        shuffle(numbered_lines)
        #Extract the number of lines needed for the training set
        training_line_list = numbered_lines[0:training_set_size]
        #Extract the number of lines needed for the testing set. Starts where the training set stopped
        testing_line_list = numbered_lines[training_set_size: training_set_size+testing_set_size]

    elif sampling_method == SamplingMethod.PsuedoRandom:
        #Tracks used lines
        used_lines: Set[int] = set()
        #The usage rate determines which points will be used. We take the proportion of images being used from the dataset and divide by 10
        # This is to cause the fetching routine to loop over all images multiple times further scrambling the dataset 
        usage_rate: float = (training_set_size + testing_set_size)/num_available_images
        
        #The training rate determins which points that are used, when selected by the usage_rate, will be placed into the training or testing dataset
        train_rate: float = training_set_size/(training_set_size+testing_set_size)

        # flag that marks if a new rate is generated. A new rate is generated only when moving from a point that was not used or a point
        # that was just placed into a dataset. If the point was previously used a value for rate is not generated
        make_new_rate: bool = True
        last_test = 0
        last_traing = 0
        while not done_fetching:
            for numb_line in numbered_lines:
                if make_new_rate:
                    rate_threshold: float = uniform(0.0,1.0)

            #check if this file line has already been used
            indx_used: bool = numb_line[1] in used_lines
            # if this index was used we simply move to the next one that wasn't without generating a new rate value
            if indx_used:
                make_new_rate = False
                continue
            #If the rate was larger than the threshold use the current index we use the datapoint described by the line
            if usage_rate > rate_threshold:
                make_new_rate = True
                #add index to used index set
                used_lines.add(numb_line[1])
                # we check if both datasets need points
                # if one does not need points then we don't need to generate rate to determine which
                # dataet to insert into
                if not train_all_data and not test_all_data:
                    #determine if the used image is for training or testing
                    is_train_img: bool = uniform(0.0, 1.0) <= train_rate
                
                # for each image we try to place it into the dataset described by is_train_img
                # but if all of the data has been gathered for one dataset the point is simply placed into the other one
                if is_train_img and not train_all_data:
                    print("Adding new Training")
                    training_line_list.append(numb_line)
                elif is_train_img and train_all_data:
                    print("Adding new Testing")
                    testing_line_list.append(numb_line)
                elif not is_train_img and not test_all_data:
                    print("Adding new Testing")
                    testing_line_list.append(numb_line)
                elif not is_train_img and test_all_data:
                    print("Adding new Training")
                    training_line_list.append(numb_line)
                #check if all of the points for both sets has been gathered
                train_all_data = training_set_size == len(training_line_list)
                test_all_data = testing_set_size == len(testing_line_list)
                #if both sets have all their points we are done
                #set the done_fetching flag and break from inner loop
                if test_all_data and train_all_data:
                    done_fetching = True
                    break


    elif sampling_method == SamplingMethod.Deterministic:
        class_starts: List[int] = [0]*(num_classes)



        #current class being considered
        curr_class:int = 0
        image_metadata: List[Tuple[str,int]] = ExtractImageMetaDataFromLines(numbered_lines)
        index = 0
        for _, img_class in image_metadata:
            if img_class != curr_class:
                curr_class = img_class
                class_starts[img_class] = index
            index += 1
        
        #used_indexes - for each class map the index used to access to the last used point for that class
        #e.g. for airfield which has an enum value of 0 used_indexes[0] will be equal to the last used index in all_lines that
        #was used for a index for the airfield class
        used_indexes: List[int] = [x-1 for x in class_starts]
        #iterate until both datasets are filled
        while not(test_all_data and train_all_data):
            #iterate over all classes
            for curr_class in range(0,num_classes):
                #check if classes are exhausted
                if curr_class == max_enum and used_indexes[curr_class]>=len(image_metadata):
                    continue
                elif curr_class != max_enum and used_indexes[curr_class]+1 == class_starts[curr_class+1]:
                    continue
                #check if training data is full, if not add a point
                if not train_all_data:
                    #increase the used_indexes for the current class
                    used_indexes[curr_class]+=1
                    #Assigne the image at the index used_indexes[curr_class] to the training set
                    training_line_list.append(numbered_lines[used_indexes[curr_class]])
                    #check if set is full
                    train_all_data = len(training_line_list) == training_set_size
                    #check if class is exhausted before trying to add a testing point
                    if curr_class == max_enum and used_indexes[curr_class]>=len(image_metadata):
                        continue
                    elif curr_class != max_enum and used_indexes[curr_class]+1 == class_starts[curr_class+1]:
                        continue
                #check if test data is full, if not add a point
                #same process as for the training data
                if not test_all_data:
                    used_indexes[curr_class]+=1
                    testing_line_list.append(numbered_lines[used_indexes[curr_class]])
                    test_all_data = len(testing_line_list) == testing_set_size
                if test_all_data and train_all_data:
                    #break out of inner loop.
                    done_fetching = True
                    break


    print(f"The base path is {base_path}")
    base_path = os.path.abspath(base_path)
    print(f"The absolute base path is {base_path}")
    training_set: TrainDataset = TrainDataset(base_path, training_line_list, output_color=output_color, transform=transform)
    testing_set: TestDataset = TestDataset(base_path,testing_line_list, output_color=output_color, transform=transform)
    return training_set, testing_set
