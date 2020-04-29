from __future__ import annotations
from typing import *
from enum import IntEnum, Enum
import re
from bisect import bisect_left
from random import uniform, shuffle
import os
import cv2 # type: ignore
import numpy as np # type: ignore
from collections import namedtuple
from collections.abc import Sequence

#Contains all of the information to fetch a image from disk as well as the ground truth scene classification
#Since all images in the image directories of the dataset are simple 8 digit numbers and all .jpg format it is simple 
#combine the basepath of the file with the class specific directory and the image number to get the full filepath
# <base path>/data/<class specific path>/<left pad image number to 8 digits><image number>.jpg 
class Index(NamedTuple):
    ImgNum: int
    ClassNum: int


#Contains the image data in the format that python-opencv uses as well as the name of the file and the ground truth classification
class DataPoint(NamedTuple):
    Img: np.ndarray
    Name: str
    GroundTruthClass: int


#Thrown when the file that contains a list of image paths relative to the data directory and their ground truth clasiification violates the format somehow
#e.g an empty line or something that doesn't match the regex used to extract the image number and class
#The format for each line of this file, currently named places365_train_standard.txt is:
# /a/airfield/00000007.jpg 0
# This is the path to the image file named 00000007.jpg which is classified as class 0. Class 0 is airfield.
# In somecases the path will be more specific about the scene
# e.g. /b/basketball_court/indoor/00004480.jpg 44
# not that this does not mean that there is an outdoor basketball court
class MalformedSpecFileError(Exception):
    def __init__(self, path:str, line:str, line_number:int, message:str):
        #path to the spec file
        self.path:str = path
        #line that caused the failure
        self.line:str = line
        #The line number of the problem line
        self.line_number: int = line_number
        #User defined message which should give more context to error
        self.message:str = message
    
    def __str__(self)->str:
        return f"There was an error reading line {self.line_number}, which contained '{self.line}', in the specfile, {self.path}, for the image locations: {self.message}"

# Thrown when open image using cv2.imread fails
class OpenImageError(Exception):
    def __init__(self, path:str, message:str):
        #Path of the image that failed to open
        self.path: str = path
        #User message
        self.message: str = message

    def __str__(self)->str:
        return f"There was an error trying to open {self.path}: {self.message}"
 


#takes list of indexes and iterates over them lazily opening each file only when required
class DataIter(Iterator, Mapping[Union[int,slice], Any]):
    def __init__(self, indexes: List[Index], base_path:str, len_img_num:int=8, middle_dir:str="data", transform:Optional[Callable]=None):
        # The length of the number in the image files names
        self.len_img_num:int = len_img_num
        # The list of indexes for the dataset
        self.indexes:List[Index] = indexes
        # The base path of the dataset directory
        self.base_path:str = os.path.join(base_path,middle_dir)
        # The current index of the iterator. This is reset each time the __iter__ method is called
        self.idx:int = 0
        # The length of the dataset
        self.len:int = len(indexes)
        # A transform that will be applied to the image before it is output by the iterator
        self.transform: Optional[Callable] = transform

    #This class itself is an iterator so it returns itself after setting the index to zero
    #May be a good idea to have this return a type which references the data inside DataIter so that multiple different iterators may be used
    # simultaniously
    def __iter__(self)->DataIter:
        self.idx=0
        return self

    # allows len() to be called on DataIter
    def __len__(self)->int:
        return self.len

    # Returns the next DataPoint in the dataset applying the transform function if one was provided
    def __next__(self)->DataPoint:
        if self.idx >= self.len:
            raise StopIteration

        #get the indexes of the current iteration target
        Img_Index: Index = self.indexes[self.idx]
        #increment the iteration index so that the next item is returned on the next call
        self.idx += 1
        img = self.load_img(Img_Index)
        return img

    def load_img(self, idx: Index)->DataPoint:
        #nubmer contained in the name of the image
        img_number:int
        #class of the image - Corresponds to the ImageClass enum defined below
        img_class:int
        #get the indexes of the current iteration target
        img_number, img_class= idx
        #Left pad the image number with the correct number of zeroes and add the .jpg extension
        img_name:str = "0"*(self.len_img_num - len(str(img_number))) + str(img_number) + ".jpg"
        #Combine with the basepath to get the full path to the images
        img_path:str = os.path.abspath(os.path.join(self.base_path, get_class_path(img_class)[1:], img_name))
        #Read the image from the path
        img: Optional[np.ndarray] = cv2.imread(img_path, 1)
        #if img is None opening the file failed. The error handling for imread sucks due to poor visibilty on the root cause of the failure
        #somehow i managed to corrupt some images this is so that I can still test evaluation path
        #if img is None:
        #    raise OpenImageError(img_path, "Opening the image failed while iterating over DataIter")
        #if the transform was supplied apply it now
        if self.transform:
            img = self.transform(img)
        return DataPoint(img, img_name, img_class) 
 
    def __getitem__(self, key: Union[int, slice])->Union[List[DataPoint],DataPoint]:
        if isinstance(key, slice):
            img_indexes:List[Index] = self.indexes[key]
            return [self.load_img(idx) for idx in img_indexes]
        elif isinstance(key, int):
            return self.load_img(self.indexes[key])
        else:
            raise KeyError("DataIter only accepts integers and slices as subscripts")

class SamplingMethod(Enum):
    PsuedoRandom = 0
    Random = 1
    Deterministic = 2    



# For actual project may be a good idea to extend this to inherit from a more powerful data loading library
# I started out trying to extend pytorch's Dataset which allows a pytorch Dataloader to handle batching
# parallel loads and some form of async io
# This would require splitting the fectching test and trainig data into two separate datasets
# and an extra paramter to insert indexes which have already been used and are off limit
# beyond that a descision on a interable dataset or map dataset would need to be made

# This dataset relies on the number of images avaiable being available on dataset object creation
# This can probably be changed to count the number of lines in the target files, I stupidly assumed that that
# woudl take up too much memory where it actually will only take up ~55 Megabytes

# Due to the way the dataset was structure it was simpler to create two different ways to sample the dataset
# for random and deterministic sampling then to pull the data and then shuffle based on the random optinoal argument

# base path - The path to the directory that must contain two subdirectories. 
#                   data - Contains the images in the same format that they are downloaded in
#                   filelist - Contains places365_train_standard.txt which is a list of all images paths relative to 
class Places365Dataset:
    def __init__(self, base_path:str, training:int, testing:int, number_images:int=1803460, sampling_method:SamplingMethod=SamplingMethod.PsuedoRandom, shuffle_dataset_after:bool=False, transform:Optional[Callable]=None):
        self.base_path:str = base_path
        self.transform: Optional[Callable] = transform
        #The current index of the file, used to track progress and provide better diagnostics 
        self.line_idx:int = 0
        #number of images in the whole dataset
        self.total_dataset_size:int = number_images
        #List of Index namedtuples that store all of the information required to open an image
        training_indexes: List[Index]
        testing_indexes: List[Index]
        #both of the data fetching functions will throw an exception if the nubmer of test datapoints nd the number of trainig datapoints
        #exceed the number samples in the dataset
        if sampling_method == SamplingMethod.PsuedoRandom:
            #psuedorandomly select points from dataset
            training_indexes, testing_indexes = self.__psuedorandom_data_fetch(training, testing, number_images)
        elif sampling_method == SamplingMethod.Random:
            #randomly select points from dataset
            training_indexes, testing_indexes = self.__random_data_fetch(training, testing, number_images)
        elif sampling_method == SamplingMethod.Deterministic:
            #deterministcly grab the same images from the dataset
            training_indexes, testing_indexes = self.__iter_data_fetch(training, testing, number_images)
        else:
            raise ValueError("sampling_method was provided a value which was not a part of the SamplingMethod enum")
        if shuffle_dataset_after and not sampling_method == SamplingMethod.Random:
            shuffle(training_indexes)
            shuffle(testing_indexes)
        #transform the training and test datasets into iterators 
        self.training_data: DataIter = DataIter(training_indexes, self.base_path, transform=transform)
        self.testing_data: DataIter = DataIter(testing_indexes, self.base_path, transform=transform)


    def __random_data_fetch(self, num_training:int, num_testing:int, num_images:int)->Tuple[List[Index], List[Index]]:
        #check that the number of images is within normal bounds
        if num_images < num_training + num_testing:
            raise ValueError(f"There are only {num_images} images, with a test set of {num_testing} and a training set of {num_training} you have excedded the size of the dataseet.")
        #List of indexs for the gathered datasets
        training_indexes: List[Index] = []
        testing_indexes: List[Index] = []

        #opens the specfile in read mode

        with self.__read_SpecFile() as f:
            all_lines = f.readlines()
        shuffle(all_lines)
        for i in range(0, num_training):
            new_idx = self.__get_indexes(all_lines[i])
            training_indexes.append(new_idx)
        for i in range(num_training,num_training+num_testing):
            new_idx = self.__get_indexes(all_lines[i])
            testing_indexes.append(new_idx)
        return training_indexes, testing_indexes


    #when provided information about the number of images wanted and the number of images in the dataset returns a training and 
    #testing dataset in a tuple with training first
    #throws an exception if the nubmer of test datapoints nd the number of trainig datapoints
    #exceed the number samples in the dataset
    #will tend towards semi ordered datasets which are balanced across classes
    def __psuedorandom_data_fetch(self, num_training:int, num_testing:int, num_images:int)->Tuple[List[Index], List[Index]]:
        #set that tracks which indexes that have been used already
        used_indexes: Set[int] = set()
        #check that the number of images is within normal bounds
        if num_images < num_training + num_testing:
            raise ValueError(f"There are only {num_images} images, with a test set of {num_testing} and a training set of {num_training} you have excedded the size of the dataseet.")
        
        #The usage rate determines which points will be used. We take the proportion of images being used from the dataset and divide by 10
        # This is to cause the fetching routine to loop over all images multiple times further scrambling the dataset 
        usage_rate: float = ((num_training + num_testing)/num_images)
        
        #The training rate determins which points that are used, when selected by the usage_rate, will be placed into the training or testing dataset
        train_rate: float = num_training/(num_training+num_testing)

        #Flags that mark when the required number of points that have been gathered
        test_all_data: bool = False
        train_all_data: bool = False

        #List of indexs for the gathered datasets
        training_indexes: List[Index] = []
        testing_indexes: List[Index] = []

        #opens the specfile in read mode

        with self.__read_SpecFile() as f:
            all_lines: List[str] = f.readlines()

        #marks if done iterating over file lines
        done_fetching: bool = False
        #flage that marks if a new rate is generated. A new rate is generated only when moving from a point that was not used or a point
        # that was just placed into a dataset. If the point was previously used a value for rate is not generated
        make_new_rate: bool = True
        self.line_idx = 0
        last_test = 0
        last_traing = 0
        while not done_fetching:
            if last_test != len(testing_indexes) or last_traing != len(training_indexes):
                print(f"There are {len(testing_indexes)} testing points and {len(training_indexes)}")
                last_test = len(testing_indexes)
                last_traing = len(training_indexes)
            if self.line_idx >= len(all_lines):
                self.line_idx = 0
            line = all_lines[self.line_idx]
            if make_new_rate:
                rate_threshold: float = uniform(0.0,1.0)
            #check if this file line has already been used
            indx_used: bool = self.line_idx in used_indexes
            # if this index was used we simply move to the next one that wasn't without generating a new rate value
            if indx_used:
                self.line_idx += 1
                make_new_rate = False
                continue
            #If the rate was larger than the threshold use the current index we use the datapoint described by the line
            if usage_rate > rate_threshold:
                make_new_rate = True
                #add index to used index set
                used_indexes.add(self.line_idx)
                # we check if both datasets need points
                # if one does not need points then we don't need to generate rate to determine which
                # dataet to insert into
                if not train_all_data and not test_all_data:
                    #determine if the used image is for training or testing
                    is_train_img: bool = uniform(0.0, 1.0) <= train_rate
                #get the indexes of the image
                new_index: Index = self.__get_indexes(line)
                # for each image we try to place it into the dataset described by is_train_img
                # but if all of the data has been gathered for one dataset the point is simply placed into the other one
                if is_train_img and not train_all_data:
                    print("Adding new Training")
                    training_indexes.append(new_index)
                elif is_train_img and train_all_data:
                    print("Adding new Testing")
                    testing_indexes.append(new_index)
                elif not is_train_img and not test_all_data:
                    print("Adding new Testing")
                    testing_indexes.append(new_index)
                elif not is_train_img and test_all_data:
                    print("Adding new Training")
                    training_indexes.append(new_index)
                #check if all of the points for both sets has been gathered
                train_all_data = num_training == len(training_indexes)
                test_all_data = num_testing == len(testing_indexes)
                #if both sets have all their points we are done
                #set the done_fetching flag and break from inner loop
                if test_all_data and train_all_data:
                    done_fetching = True
                    break
            #the line tracking is purely for diagnostic purpose so i can tell which line number is broken in specfile
            self.line_idx += 1
        return training_indexes, testing_indexes
    
    #Iterate over the data assigning points alternativly to test and training 
    def __iter_data_fetch(self, num_training: int, num_testing:int, num_images:int)->Tuple[List[Index], List[Index]]:
        #check dataset bounds
        if num_images < num_training + num_testing:
            raise ValueError(f"There are only {num_images} images, with a test set of {num_testing} and a training set of {num_training} you have excedded the size of the dataseet.")
        #A list that is filled with the starting index of each class
        #e.g. airfield, which is 0 in the enum will always start at index 0 and therefore class_start[0] == 0
        class_start: List[int] = [0]*(max_enum_idx()+1)
        #flags for full dataset
        test_all_data: bool = False
        train_all_data: bool = False

        # datasets containing the indexes of the images
        training_indexes: List[Index] = []
        testing_indexes: List[Index] = []

        #current class being considered
        curr_class:int = 0
        #grab_test:bool = True
        #grab_train = True
        self.line_idx = 0
        max_enum: int = max_enum_idx()
        with self.__read_SpecFile() as f:
            all_lines: List[str] = f.readlines()
            #get the class boundaries and set all of the starting points in class_start
            for line in all_lines:
                img_number: int
                class_number: int
                img_number, class_number = self.__get_indexes(line)
                if class_number != curr_class:
                    curr_class = class_number
                    class_start[class_number] = self.line_idx
                self.line_idx += 1 
            #used_indexes - for each class map the index used to access to the last used point for that class
            #e.g. for airfield which has an enum value of 0 used_indexes[0] will be equal to the last used index in all_lines that
            #was used for a index for the airfield class
            used_indexes: List[int] = [x-1 for x in class_start]
            new_idx: Index
            #iterate until both datasets are filled
            while not(test_all_data and train_all_data):
                #iterate over all classes
                for curr_class in range(0,max_enum_idx()+1):
                    #check if classes are exhausted
                    if curr_class == max_enum and used_indexes[curr_class]>=len(all_lines):
                        continue
                    elif curr_class != max_enum and used_indexes[curr_class]+1 == class_start[curr_class+1]:
                        continue
                    #check if training data is full, if not add a point
                    if not train_all_data:
                        #increase the used_indexes for the current class
                        used_indexes[curr_class]+=1
                        self.line_idx = used_indexes[curr_class]
                        #create index from line
                        new_idx = self.__get_indexes(all_lines[self.line_idx])
                        #append to set
                        training_indexes.append(new_idx)
                        #check if set is full
                        train_all_data = len(training_indexes) == num_training
                        #check if class is exhausted before trying to add a testing point
                        if curr_class == max_enum and used_indexes[curr_class]>=len(all_lines):
                            continue
                        elif curr_class != max_enum and used_indexes[curr_class]+1 == class_start[curr_class+1]:
                            continue
                    #check if test data is full, if not add a point
                    #same process as for the training data
                    if not test_all_data:
                        used_indexes[curr_class]+=1
                        self.line_idx = used_indexes[curr_class]
                        new_idx = self.__get_indexes(all_lines[self.line_idx])
                        testing_indexes.append(new_idx)
                        test_all_data = len(testing_indexes) == num_testing
                    if test_all_data and train_all_data:
                        #break out of inner loop. With this condition fufilled the outer loop will immediately be exited
                        break
        return training_indexes, testing_indexes

    def __get_indexes(self,line:str)->Index:
        m = re.match(r'^.*?(\d+)\.jpg\s*(\d+)$', line)
        if m:
            return Index(int(m.groups('1')[0]), int(m.groups('1')[1]))
        raise MalformedSpecFileError(self.__get_SpecFile_path(),line, self.line_idx, "The index extracting regex could not extract the image number and class from the line")
                        
    def __read_SpecFile(self)->TextIO:
        path:str = self.__get_SpecFile_path()
        f:TextIO = open(path,'r')
        return f

    def __get_SpecFile_path(self)->str:
        return os.path.join(self.base_path , "places365_train_standard.txt")
                        

def get_class_path(ClassEnum)->str:
    return CLASS_PATHS[ClassEnum]

def max_enum_idx()->int:
    return 364

#allows to map from enum to path
CLASS_PATHS: Tuple[str, ...] = ( "/a/airfield", "/a/airplane_cabin", "/a/airport_terminal", "/a/alcove", "/a/alley", "/a/amphitheater", "/a/amusement_arcade", "/a/amusement_park", "/a/apartment_building/outdoor", "/a/aquarium", "/a/aqueduct", "/a/arcade", "/a/arch", "/a/archaelogical_excavation", "/a/archive", "/a/arena/hockey", "/a/arena/performance", "/a/arena/rodeo", "/a/army_base", "/a/art_gallery", "/a/art_school", "/a/art_studio", "/a/artists_loft", "/a/assembly_line", "/a/athletic_field/outdoor", "/a/atrium/public", "/a/attic", "/a/auditorium", "/a/auto_factory", "/a/auto_showroom", "/b/badlands", "/b/bakery/shop", "/b/balcony/exterior", "/b/balcony/interior", "/b/ball_pit", "/b/ballroom", "/b/bamboo_forest", "/b/bank_vault", "/b/banquet_hall", "/b/bar", "/b/barn", "/b/barndoor", "/b/baseball_field", "/b/basement", "/b/basketball_court/indoor", "/b/bathroom", "/b/bazaar/indoor", "/b/bazaar/outdoor", "/b/beach", "/b/beach_house", "/b/beauty_salon", "/b/bedchamber", "/b/bedroom", "/b/beer_garden", "/b/beer_hall", "/b/berth", "/b/biology_laboratory", "/b/boardwalk", "/b/boat_deck", "/b/boathouse", "/b/bookstore", "/b/booth/indoor", "/b/botanical_garden", "/b/bow_window/indoor", "/b/bowling_alley", "/b/boxing_ring", "/b/bridge", "/b/building_facade", "/b/bullring", "/b/burial_chamber", "/b/bus_interior", "/b/bus_station/indoor", "/b/butchers_shop", "/b/butte", "/c/cabin/outdoor", "/c/cafeteria", "/c/campsite", "/c/campus", "/c/canal/natural", "/c/canal/urban", "/c/candy_store", "/c/canyon", "/c/car_interior", "/c/carrousel", "/c/castle", "/c/catacomb", "/c/cemetery", "/c/chalet", "/c/chemistry_lab", "/c/childs_room", "/c/church/indoor", "/c/church/outdoor", "/c/classroom", "/c/clean_room", "/c/cliff", "/c/closet", "/c/clothing_store", "/c/coast", "/c/cockpit", "/c/coffee_shop", "/c/computer_room", "/c/conference_center", "/c/conference_room", "/c/construction_site", "/c/corn_field", "/c/corral", "/c/corridor", "/c/cottage", "/c/courthouse", "/c/courtyard", "/c/creek", "/c/crevasse", "/c/crosswalk", "/d/dam", "/d/delicatessen", "/d/department_store", "/d/desert/sand", "/d/desert/vegetation", "/d/desert_road", "/d/diner/outdoor", "/d/dining_hall", "/d/dining_room", "/d/discotheque", "/d/doorway/outdoor", "/d/dorm_room", "/d/downtown", "/d/dressing_room", "/d/driveway", "/d/drugstore", "/e/elevator/door", "/e/elevator_lobby", "/e/elevator_shaft", "/e/embassy", "/e/engine_room", "/e/entrance_hall", "/e/escalator/indoor", "/e/excavation", "/f/fabric_store", "/f/farm", "/f/fastfood_restaurant", "/f/field/cultivated", "/f/field/wild", "/f/field_road", "/f/fire_escape", "/f/fire_station", "/f/fishpond", "/f/flea_market/indoor", "/f/florist_shop/indoor", "/f/food_court", "/f/football_field", "/f/forest/broadleaf", "/f/forest_path", "/f/forest_road", "/f/formal_garden", "/f/fountain", "/g/galley", "/g/garage/indoor", "/g/garage/outdoor", "/g/gas_station", "/g/gazebo/exterior", "/g/general_store/indoor", "/g/general_store/outdoor", "/g/gift_shop", "/g/glacier", "/g/golf_course", "/g/greenhouse/indoor", "/g/greenhouse/outdoor", "/g/grotto", "/g/gymnasium/indoor", "/h/hangar/indoor", "/h/hangar/outdoor", "/h/harbor", "/h/hardware_store", "/h/hayfield", "/h/heliport", "/h/highway", "/h/home_office", "/h/home_theater", "/h/hospital", "/h/hospital_room", "/h/hot_spring", "/h/hotel/outdoor", "/h/hotel_room", "/h/house", "/h/hunting_lodge/outdoor", "/i/ice_cream_parlor", "/i/ice_floe", "/i/ice_shelf", "/i/ice_skating_rink/indoor", "/i/ice_skating_rink/outdoor", "/i/iceberg", "/i/igloo", "/i/industrial_area", "/i/inn/outdoor", "/i/islet", "/j/jacuzzi/indoor", "/j/jail_cell", "/j/japanese_garden", "/j/jewelry_shop", "/j/junkyard", "/k/kasbah", "/k/kennel/outdoor", "/k/kindergarden_classroom", "/k/kitchen", "/l/lagoon", "/l/lake/natural", "/l/landfill", "/l/landing_deck", "/l/laundromat", "/l/lawn", "/l/lecture_room", "/l/legislative_chamber", "/l/library/indoor", "/l/library/outdoor", "/l/lighthouse", "/l/living_room", "/l/loading_dock", "/l/lobby", "/l/lock_chamber", "/l/locker_room", "/m/mansion", "/m/manufactured_home", "/m/market/indoor", "/m/market/outdoor", "/m/marsh", "/m/martial_arts_gym", "/m/mausoleum", "/m/medina", "/m/mezzanine", "/m/moat/water", "/m/mosque/outdoor", "/m/motel", "/m/mountain", "/m/mountain_path", "/m/mountain_snowy", "/m/movie_theater/indoor", "/m/museum/indoor", "/m/museum/outdoor", "/m/music_studio", "/n/natural_history_museum", "/n/nursery", "/n/nursing_home", "/o/oast_house", "/o/ocean", "/o/office", "/o/office_building", "/o/office_cubicles", "/o/oilrig", "/o/operating_room", "/o/orchard", "/o/orchestra_pit", "/p/pagoda", "/p/palace", "/p/pantry", "/p/park", "/p/parking_garage/indoor", "/p/parking_garage/outdoor", "/p/parking_lot", "/p/pasture", "/p/patio", "/p/pavilion", "/p/pet_shop", "/p/pharmacy", "/p/phone_booth", "/p/physics_laboratory", "/p/picnic_area", "/p/pier", "/p/pizzeria", "/p/playground", "/p/playroom", "/p/plaza", "/p/pond", "/p/porch", "/p/promenade", "/p/pub/indoor", "/r/racecourse", "/r/raceway", "/r/raft", "/r/railroad_track", "/r/rainforest", "/r/reception", "/r/recreation_room", "/r/repair_shop", "/r/residential_neighborhood", "/r/restaurant", "/r/restaurant_kitchen", "/r/restaurant_patio", "/r/rice_paddy", "/r/river", "/r/rock_arch", "/r/roof_garden", "/r/rope_bridge", "/r/ruin", "/r/runway", "/s/sandbox", "/s/sauna", "/s/schoolhouse", "/s/science_museum", "/s/server_room", "/s/shed", "/s/shoe_shop", "/s/shopfront", "/s/shopping_mall/indoor", "/s/shower", "/s/ski_resort", "/s/ski_slope", "/s/sky", "/s/skyscraper", "/s/slum", "/s/snowfield", "/s/soccer_field", "/s/stable", "/s/stadium/baseball", "/s/stadium/football", "/s/stadium/soccer", "/s/stage/indoor", "/s/stage/outdoor", "/s/staircase", "/s/storage_room", "/s/street", "/s/subway_station/platform", "/s/supermarket", "/s/sushi_bar", "/s/swamp", "/s/swimming_hole", "/s/swimming_pool/indoor", "/s/swimming_pool/outdoor", "/s/synagogue/outdoor", "/t/television_room", "/t/television_studio", "/t/temple/asia", "/t/throne_room", "/t/ticket_booth", "/t/topiary_garden", "/t/tower", "/t/toyshop", "/t/train_interior", "/t/train_station/platform", "/t/tree_farm", "/t/tree_house", "/t/trench", "/t/tundra", "/u/underwater/ocean_deep", "/u/utility_room", "/v/valley", "/v/vegetable_garden", "/v/veterinarians_office", "/v/viaduct", "/v/village", "/v/vineyard", "/v/volcano", "/v/volleyball_court/outdoor", "/w/waiting_room", "/w/water_park", "/w/water_tower", "/w/waterfall", "/w/watering_hole", "/w/wave", "/w/wet_bar", "/w/wheat_field", "/w/wind_farm", "/w/windmill", "/y/yard", "/y/youth_hostel", "/z/zen_garden" )
#all classes of scenes
class ImageClass(IntEnum):
    airfield = 0
    airplane_cabin = 1
    airport_terminal = 2
    alcove = 3
    alley = 4
    amphitheater = 5
    amusement_arcade = 6
    amusement_park = 7
    apartment_building__outdoor = 8
    aquarium = 9
    aqueduct = 10
    arcade = 11
    arch = 12
    archaelogical_excavation = 13
    archive = 14
    arena__hockey = 15
    arena__performance = 16
    arena__rodeo = 17
    army_base = 18
    art_gallery = 19
    art_school = 20
    art_studio = 21
    artists_loft = 22
    assembly_line = 23
    athletic_field__outdoor = 24
    atrium__public = 25
    attic = 26
    auditorium = 27
    auto_factory = 28
    auto_showroom = 29
    badlands = 30
    bakery__shop = 31
    balcony__exterior = 32
    balcony__interior = 33
    ball_pit = 34
    ballroom = 35
    bamboo_forest = 36
    bank_vault = 37
    banquet_hall = 38
    bar = 39
    barn = 40
    barndoor = 41
    baseball_field = 42
    basement = 43
    basketball_court__indoor = 44
    bathroom = 45
    bazaar__indoor = 46
    bazaar__outdoor = 47
    beach = 48
    beach_house = 49
    beauty_salon = 50
    bedchamber = 51
    bedroom = 52
    beer_garden = 53
    beer_hall = 54
    berth = 55
    biology_laboratory = 56
    boardwalk = 57
    boat_deck = 58
    boathouse = 59
    bookstore = 60
    booth__indoor = 61
    botanical_garden = 62
    bow_window__indoor = 63
    bowling_alley = 64
    boxing_ring = 65
    bridge = 66
    building_facade = 67
    bullring = 68
    burial_chamber = 69
    bus_interior = 70
    bus_station__indoor = 71
    butchers_shop = 72
    butte = 73
    cabin__outdoor = 74
    cafeteria = 75
    campsite = 76
    campus = 77
    canal__natural = 78
    canal__urban = 79
    candy_store = 80
    canyon = 81
    car_interior = 82
    carrousel = 83
    castle = 84
    catacomb = 85
    cemetery = 86
    chalet = 87
    chemistry_lab = 88
    childs_room = 89
    church__indoor = 90
    church__outdoor = 91
    classroom = 92
    clean_room = 93
    cliff = 94
    closet = 95
    clothing_store = 96
    coast = 97
    cockpit = 98
    coffee_shop = 99
    computer_room = 100
    conference_center = 101
    conference_room = 102
    construction_site = 103
    corn_field = 104
    corral = 105
    corridor = 106
    cottage = 107
    courthouse = 108
    courtyard = 109
    creek = 110
    crevasse = 111
    crosswalk = 112
    dam = 113
    delicatessen = 114
    department_store = 115
    desert__sand = 116
    desert__vegetation = 117
    desert_road = 118
    diner__outdoor = 119
    dining_hall = 120
    dining_room = 121
    discotheque = 122
    doorway__outdoor = 123
    dorm_room = 124
    downtown = 125
    dressing_room = 126
    driveway = 127
    drugstore = 128
    elevator__door = 129
    elevator_lobby = 130
    elevator_shaft = 131
    embassy = 132
    engine_room = 133
    entrance_hall = 134
    escalator__indoor = 135
    excavation = 136
    fabric_store = 137
    farm = 138
    fastfood_restaurant = 139
    field__cultivated = 140
    field__wild = 141
    field_road = 142
    fire_escape = 143
    fire_station = 144
    fishpond = 145
    flea_market__indoor = 146
    florist_shop__indoor = 147
    food_court = 148
    football_field = 149
    forest__broadleaf = 150
    forest_path = 151
    forest_road = 152
    formal_garden = 153
    fountain = 154
    galley = 155
    garage__indoor = 156
    garage__outdoor = 157
    gas_station = 158
    gazebo__exterior = 159
    general_store__indoor = 160
    general_store__outdoor = 161
    gift_shop = 162
    glacier = 163
    golf_course = 164
    greenhouse__indoor = 165
    greenhouse__outdoor = 166
    grotto = 167
    gymnasium__indoor = 168
    hangar__indoor = 169
    hangar__outdoor = 170
    harbor = 171
    hardware_store = 172
    hayfield = 173
    heliport = 174
    highway = 175
    home_office = 176
    home_theater = 177
    hospital = 178
    hospital_room = 179
    hot_spring = 180
    hotel__outdoor = 181
    hotel_room = 182
    house = 183
    hunting_lodge__outdoor = 184
    ice_cream_parlor = 185
    ice_floe = 186
    ice_shelf = 187
    ice_skating_rink__indoor = 188
    ice_skating_rink__outdoor = 189
    iceberg = 190
    igloo = 191
    industrial_area = 192
    inn__outdoor = 193
    islet = 194
    jacuzzi__indoor = 195
    jail_cell = 196
    japanese_garden = 197
    jewelry_shop = 198
    junkyard = 199
    kasbah = 200
    kennel__outdoor = 201
    kindergarden_classroom = 202
    kitchen = 203
    lagoon = 204
    lake__natural = 205
    landfill = 206
    landing_deck = 207
    laundromat = 208
    lawn = 209
    lecture_room = 210
    legislative_chamber = 211
    library__indoor = 212
    library__outdoor = 213
    lighthouse = 214
    living_room = 215
    loading_dock = 216
    lobby = 217
    lock_chamber = 218
    locker_room = 219
    mansion = 220
    manufactured_home = 221
    market__indoor = 222
    market__outdoor = 223
    marsh = 224
    martial_arts_gym = 225
    mausoleum = 226
    medina = 227
    mezzanine = 228
    moat__water = 229
    mosque__outdoor = 230
    motel = 231
    mountain = 232
    mountain_path = 233
    mountain_snowy = 234
    movie_theater__indoor = 235
    museum__indoor = 236
    museum__outdoor = 237
    music_studio = 238
    natural_history_museum = 239
    nursery = 240
    nursing_home = 241
    oast_house = 242
    ocean = 243
    office = 244
    office_building = 245
    office_cubicles = 246
    oilrig = 247
    operating_room = 248
    orchard = 249
    orchestra_pit = 250
    pagoda = 251
    palace = 252
    pantry = 253
    park = 254
    parking_garage__indoor = 255
    parking_garage__outdoor = 256
    parking_lot = 257
    pasture = 258
    patio = 259
    pavilion = 260
    pet_shop = 261
    pharmacy = 262
    phone_booth = 263
    physics_laboratory = 264
    picnic_area = 265
    pier = 266
    pizzeria = 267
    playground = 268
    playroom = 269
    plaza = 270
    pond = 271
    porch = 272
    promenade = 273
    pub__indoor = 274
    racecourse = 275
    raceway = 276
    raft = 277
    railroad_track = 278
    rainforest = 279
    reception = 280
    recreation_room = 281
    repair_shop = 282
    residential_neighborhood = 283
    restaurant = 284
    restaurant_kitchen = 285
    restaurant_patio = 286
    rice_paddy = 287
    river = 288
    rock_arch = 289
    roof_garden = 290
    rope_bridge = 291
    ruin = 292
    runway = 293
    sandbox = 294
    sauna = 295
    schoolhouse = 296
    science_museum = 297
    server_room = 298
    shed = 299
    shoe_shop = 300
    shopfront = 301
    shopping_mall__indoor = 302
    shower = 303
    ski_resort = 304
    ski_slope = 305
    sky = 306
    skyscraper = 307
    slum = 308
    snowfield = 309
    soccer_field = 310
    stable = 311
    stadium__baseball = 312
    stadium__football = 313
    stadium__soccer = 314
    stage__indoor = 315
    stage__outdoor = 316
    staircase = 317
    storage_room = 318
    street = 319
    subway_station__platform = 320
    supermarket = 321
    sushi_bar = 322
    swamp = 323
    swimming_hole = 324
    swimming_pool__indoor = 325
    swimming_pool__outdoor = 326
    synagogue__outdoor = 327
    television_room = 328
    television_studio = 329
    temple__asia = 330
    throne_room = 331
    ticket_booth = 332
    topiary_garden = 333
    tower = 334
    toyshop = 335
    train_interior = 336
    train_station__platform = 337
    tree_farm = 338
    tree_house = 339
    trench = 340
    tundra = 341
    underwater__ocean_deep = 342
    utility_room = 343
    valley = 344
    vegetable_garden = 345
    veterinarians_office = 346
    viaduct = 347
    village = 348
    vineyard = 349
    volcano = 350
    volleyball_court__outdoor = 351
    waiting_room = 352
    water_park = 353
    water_tower = 354
    waterfall = 355
    watering_hole = 356
    wave = 357
    wet_bar = 358
    wheat_field = 359
    wind_farm = 360
    windmill = 361
    yard = 362
    youth_hostel = 363
    zen_garden = 364


def prettystr(self:ImageClass)->str:
    return str(self)[11:]

def test_all_sampling_types():
    for i in range(0,3):
        enum_type:SamplingMethod = SamplingMethod(i)
        dataset:Places365Dataset = Places365Dataset("./dataset", 3, 4, sampling_method=enum_type)
        train_count:int = 0
        test_count:int = 0
        for t in dataset.training_data:
            train_count +=1
        for t in dataset.testing_data:
            test_count += 1
        print(f"For the sampling method {str(enum_type)} there were {train_count} training points and {test_count} testing points")

def make_path(datapoint: Union[DataPoint, List[DataPoint]])->Union[str,List[str]]:
    make_point_path = lambda point: os.path.join("./test/train", str(ImageClass(point.GroundTruthClass))[11:] + point.Name)
    if isinstance(datapoint, list):
        path_list:List[str] = []
        for point in datapoint:
            path_list.append(make_point_path(datapoint))
        return path_list
    return make_point_path(make_point_path(datapoint))

def test_shuffling_after():
    dataset:Places365Dataset = Places365Dataset("./dataset", 3, 4, sampling_method=SamplingMethod.Deterministic, shuffle_dataset_after=True)
    train_count:int = 0
    test_count:int = 0
    print("Training dataset")
    for t in dataset.training_data:
        img_path = os.path.join("./test/train", str(ImageClass(t[2]))[11:] + t[1])
        print(f"{img_path}")
        train_count +=1
    print("Testing Indexing")
    print(f"training[-1]:{make_path(dataset.training_data[-1])}")
    print(f"training[1]{make_path(dataset.training_data[1])}")
    print(f"training[0:3]{[make_path(point) for point in dataset.training_data][0:3]}")
    print(f"training[-1:-12:-1]{[make_path(point) for point in dataset.training_data][-1:-12:-1]}")

    print("\n\n\nTesting dataset")
    for t in dataset.testing_data:
        img_path = os.path.join("./test/train", str(ImageClass(t[2]))[11:] + t[1])
        print(f"{img_path}")
        test_count += 1

if __name__ == "__main__":
    #test_all_sampling_types()
    test_shuffling_after()
     
     
        #img_path = os.path.join("./test/train", str(ImageClass(t[2]))[11:] + t[1])
        #wrote_image = cv2.imwrite(img_path, t[0])
        #if not wrote_image:
        #    print(f"Could not write file to {img_path}")
        #else:
        #    print(f"wrote file to {img_path}")
   

    