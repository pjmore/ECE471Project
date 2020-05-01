# ECE471Project
## Instructions
### Dataset
The dataset can be found at:

http://places2.csail.mit.edu/download.html

It is roughly 100 GB and our run used a small subset of the dataset at 2000 images, unfortunately the implementation is extremely memory innefecient and 16GB wasn't suffecient. Could use the small pictures dataset although the loss in detail will probably impact performance of the classifier.

### Configuration

There are two paramters that may need to be changed in config.py, BaseImageListPath and 
BaseDatasetPath. 

BaseImageListPath - The path to the image list file. This was places365_train_standard.txt in the standard dataset.

BaseDatasetPath is the path to the root directory of the dataset. This is prepended to each file listed in the imagelist file

### Running 

To train, test, and print the performance metrics run the following command

<Python 3 interpreter> PLSA.py



## Dependency Licenses

All dependency licenses can be found in the licenses.txt file
