from __future__ import annotations
from typing import *
import numpy as np #type: ignore
from enum import Enum
from DataLoader import LoadDatasets, SamplingMethod, Colour, TrainDataset, TestDataset
from VisualWords import MakeGrayPatchExtractor
from sklearn import cluster, metrics, neighbors #type: ignore
import itertools
import math
import os
import sys
import numba #type: ignore

class ClusteringMethod(Enum):
    kmeans = 0
    OPTICS = 1

BaseDatasetPath = "./dataset/data"
BaseImagListPath = "./dataset/filelist/test_file_list.txt"

TrainSize = 80
TestSize = 10

SampleMethod = SamplingMethod.Deterministic


ImageColour = Colour.GRAY
ImageTransform: Any = None


Cluster_Method = ClusteringMethod.OPTICS

Eps = 1e-6
MaxIter = 1000

class FeatureExtractors(Enum):
    
    Gray_Patches = 0
    Colour_Patches = 1
    Gray_SIFT = 2
    Colour_SIFT = 3
    Gray_SIFT_Sparse = 4


@numba.jit(nopython=True)
def logLiklihood_jit(P_TopicGivenDocument, Co_OccurenceTable, P_TopicGivenDocumentWord, P_WordGivenTopic) -> float:
    ll = 0.0
    for N in range(P_TopicGivenDocument[1]):
        for M in range(Co_OccurenceTable.shape[0]):
            partial_sum = 0.0
            for K in range(P_TopicGivenDocument.shape[0]):
                partial_sum += P_TopicGivenDocumentWord[K, N, M]*np.log(P_WordGivenTopic[M, K] * P_TopicGivenDocument[K,N])

            ll += partial_sum * Co_OccurenceTable[N,M]
    return ll



class PLSA:
    def __init__(self, NumVisualWords: int, NumTopics: int, NumNeighbors:int ):
        self.TrainingDataset: TrainDataset
        self.TestingDataset: TestDataset
        self.TrainingDataset, self.TestingDataset = LoadDatasets(BaseImagListPath, BaseDatasetPath, TrainSize, TestSize, 4, SampleMethod, ImageColour, ImageTransform)
        self.Extractor = MakeGrayPatchExtractor() 
        self.Co_OccurenceTable = np.zeros((1,1))
        self.NumVisualWords = NumVisualWords
        self.NumTopics = NumTopics
        self.NumNeighbors = NumNeighbors

        self.WordClassifier = neighbors.KNeighborsClassifier()

        # Axis 0 - Number of topics
        # Axis 1 - Length of training set or number of documents
        self.P_TopicGivenDocument = np.random.random((self.NumTopics, len(self.TrainingDataset)), dtype="float64")
        #Normalize the conditional distribution 
        self.P_TopicGivenDocument = self.P_TopicGivenDocument/np.sum(self.P_TopicGivenDocument, axis=0)
        
        
        # Axis 0 - Number of unique words
        # Axis 1 - Number of topics
        self.P_WordGivenTopic = np.full((NumVisualWords, NumTopics), 1/NumVisualWords, dtype="float64")
        #A uniform distribution is assumed at first so no additional normalization is required


        # Axis 0 - number of topics
        # Axis 1 - number of documents
        # Axis 2 - number of words
        self.P_TopicGivenDocumentWord = np.random.random((NumTopics, len(self.TrainingDataset), NumVisualWords), dtype="float64")
        self.P_TopicGivenDocumentWord = self.P_TopicGivenDocumentWord/ np.sum(self.P_TopicGivenDocumentWord, axis = 0)

    def train(self):
        AllVisualWords = np.zeros((1,1))
        NumberOfImageFeatures = []
        numFeatures = -1
        feature_idx = 0
        image_idx = 0
        for img in self.TrainingDataset:
            vWords = self.Extractor(img[0])
            NumberOfImageFeatures.append(vWords.NumFeatureVecs)
            if numFeatures == -1:
                totalNumberOfFeatures = vWords.NumFeatureVecs * len(self.TrainingDataset)
                numFeatures = vWords.NumFeatureVecs
                AllVisualWords = np.zeros((totalNumberOfFeatures, vWords.FeatureDim))
            if AllVisualWords.shape[0] < feature_idx + vWords.NumFeatureVecs:
                AllVisualWords = np.pad(AllVisualWords,((0,vWords.NumFeatureVecs*(len(self.TrainingDataset)-image_idx)), (0,0)))
            for feature in vWords:
                try:
                    AllVisualWords[feature_idx, :] = feature
                    feature_idx += 1
                except Exception as e:
                    print(f"On the {image_idx}th image out of {len(self.TrainingDataset)}")
                    print(f"There are {numFeatures} per image\n On the {feature_idx}th feature of the image")
                    print(e)
                    exit(1)
            image_idx += 1
        labels  = self.cluster(AllVisualWords[0:feature_idx,:])




        img_index = 0
        img_word_index = 0

        self.Co_OccurenceTable = np.zeros((self.NumVisualWords, len(self.TrainingDataset)), dtype="uint32")
        for label in labels:
            self.Co_OccurenceTable[label, img_index] += 1
            img_word_index += 1
            if img_word_index == NumberOfImageFeatures[img_index]:
                img_word_index = 0
                img_index += 1

        

        #Beginning EM maximization


        logLiklihood = sys.float_info.min
        for iteration in range(MaxIter): 
            pass

        

        
    def logLiklihood(self, P_TopicGivenDocument, Co_OccurencTable, P_TopicGivenDocumentWord, P_WordGivenTopic) -> float:
        return logLiklihood_jit(P_TopicGivenDocument, Co_OccurencTable, P_TopicGivenDocumentWord, P_WordGivenTopic)
        
        
        
    def cluster(self, data)->List[int]:
        if Cluster_Method == ClusteringMethod.kmeans:
            print("Clustering using kmeans")
            return self.kmeans_cluster(data)
        elif Cluster_Method == ClusteringMethod.OPTICS:
            print("Clustering using the OPTICS")
            return self.OPTICS_cluster(data)
        else:
            raise ValueError("Did not recognize clustering algorithm")
        
    def kmeans_cluster(self, data)->List[int]:
        c = cluster.MiniBatchKMeans(n_clusters=self.NumVisualWords, init='k-means++', max_iter=1000).fit(data)
        return c.labels_
        

    def OPTICS_cluster(self, data)->List[int]:
        per_dimension_eps = 5
        eps_total = per_dimension_eps*math.sqrt(data.shape[1])
        c = cluster.OPTICS(n_jobs=3, max_eps=eps_total).fit(data)
        self.NumVisualWords = c.cluster_hierarchy_.shape[0]
        return c.labels_


if __name__ == "__main__":
    #Using values from paper
    NumVisualWords = 1500
    NumTopics = 25
    NumNeighbors = 10

    model = PLSA(NumVisualWords, NumTopics, NumNeighbors)
    model.train()






