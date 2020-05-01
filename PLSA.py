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
import cmath
import numba #type: ignore  
from numba import prange #type: ignore
import config
class ClusteringMethod(Enum):
    kmeans = 0
    OPTICS = 1

class SummaryPerformanceMetrics(NamedTuple):
    Accuracy: float
    Support: int
    MacroAvgPrecision: float
    MacroAvgRecall: float
    MacroAvgf1: float
    MacroAvgSupport:int
    WeightAvgPrecision: float
    WeightedAvgRecall: float
    WeightedAvgf1: float
    WeightedAvgSupport: int

    def __str__(self):
        return f"Summary Performance Metrics:\n\tAccuracy: {self.Accuracy}\n\tSupport: {self.Support}\n\tMacro average precision: {self.MacroAvgPrecision}\n\tMacro average recall: {self.MacroAvgRecall}\n\tMacro average f1 score: {self.MacroAvgf1}\n\tMacro average support: {self.MacroAvgSupport}\n\tWeighted average precision: {self.WeightAvgPrecision}\n\tWeighted average recall: {self.WeightedAvgRecall}\n\tWeighted average f1 score: {self.WeightedAvgf1}\n\tWeighted average support: {self.WeightedAvgSupport}"









class FeatureExtractors(Enum):
    
    Gray_Patches = 0
    Colour_Patches = 1
    Gray_SIFT = 2
    Colour_SIFT = 3
    Gray_SIFT_Sparse = 4

@numba.jit(nopython=True)
def Expectation(P_TopicGivenWordDocument, P_TopicGivenDocument, P_WordGivenTopic):
    NTopics = P_TopicGivenDocument.shape[0]
    NWords = P_WordGivenTopic.shape[0]
    NDocs = P_TopicGivenDocument.shape[1]
    for i in prange(NDocs):
        for j in range(NWords):
            divisor = 0
            for k in range(NTopics):
                divisor += P_WordGivenTopic[j,k]*P_TopicGivenDocument[k,i]
            if divisor == 0:
                for k in range(NTopics):
                    P_TopicGivenWordDocument[k,i,j] = 0
            else:
                for k in range(NTopics):
                    P_TopicGivenWordDocument[k,i,j] =  P_WordGivenTopic[j,k]*P_TopicGivenDocument[k,i]/divisor
    return P_TopicGivenWordDocument

@numba.jit(nopython=True)
def Maximize_TopicGivenDocument(Old_P_TopicGivenDocument, Co_OccurenceTable, P_TopicGivenDocumentWord):
    NTopics = P_TopicGivenDocumentWord.shape[0]
    NWords = P_TopicGivenDocumentWord.shape[2]
    NDocs = P_TopicGivenDocumentWord.shape[1]
    N = np.sum(Co_OccurenceTable, axis=0)
    #print(Old_P_TopicGivenDocument.shape)
    #print(P_TopicGivenDocumentWord.shape)
    #print(f"NTopcis:{NTopics}")
    #print(f"NWords:{NWords}")
    #print(f"NDocs:{NDocs}")
    for i in prange(NDocs):
        for k in range(NTopics):
            Old_P_TopicGivenDocument[k,i] = 0
            for j in range(NWords):
                Old_P_TopicGivenDocument[k,i] += Co_OccurenceTable[j,i]*P_TopicGivenDocumentWord[k,i,j]

            Old_P_TopicGivenDocument[k,i]/N[i]
    return Old_P_TopicGivenDocument

@numba.jit(nopython=True)
def Maximize_WordGivenTopic(Old_P_WordGivenDocument, Co_OccurenceTable, P_TopicGivenDocumentWord, NormalizationFactors):
    NTopics = P_TopicGivenDocumentWord.shape[0]
    NWords = P_TopicGivenDocumentWord.shape[2]
    NDocs = P_TopicGivenDocumentWord.shape[1]
    for k in range(NTopics):
        NormalizationFactors[k] = 0
    for m in prange(NWords):
        for i in range(NDocs):
            for k in range(NTopics):
                NormalizationFactors[k] += Co_OccurenceTable[m,i]*P_TopicGivenDocumentWord[k,i,m]
    
    for j in prange(NWords):
        for k in range(NTopics):
            partial_sum = 0
            if NormalizationFactors[k] == 0:
                Old_P_WordGivenDocument[j,k] = 0
                continue
            for i in range(NDocs):
                partial_sum += Co_OccurenceTable[j,i]*P_TopicGivenDocumentWord[k,i,j]
            Old_P_WordGivenDocument[j,k] = partial_sum/NormalizationFactors[k]
    return Old_P_WordGivenDocument


@numba.jit(nopython=True)
def logLiklihood_jit(P_TopicGivenDocument, Co_OccurenceTable, P_TopicGivenDocumentWord, P_WordGivenTopic) -> float:
    ll = 0.0
    NTopics = P_TopicGivenDocumentWord.shape[0]
    NWords = P_TopicGivenDocumentWord.shape[2]
    NDocs = P_TopicGivenDocumentWord.shape[1]
    for N in prange(NDocs):
        for M in range(NWords):
            partial_sum = 0.0
            for K in range(NTopics):
                Nan_test = P_TopicGivenDocumentWord[K, N, M]*np.log(P_WordGivenTopic[M, K] * P_TopicGivenDocument[K,N])
                if not cmath.isnan(Nan_test):
                    partial_sum += Nan_test 

            ll += partial_sum * Co_OccurenceTable[N,M]
    return ll



class PLSA:
    def __init__(self, 
                    BaseImageListPath:str = "./dataset/filelist/places365_train_standard.txt", 
                    BaseDatasetPath:str="./dataset/data",  
                    NumVisualWords: int=1500, 
                    NumTopics: int=25, 
                    NumNeighbors:int = 10,
                    NumCategories:int = 365, 
                    TrainSize:int = 1000,
                    TestSize: int = 1000,
                    SamplingMethod = SamplingMethod.Deterministic,
                    ImageColour = Colour.GRAY,
                    ImageTransform: Any = None,
                    Eps = 0.000001,
                    MaxIter = 1000
                ):
        self.SamplingMethod = SamplingMethod
        self.NumCategories = NumCategories
        self.NumVisualWords = NumVisualWords
        self.NumTopics = NumTopics
        self.NumNeighbors = NumNeighbors

        self.Eps = Eps
        self.MaxIter = MaxIter

        self.TrainingDataset: TrainDataset
        self.TestingDataset: TestDataset
        self.TrainingDataset, self.TestingDataset = LoadDatasets(BaseImageListPath, BaseDatasetPath, TrainSize, TestSize, NumCategories, SamplingMethod, ImageColour, ImageTransform)
        self.Extractor = MakeGrayPatchExtractor() 





        self.KNNClassifier = neighbors.KNeighborsClassifier(n_neighbors=NumNeighbors)

        self.ImageLabels = np.zeros((1,1))
        self.WordCenters = np.zeros((1,1))

        
        
        # Axis 0 - Number of unique words
        # Axis 1 - Number of topics
        self.P_WordGivenTopic = np.full((NumVisualWords, NumTopics), 1/NumVisualWords, dtype="float64")
        #A uniform distribution is assumed at first so no additional normalization is required




    def train(self):
        print("Beginning training")
        AllVisualWords = np.zeros((1,1))
        NumberOfImageFeatures = []
        self.ImageLabels = np.zeros((len(self.TrainingDataset),))
        numFeatures = -1
        feature_idx = 0
        image_idx = 0
        print("Extracting all image features")
        for img in self.TrainingDataset:
            vWords = self.Extractor(img[0])
            self.ImageLabels[image_idx] = img[1]
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
        print("Completed feature extraction, beginning cluster visual words")
        labels  = self.cluster(AllVisualWords[0:feature_idx,:])
        print("Done clustering beginning EM")



        img_index = 0
        img_word_index = 0

        Co_OccurenceTable = np.zeros((self.NumVisualWords, len(self.TrainingDataset)), dtype="uint32")
        for label in labels:
            Co_OccurenceTable[label, img_index] += 1
            img_word_index += 1
            if img_word_index == NumberOfImageFeatures[img_index]:
                img_word_index = 0
                img_index += 1
        




        # Axis 0 - Number of topics
        # Axis 1 - Length of training set or number of documents
        P_TopicGivenDocument = np.random.random((self.NumTopics, len(self.TrainingDataset)))
        #Normalize the conditional distribution 
        P_TopicGivenDocument = P_TopicGivenDocument/np.sum(P_TopicGivenDocument, axis=0)


        # Axis 0 - number of topics
        # Axis 1 - number of documents
        # Axis 2 - number of words
        P_TopicGivenDocumentWord = np.random.random((self.NumTopics, len(self.TrainingDataset), self.NumVisualWords))
        P_TopicGivenDocumentWord = P_TopicGivenDocumentWord/ np.sum(P_TopicGivenDocumentWord, axis = 0)

        #Beginning EM maximization

        NormalizationFactors = np.zeros((self.NumTopics,), dtype="float64")

        Old_logLiklihood = -sys.float_info.max
        New_logLiklihood = 0.0
        for iteration in range(self.MaxIter): 
            New_logLiklihood = logLiklihood_jit(P_TopicGivenDocument, Co_OccurenceTable, P_TopicGivenDocumentWord, self.P_WordGivenTopic)
            if New_logLiklihood - Old_logLiklihood < self.Eps:
                break
            print(f"\t[{iteration}]: delta= {New_logLiklihood - Old_logLiklihood}")
            Old_logLiklihood = New_logLiklihood
            #Expectation portion
            P_TopicGivenDocumentWord = Expectation(P_TopicGivenDocumentWord, P_TopicGivenDocument, self.P_WordGivenTopic)
            #Maximization portion
            P_TopicGivenDocument = Maximize_TopicGivenDocument(P_TopicGivenDocument, Co_OccurenceTable, P_TopicGivenDocumentWord)
            self.P_WordGivenTopic = Maximize_WordGivenTopic(self.P_WordGivenTopic, Co_OccurenceTable, P_TopicGivenDocumentWord, NormalizationFactors)
        print("Done EM training KNN classifier")
        self.KNNClassifier.fit(P_TopicGivenDocument.T, self.ImageLabels)
        print("Done training")
        

        
    def calculate_Z_vector(self, image):
        vWords = self.Extractor(image)
        #Z vector is equivalent to a portion of the topic specific distribution given the document

        #the distribution of the words given the topics is the distribution that was fitted to the training data        
        P_TopicGivenDocumentWord = np.random.random((self.NumTopics, 1, self.NumVisualWords))
        P_TopicGivenDocumentWord = P_TopicGivenDocumentWord/ np.sum(P_TopicGivenDocumentWord, axis = 0)


        P_TopicGivenDocument = np.random.random((self.NumTopics, 1))
        P_TopicGivenDocument  = P_TopicGivenDocument/ np.sum(P_TopicGivenDocument, axis=0)


        #Equivalent to the co-occurence table
        WordOccurrence = np.zeros((self.NumVisualWords,1))
        for feature in vWords:
            WordOccurrence[self.classify_word(feature)] += 1

        Old_logLiklihood = -sys.float_info.max
        New_logLiklihood = 0.0
        for iteration in range(self.MaxIter):
            New_logLiklihood = logLiklihood_jit(P_TopicGivenDocument, WordOccurrence, P_TopicGivenDocumentWord, self.P_WordGivenTopic )
            if New_logLiklihood - Old_logLiklihood < self.Eps:
                break
            Old_logLiklihood = New_logLiklihood
            P_TopicGivenDocumentWord = Expectation(P_TopicGivenDocumentWord, P_TopicGivenDocument, self.P_WordGivenTopic)
            P_TopicGivenDocument = Maximize_TopicGivenDocument(P_TopicGivenDocument, WordOccurrence, P_TopicGivenDocumentWord)
        return P_TopicGivenDocument.T




    def classify_word(self, feature):
        word_class = -1
        min_distance = sys.float_info.max
        for i in range(self.NumVisualWords):
            distance = np.linalg.norm(self.WordCenters[i,:] - feature, ord=2)
            if distance < min_distance:
                word_class = i
                min_distance = distance
        return word_class


    def classify_image(self, image):
        Z = self.calculate_Z_vector(image)
        img_category = self.KNNClassifier.predict(Z)
        return img_category

        
    def test_PLSA(self):
        print("Beginning to test the model")
        predictedLabels = []
        GTLabels = []
        image_idx = 0
        for img, GT_Label in self.TestingDataset:
            predictedLabels.append(self.classify_image(img)) 
            GTLabels.append(GT_Label) 
        labels = [x for x in range(self.NumCategories)]
        print("Done classifying and beginning evaluation")
        report =   metrics.classification_report(GTLabels, predictedLabels,labels=labels, output_dict=True, zero_division=1)
        support = 0
        for k,v in report.items():
            if isinstance(v, dict):
                support += v['support']
        summaryReport = SummaryPerformanceMetrics(\
        report['accuracy'],\
        support,\
        report['macro avg']['precision'],\
        report['macro avg']['recall'],\
        report['macro avg']['f1-score'],\
        report['macro avg']['support'],\
        report['weighted avg']['precision'],\
        report['weighted avg']['recall'],\
        report['weighted avg']['f1-score'],\
        report['weighted avg']['support']\
        )
        print(summaryReport)

            
        
    def cluster(self, data)->List[int]:
        c = cluster.MiniBatchKMeans(n_clusters=self.NumVisualWords, init='k-means++', init_size=self.NumVisualWords*3,max_iter=100).fit(data)
        self.WordCenters = c.cluster_centers_
        return c.labels_
        



def runModel():
    model = PLSA(
        BaseImageListPath = config.BaseImageListPath,
        BaseDatasetPath= config.BaseDatasetPath ,  
        NumVisualWords=config.NumVisualWords, 
        NumTopics=config.NumTopics, 
        NumNeighbors = config.NumNeighbors,
        NumCategories = config.NumCategories, 
        TrainSize = config.TrainSize,
        TestSize = config.TestSize,
        SamplingMethod = config.SampleMethod,
        ImageColour = config.ImageColour,
        ImageTransform = config.ImageTransform,
        Eps = config.Eps,
        MaxIter = config.MaxIter
    )
    model.train()
    model.test_PLSA()

if __name__ == "__main__":
    runModel()







