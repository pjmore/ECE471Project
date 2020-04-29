from __future__ import annotations
from typing import * 
from sklearn.metrics import classification_report # type:ignore
from DataLoader import ImageClass, SamplingMethod,Places365Dataset, DataPoint, prettystr, max_enum_idx
from random import randint 

class MisMatchedDimensionsError(Exception):
    def __init__(self, message:str, left:List, right:List ):
        self.message:str = message
        self.len_left = len(left)
        self.len_right = len(right)

    def __str__(self)->str:
        return f"The length of the left list was {self.len_left} while the length of the right was {self.len_right}: {self.message}"



def check_len(left:List, right:List, message:str):
    if len(left) != len(right):
        raise MisMatchedDimensionsError(message, left, right)
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


class Predictions(NamedTuple):
    Predicted: List[ImageClass]
    Actual: List[ImageClass]
         
    
    def add_prediction(self, predicted:Optional[Union[List[ImageClass],ImageClass]], actual:Optional[Union[List[ImageClass],ImageClass]])->None:
        #check_len(predicted, actual, "Error adding to Predictions")
        if isinstance(predicted, list) and isinstance(actual, list):
            check_len(predicted, actual, "Error creating Predictions")
            self.Predicted.extend(predicted)
            self.Actual.extend(actual)
        elif isinstance(predicted, ImageClass) and isinstance(actual, ImageClass):
            self.Predicted.append(predicted)
            self.Actual.append(actual)
        elif predicted is None and actual is None:
            pass
        else:
            raise TypeError(f"Expected actual and predicted to be the same type of List[ImageClass], ImageClass, or None. Found predicted as {str(type(predicted))} and actual as {str(type(actual))}")


    def __str__(self)->str:
        return f"Predictions:\n\tActual:{[prettystr(x) for x in self.Actual]}\n\tPredicted{[prettystr(x) for x in self.Predicted]}\n"

    
    def predictions_report_dict(self)->Dict:
        target_names=[prettystr(ImageClass(x)) for x in range(0, max_enum_idx()+1)]
        labels=[int(x) for x in range(0,max_enum_idx()+1)]
        true = [int(x) for x in self.Actual]
        pred = [int(x) for x in self.Predicted]
        return classification_report(true, pred,labels=labels, target_names=target_names, output_dict=True, zero_division=1)

    def predictions_report_print(self)->None:
        target_names=[prettystr(ImageClass(x)) for x in range(0, max_enum_idx()+1)]
        labels=[int(x) for x in range(0,max_enum_idx()+1)]
        true = [int(x) for x in self.Actual]
        pred = [int(x) for x in self.Predicted]
        report = classification_report(true, pred,labels=labels, target_names=target_names, output_dict=False)
        print(report)
    
    def predictions_to_summary_metrics(self)->SummaryPerformanceMetrics:
        report = self.predictions_report_dict()
        support:int = 0
        for k,v in report.items():
            if isinstance(v, dict):
                support += v['support']
            else:
                print(f"{k}={v}")
        metrics = SummaryPerformanceMetrics(\
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
        return metrics


         

#

#Accuracy: {self.Accuracy}\n\tSupport: {self.Support}\n\tMacroAvgPrecision: {self.MacroAvgPrecision}\n\tMacroAvgRecall: {self.MacroAvgRecall}\n\tMacroAvgf1: {self.MacroAvgf1}\n\tMacroAvgSupport: {self.MacroAvgSupport}\n\tWeightAvgPrecision: {self.WeightAvgPrecision}\n\tWeightedAvgRecall: {self.WeightedAvgRecall}\n\tWeightedAvgf1: {self.WeightedAvgf1}\n\tWeightedAvgSupport: {self.WeightedAvgSupport}\n\t

def NewPredictions(predicted: Optional[Union[List[ImageClass], ImageClass]]=None, actual:Optional[Union[List[ImageClass], ImageClass]]=None)->Predictions:
    new_p = Predictions([],[])
    new_p.add_prediction(predicted,actual)
    return new_p

class PLSA_classifier:
    def __init__(self):
        self.max_rand = max_enum_idx()

    def classify(self, point:DataPoint)->ImageClass:
        groundtruth: int = randint(0,self.max_rand)
        return ImageClass(groundtruth)

    def train(self, DataPoint)->None:
        return None

def LoadAndUseData(base_path:str="./dataset"):
    loader = Places365Dataset(base_path, 1, 2000,sampling_method=SamplingMethod.Random, shuffle_dataset_after=True)
    test_dataset = loader.testing_data
    train_dataset = loader.training_data
    prediction_tracker = NewPredictions()
    classifier = PLSA_classifier()
    for point in train_dataset:
        classifier.train(point)
    for point in test_dataset:
        class_output = classifier.classify(point)
        prediction_tracker.add_prediction(class_output, ImageClass(point.GroundTruthClass))
    perf_metrics = prediction_tracker.predictions_to_summary_metrics()
    print(perf_metrics)

if __name__ == "__main__":
    LoadAndUseData()
