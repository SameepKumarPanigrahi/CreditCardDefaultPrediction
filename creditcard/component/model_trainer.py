from creditcard.exception import CreditCardException
from creditcard.logger import logging
from creditcard.entity.artifact_entity import *
from creditcard.entity.config_entity import *
from creditcard.util.util import *

class CreditCardEstimatorModel:
    def __init__(self, preprocessing_object, trained_model_object):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_object: trained_model_object
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
        
    def predict(self, X):
        transformed_feature = self.preprocessing_object.transform(X)
        return self.trained_model_object.predict(transformed_feature)
    
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"