from creditcard.logger import logging
from creditcard.exception import CreditCardException
from creditcard.entity.config_entity import *
from creditcard.entity.artifact_entity import *
from creditcard.config.configuration import *
from creditcard.component.data_ingestion import *
from creditcard.component.data_validation import *

import os, sys

class Pipeline:
    def __init__(self, config: Configuration = Configuration()):
        try:
            self.config = config
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def start_data_ingestion(self)-> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config = self.config.get_data_ingestion_config())
            
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config = self.config.get_data_validation_config(),
                                             data_ingestion_artifact = data_ingestion_artifact)
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            
            # data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            # data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact, data_validation_artifact)
            # model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            # model_evaluation_artifact = self.start_model_evaluation(data_transformation_artifact=data_transformation_artifact,
            #                                                         model_trainer_artifact=model_trainer_artifact)
            # model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
        except Exception as e:
            raise CreditCardException(e, sys) from e
    