from creditcard.logger import logging
from creditcard.exception import CreditCardException
from creditcard.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from creditcard.entity.config_entity import DataValidationConfig
from creditcard.util.util import *
from creditcard.constants import *

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import sys, os
import pandas as pd
import json

class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            logging.info(f"{'=' *30} Data Validation log started . {'=' *30}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def get_train_and_test_df(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            return train_df, test_df
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def get_and_save_data_drift_report(self):
        try:
           profile = Profile(sections=[DataDriftProfileSection()])
           train_df, test_df = self.get_train_and_test_df()
           profile.calculate(train_df, test_df)
           
           report = json.loads(profile.json())
           report_file_path = self.data_validation_config.report_file_path
           report_dir = os.path.dirname(report_file_path)
           os.makedirs(report_dir, exist_ok=True)
           
           with open(report_file_path, "w") as report_file:
               json.dump(report, report_file, indent=6)
           
           return report
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def save_data_drift_report_page(self):
        try:
            dashbord = Dashboard(tabs=[DataDriftTab()])
            train_df, test_df = self.get_train_and_test_df()
            dashbord.calculate(train_df, test_df)
            
            report_page_file_path = self.data_validation_config.report_page_file_path
            report_page_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_page_dir, exist_ok=True)
            
            dashbord.save(report_page_file_path)
            
        except Exception as e:
            raise CreditCardException(e, sys)
    
    def is_data_drift_found(self) -> bool:
        try:
            report = self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()
            return True
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def validate_dataset_schema(self) -> bool:
        try:
            validation_status = False
            
            schema_file_path = self.data_validation_config.schema_file_path
            schema = read_yaml_file(file_path=schema_file_path)
        
            train_df, test_df = self.get_train_and_test_df()
            
            all_columns_with_data_type = schema[DATASET_SCHEMA_COLUMNS_KEY]
            logging.info(f"Verifying number of column of train and test dataFrame")
           
            #Check the number of column
            if(train_df.shape[1] == len(all_columns_with_data_type) and test_df.shape[1] == len(all_columns_with_data_type)):
                for key, value in all_columns_with_data_type.items():
                    #Check that specific key present in the dataframe as column
                    if (key in train_df and key in test_df):
                        #Check the datatype of the column
                        if (train_df[key].dtypes != value or test_df[key].dtypes != value):
                            logging.info(f"The datatype of of column : {key} is not equal to either trainDataframe or test dataFrame")
                            return validation_status
                    else:
                        logging.info(f"This column : {key} is not present in either train dataframe or test dataframe")
                        return validation_status
            else:
                logging.info(f"Either train data or test data is missing some column ")
                return validation_status
            validation_status = True
            return validation_status 
        except Exception as e:
            raise CreditCardException(e,sys) from e
        
    def is_train_test_file_exist(self) -> bool:
        try:
            logging.info(f"Checking if training and test file is exist or not")
            is_train_file_path_exist = False
            is_test_file_path_exist = False
            
            training_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            
            is_train_file_path_exist = os.path.exists(training_file_path)
            is_test_file_path_exist = os.path.exists(test_file_path)
            
            is_available = is_train_file_path_exist and is_test_file_path_exist
            logging.info(f"Is train and test file path exist ?-> {is_available}")
            
            if not is_available:
                message = f"Training file path : {training_file_path} or testing file path : {test_file_path}" \
                            "is not present ."
                raise Exception(message)
            return is_available
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            self.is_train_test_file_exist()
            self.validate_dataset_schema()
            self.is_data_drift_found()
            
            data_validation_artifact = DataValidationArtifact(
                schema_file_path = self.data_validation_config.schema_file_path,
                report_file_path = self.data_validation_config.report_file_path,
                report_page_file_path = self.data_validation_config.report_page_file_path,
                is_validated = True,
                message = "Data validation performed successfuly"
            )
            return data_validation_artifact
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def __del__(self):
        logging.info(f"{'='*30} Data Validation log completed {'='*30}")