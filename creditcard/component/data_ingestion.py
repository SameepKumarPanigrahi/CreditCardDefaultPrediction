import shutil
from tkinter import E
from creditcard.logger import logging
from creditcard.exception import CreditCardException
from creditcard.entity.config_entity import DataIngestionConfig
from creditcard.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd
import numpy as np
import sys, os
import filecmp

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"{'=' *30} Data Ingestion log started . {'=' *30}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def copy_creditcard_data(self,):
        try:
            # Extract the training file path
            training_file_path = self.data_ingestion_config.training_file_path
            logging.info(f"Training file path is this one : {training_file_path}")
             
            #Folder location to download the file 
            training_file_name = self.data_ingestion_config.training_file_name
            logging.info(f"Training file name is this one : {training_file_name}")
            
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            logging.info(f"Raw data directory is this one : {raw_data_dir}")
            
            if os.path.exists(raw_data_dir):
                 os.remove(raw_data_dir)
            os.makedirs(raw_data_dir, exist_ok= True)
            
            training_files = os.listdir(path=training_file_path)
            logging.info(f" This is trianing fiel : {training_files}")
            for train_file in training_files:
                if training_file_name == train_file:
                    source_path = os.path.join(training_file_path, train_file)
                    destination_path = os.path.join(raw_data_dir, train_file)
                    shutil.copy(src=source_path, dst=destination_path)
            
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            #Extract the folder where the file is there
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            
            #Extract the file name in this folder only one file will be present
            file_name = os.listdir(raw_data_dir)[0]
            
            creditcard_file_path = os.path.join(raw_data_dir, file_name)
            logging.info(f"Reading csv file : [{creditcard_file_path}]")
            creditcard_data_frame = pd.read_csv(creditcard_file_path)
            
            #As its a classification problem so we can use target column for stratified split
            target_column = creditcard_data_frame.columns[-1]
            logging.info(f"Target column is this one : {target_column}")
            
            logging.info(f"Splitting data into train and test")
            start_train_set = None
            start_test_set = None
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state= 42)
            
            for train_index, test_index in split.split(creditcard_data_frame, creditcard_data_frame[target_column]):
                start_train_set = creditcard_data_frame.loc[train_index]
                start_test_set = creditcard_data_frame.loc[test_index]
            
            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir, file_name)
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir, file_name)
            
            if start_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir, exist_ok=True)
                logging.info(f"Exporting training dataset to file : [{train_file_path}]")
                start_train_set.to_csv(train_file_path, index=False)
            
            if start_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok=True)
                logging.info(f"Exporting test dataset to file : [{test_file_path}]")
                start_test_set.to_csv(test_file_path, index=False)
            
            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                                            test_file_path=test_file_path,
                                                            is_ingested=True,
                                                            message=f"Data ingestion completed Successfully.")
            logging.info(f"Data Ingestion artifact is this one : [{data_ingestion_artifact}]")
            return data_ingestion_artifact
            
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            self.copy_creditcard_data()
            return self.split_data_as_train_test()
        
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def __del__(self):
        logging.info(f"{'='*30} Data Ingestion log completed {'='*30}")
               