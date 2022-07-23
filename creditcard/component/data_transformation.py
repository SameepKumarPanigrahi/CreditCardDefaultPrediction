from ast import Pass
from distutils.log import info
from creditcard.exception import CreditCardException
from creditcard.logger import logging
from creditcard.entity.artifact_entity import *
from creditcard.entity.config_entity import *
from creditcard.util.util import *

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys, os
import scipy.stats as stat

class FeatureGenerator(BaseEstimator, TransformerMixin):

    def __init__(self, column_to_be_droped, column_needs_to_replace_value,
                 coumn_needs_to_be_transformed_to_normal_distribution)->None:
       self.column_to_be_droped = column_to_be_droped
       self.column_needs_to_replace_value = column_needs_to_replace_value
       self.coumn_needs_to_be_transformed_to_normal_distribution = coumn_needs_to_be_transformed_to_normal_distribution

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            #Drop the Id column from the Dataframe
            X = X.drop([self.column_to_be_droped], axis=1)
            #Change the SEX column value previously it was (1=male, 2=female) we will convert this to (1=male, 0=female)
            X[self.column_needs_to_replace_value] = X[self.column_needs_to_replace_value].map({2:0, 1:1})
            
            #Apply Box cox transformation to get a data in normal distribution
            for column in self.coumn_needs_to_be_transformed_to_normal_distribution:
                if X[column].min() < 0:
                     X.loc[X[column] < 0, column] = 0
                transformed_column_value, param = stat.boxcox(X[column]+1)
                X[column] = transformed_column_value
            return X
        except Exception as e:
            raise CreditCardException(e, sys) from e


class DataTransformation:
    def __init__(self, data_transformation_config:DataTransformationConfig,
                 data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_artifact:DataValidationArtifact) -> None:
        try:
            logging.info(f"{'='*30} Data Transformation log startred {'='*30}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def get_data_transformer_object(self)->ColumnTransformer:
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path
            dataset_schema = read_yaml_file(file_path=schema_file_path)
            
            column_to_be_droped = dataset_schema[COLUMN_TO_BE_DROPED]
            column_needs_to_replace_value = dataset_schema[COLUMN_NEEDS_TO_REPLACE_VALUE]
            coumn_needs_to_be_transformed_to_normal_distribution = dataset_schema[COUMN_NEEDS_TO_BE_TRANSFORMED_TO_NORMAL_DISTRIBUTION]
            logging.info(f"Column to be droped : {column_to_be_droped}")
            logging.info(f"Column needs to replace value : {column_needs_to_replace_value}")
            logging.info(f"Column needs to be transformed to normal distribution : {coumn_needs_to_be_transformed_to_normal_distribution}")
            
            pipeline = Pipeline(steps=[
                ('feature_generator', FeatureGenerator(
                    column_to_be_droped=column_to_be_droped,
                    column_needs_to_replace_value=column_needs_to_replace_value,
                    coumn_needs_to_be_transformed_to_normal_distribution=coumn_needs_to_be_transformed_to_normal_distribution)),
                ('sclar', StandardScaler())
            ])
            preprocessing = ColumnTransformer([
                ("pipeline", pipeline, dataset_schema[ALL_FEATURE_COLUMNS])
            ])
            return preprocessing
        except Exception as e:  
            raise CreditCardException(e, sys) from e
    
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info(f"Obtaining Preprocessing Object")
            preprocessing_object = self.get_data_transformer_object()
            
            logging.info(f"Obtaining the training and testing filr path")
            training_file_path = self.data_ingestion_artifact.train_file_path
            testing_file_path = self.data_ingestion_artifact.test_file_path
            
            schema_file_path = self.data_validation_artifact.schema_file_path
            
            logging.info(f"Loading training data as Pandas Dataframe.")
            train_df = load_data(file_path = training_file_path, schema_file_path = schema_file_path)
            test_df = load_data(file_path = testing_file_path, schema_file_path = schema_file_path)
            
            schema = read_yaml_file(file_path= schema_file_path)
            target_column_name = schema[TARGET_COLUMN_KEY]
            
            logging.info(f"Splitting input and target feature from training and testing dataframe")
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir
            
            train_file_name = os.path.basename(training_file_path).replace(".csv", ".npz")
            test_file_name = os.path.basename(testing_file_path).replace(".csv", ".npz")
            
            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)
            
            logging.info(f"Saving transformed training and testing array.")
            logging.info(f"Start Saving the transformed train file in the: {transformed_train_dir} and tranasformed test file in the {transformed_test_dir}")
            save_numpy_array_data(transformed_train_file_path, train_arr)
            save_numpy_array_data(transformed_test_file_path, test_arr)
            
            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path
            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path, obj=preprocessing_object)
            
            data_transformation_artifact = DataTransformationArtifact(
                message="Data Transformed Successfully",
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path= transformed_test_file_path,
                preprocessed_object_file_path=preprocessing_obj_file_path,
                is_transformed= True
            )
            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact 
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def __del__(self):
        logging.info(f"{'='*30} Data Transformation log completed {'='*30}")