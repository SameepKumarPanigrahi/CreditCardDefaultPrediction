from creditcard.exception import CreditCardException
from creditcard.logger import logging
from creditcard.entity.artifact_entity import *
from creditcard.entity.config_entity import *
from creditcard.util.util import *
from creditcard.entity.model_factory import *

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
    
class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'=' * 30} Model Trainer log started {'=' * 30}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e :
            raise CreditCardException(e, sys) from e
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info(f"Loading transformed training dataset")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            train_array = load_numpy_array_data(transformed_train_file_path)
            
            logging.info(f"Loading transformed testing dataset")
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path
            test_array = load_numpy_array_data(transformed_test_file_path)
            
            logging.info(f"Splitting Training And Testng input's input feature and output feature")
            X_train, y_train, X_test, y_test = train_array[:, :-1],  train_array[:, -1], test_array[:, :-1], test_array[:, -1]
            
            logging.info(f"Extracting model config file path")
            model_config_file_path = self.model_trainer_config.model_config_file_path
            
            logging.info(f"Initializing model factory class using above model config file: {model_config_file_path}")
            model_factory = ModelFactory(model_config_file_path=model_config_file_path)
            
            base_accuracy = self.model_trainer_config.base_accuracy
            logging.info(f"Expected Base Accuracy is this much : {base_accuracy}")
            
            logging.info("Initaing operation model selection ")
            best_model = model_factory.get_best_model(X= X_train, y=y_train, base_accuracy=base_accuracy)
            logging.info(f"Best Model Found on the training dataset {best_model}")
            
            logging.info(f"Extracting trained model list")
            grid_searched_best_model_list: List[GridSearchedBestModel] = model_factory.grid_searched_best_model_list
            
            model_list = [model.best_model for model in grid_searched_best_model_list]
            logging.info(f"Evaluation all trained model on training and testing dataset both")
            metric_info:MetricInfoArtifact = evaluate_classification_model(model_list=model_list,
                                                                       X_train=X_train,
                                                                       y_train=y_train,
                                                                       X_test=X_test,
                                                                       y_test=y_test,
                                                                       base_accuracy=base_accuracy)
            logging.info(f"Best model found on the training and testing data {metric_info.model_name}")
            
            preprocessing_object = load_object(self.data_transformation_artifact.preprocessed_object_file_path)
            model_object = metric_info.model_object
            
            trained_model_file_path = self.model_trainer_config.trained_model_file_path
            housing_model = CreditCardEstimatorModel(preprocessing_object=preprocessing_object, trained_model_object=model_object)
            logging.info(f"Saving model at path: {trained_model_file_path}")
            save_object(file_path=trained_model_file_path, obj=housing_model)
            
            model_trainer_artifact = ModelTrainerArtifact(is_trained=True,
                                                          message= "Model Trained Successfully",
                                                          trained_model_file_path=trained_model_file_path,
                                                          train_accuracy= metric_info.train_accuracy,
                                                          test_accuracy= metric_info.test_accuracy,
                                                          recall= metric_info.recall,
                                                          precession= metric_info.precession,
                                                          f1_score =  metric_info.f1_score,
                                                          model_accuracy=metric_info.model_accuracy)
            return model_trainer_artifact
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def __del__(self):
        logging.info(f"{'='*30} Model Trainer log completed {'='*30}")