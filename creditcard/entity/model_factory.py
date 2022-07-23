from creditcard.exception import CreditCardException
from creditcard.logger import logging
from creditcard.entity.artifact_entity import *
from creditcard.entity.config_entity import *
from creditcard.util.util import *

import numpy as np
import sys, os
import importlib
from typing import List

GRID_SEARCH_KEY = 'grid_search'
MODULE_KEY = 'module'
CLASS_KEY = 'class'
PARAM_KEY = 'params'
MODEL_SELECTION_KEY = 'model_selection'
SEARCH_PARAM_GRID_KEY = "search_param_grid"

InitializedModelDetail = namedtuple("InitializedModelDetail",
                                    ["model_serial_number", "model", "param_grid_search", "model_name"])

GridSearchedBestModel = namedtuple("GridSearchedBestModel", ["model_serial_number",
                                                             "model",
                                                             "best_model",
                                                             "best_parameters",
                                                             "best_score",
                                                             ])

BestModel = namedtuple("BestModel", ["model_serial_number",
                                     "model",
                                     "best_model",
                                     "best_parameters",
                                     "best_score", ])

MetricInfoArtifact = namedtuple("MetricInfoArtifact",
                                ["model_name", "model_object", "train_rmse", "test_rmse", "train_accuracy",
                                 "test_accuracy", "model_accuracy", "index_number"])



class ModelFactory:
    def __init__(self, model_config_file_path:str = None):
        try:
            self.config:dict = read_yaml_file(file_path=model_config_file_path)
            
            self.grid_search_cv_module = self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_cv_class = self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_cv_property_data:dict = dict(self.config[GRID_SEARCH_KEY][PARAM_KEY])
            
            self.models_initialization_config:dict = dict(self.config[MODEL_SELECTION_KEY])
            
            self.initialized_model_list = None
            self.grid_searched_best_model_list = None
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    @staticmethod
    def update_property_of_class(instance_ref:object, property_data:dict)->object:
        try:
            if not isinstance(property_data, dict):
                raise Exception("Property data parameter required to be a dictionary")
            logging.info(f"The propery values that going to be update is this {property_data.items()}")
            for key, value in property_data.items():
                setattr(instance_ref, key, value)
            return instance_ref
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    @staticmethod
    def read_params(config_path: str) -> dict:
        try:
            with open(config_path) as yaml_file:
                config:dict = yaml.safe_load(yaml_file)
            return config
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    @staticmethod
    def class_for_name(module_name:str, class_name:str):
        """
        Args:
            module_name (str): _description_
            class_name (str): _description_
        Description:
            It import the class from specific module 
        Raises:
            HousingException: _description_
        Returns:
            _type_: _description_
        """
        try:
            # load the module, will raise ImportError if module cannot be loaded
            module = importlib.import_module(module_name)
            # get the class, will raise AttributeError if class cannot be found
            logging.info(f"Executing code to import the {class_name} from module {module}")
            class_ref = getattr(module, class_name)
            return class_ref
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def execute_grid_search_operation(self, initialized_model: InitializedModelDetail,
                                      input_feature, output_feature)->GridSearchedBestModel:
        """
        excute_grid_search_operation(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_grid: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return GridSearchOperation object
        """
        
        try:
            grid_search_cv_ref = ModelFactory.class_for_name(module_name = self.grid_search_cv_module,
                                                             class_name = self.grid_search_cv_class)
            grid_search_cv = grid_search_cv_ref(estimator= initialized_model.model,
                                                param_grid = initialized_model.param_grid_search)
            grid_search_cv = ModelFactory.update_property_of_class(instance_ref=grid_search_cv,
                                                                   property_data=self.grid_search_cv_property_data)
            
            message = f'{">>"* 30} f"Training {type(initialized_model.model).__name__} Started." {"<<"*30}'
            logging.info(message)
            grid_search_cv.fit(input_feature, output_feature)
            message = f'{">>"* 30} f"Training {type(initialized_model.model).__name__}" completed {"<<"*30}'
            logging.info(message)
            grid_search_best_model = GridSearchedBestModel(model_serial_number=initialized_model.model_serial_number,
                                                           model=initialized_model.model,
                                                           best_model=grid_search_cv.best_estimator_,
                                                           best_parameters=grid_search_cv.best_params_,
                                                           best_score=grid_search_cv.best_score_
                                                          )
            return grid_search_best_model 
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def get_initialized_model_list(self)->List[InitializedModelDetail]:
        """
        From the model_initialization_config it will extract the dictionary .
        It will loop through te dictionary and create the model object then it will append this in a list .
        This function will return a list of model details.
        return List[ModelDetail]
        """
        try:
            initialized_model_list = []
            for model_serial_number in self.models_initialization_config.keys():
                model_initialization_config = self.models_initialization_config[model_serial_number]
                model_object_ref = ModelFactory.class_for_name(module_name=model_initialization_config[MODULE_KEY],
                                                               class_name=model_initialization_config[CLASS_KEY]
                                                               )
                model = model_object_ref()
                
                if PARAM_KEY in model_initialization_config:
                    model_obj_property_data = dict(model_initialization_config[PARAM_KEY])
                    model = ModelFactory.update_property_of_class(instance_ref=model,
                                                                  property_data=model_obj_property_data)
                param_grid_search = model_initialization_config[SEARCH_PARAM_GRID_KEY]
                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}"
                model_initialization_config = InitializedModelDetail(model_serial_number=model_serial_number,
                                                                     model=model,
                                                                     param_grid_search=param_grid_search,
                                                                     model_name=model_name)
                initialized_model_list.append(model_initialization_config)
            self.initialized_model_list = initialized_model_list
            return self.initialized_model_list
        except Exception as e:
            raise CreditCardException(e, sys) from e 
        
    def initiate_best_parameter_search_for_initialized_model(self, initialized_model: InitializedModelDetail,
                                                             input_feature,
                                                             output_feature) -> GridSearchedBestModel:
        """
        initiate_best_model_parameter_search(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_grid: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return a GridSearchOperation
        """
        try:
            return self.execute_grid_search_operation(initialized_model=initialized_model, 
                                                      input_feature=input_feature,
                                                      output_feature=output_feature)
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    def initiate_best_parameter_search_for_initialized_models(self, initialized_model_list: List[InitializedModelDetail],
                                                             input_feature,
                                                             output_feature) -> List[GridSearchedBestModel]:
        """
        initiate_best_model_parameter_search(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_grid: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return a GridSearchOperation
        """
        try:
           self.grid_searched_best_model_list = []
           for initialized_model_detail in initialized_model_list:
               grid_searched_best_model :GridSearchedBestModel = self.execute_grid_search_operation(
                                                                 initialized_model=initialized_model_detail, 
                                                                 input_feature=input_feature,
                                                                 output_feature=output_feature
                                                                 )
               self.grid_searched_best_model_list.append(grid_searched_best_model)
           
           return self.grid_searched_best_model_list
        except Exception as e:
            raise CreditCardException(e, sys) from e
        
    @staticmethod
    def get_model_detail(model_details:List[InitializedModelDetail],
                         model_serial_number:str)->InitializedModelDetail:
        try:
            for initialized_model_detail in model_details:
                if initialized_model_detail.model_serial_number == model_serial_number:
                    return initialized_model_detail
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list:List[GridSearchedBestModel],
                                                          base_accuracy = 0.6
                                                          )->BestModel:
        try:
            best_model = None
            for grid_searched_best_model in grid_searched_best_model_list:
                if base_accuracy < grid_searched_best_model.best_score:
                    logging.info(f"Accepting model found : {grid_searched_best_model}")
                    base_accuracy = grid_searched_best_model.best_score
                    best_model = grid_searched_best_model
            if not best_model:
                raise Exception(f"None of Model has base accuracy: {base_accuracy}")
            logging.info(f"Best model: {best_model}")
            return best_model
        except Exception as e:
            raise CreditCardException(e, sys) from e
    
    def get_best_model(self, X, y, base_accuracy = 0.6)->BestModel:
        try:
            logging.info(f"Started initializing model from config file")
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f"Initialized model : {initialized_model_list}")
            grid_searched_best_model_list = self.initiate_best_parameter_search_for_initialized_models(
                initialized_model_list=initialized_model_list,
                input_feature=X,
                output_feature=y
            )
            return ModelFactory.get_best_model_from_grid_searched_best_model_list(
                grid_searched_best_model_list=grid_searched_best_model_list
                )
        except Exception as e:
            raise CreditCardException(e, sys) from e