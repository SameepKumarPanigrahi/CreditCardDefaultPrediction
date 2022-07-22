from creditcard.logger import logging
from creditcard.exception import CreditCardException
from creditcard.entity.config_entity import *
from creditcard.entity.artifact_entity import *
from creditcard.config.configuration import *
from creditcard.component.data_ingestion import *
from creditcard.pipeline.pipeline import *

def main():
    try:
        pipeline = Pipeline()
        pipeline.run_pipeline()
        # data_ingestion_config = Configuration().get_data_ingestion_config()
        # print(data_ingestion_config)
    except Exception as e:
        logging.error(f"{e}")
        print(e)
if __name__=="__main__":
    main()