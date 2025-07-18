#main purpose of data ingestion is to load data from various sources.And this is the entry point of data pipeline because it has __main__ in the end.


import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig


@dataclass
class DataIngestionConfig:
    """here we are defining all the path variables for data ingestion"""
    train_data_path: str = os.path.join("artifact", "train_data.csv")
    test_data_path: str = os.path.join("artifact", "test_data.csv")
    raw_data_path: str = os.path.join("artifact", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info( "Data ingestion initiated")
        try:
            df= pd.read_csv('notebook\data\stud.csv')
            logging.info("read dataset and saved in data frame")

            log_dir_path=os.path.dirname(self.ingestion_config.train_data_path)
            os.makedirs(log_dir_path,exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("data saved in raw data path")

            train_set,test_set= train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index= False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index= False, header=True)
            logging.info("data saved in train and test data path")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
                            



        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    #calling the function from above and storing the return values in train_path and test_path
    train_path,test_path = obj.initiate_data_ingestion() 
    #creating an object of datatransformation class
    datatransformation = DataTransformation() 
    #calling the function of datatransformation class and passing the train_path and test_path
    train_arr,test_arr,_=datatransformation.initiate_data_transformation(train_path,test_path) 
    #creating an object of modeltrainer class
    modeltrainer= ModelTrainer()
    #calling the function of modeltrainer class and passing the train_arr and test_arr
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))