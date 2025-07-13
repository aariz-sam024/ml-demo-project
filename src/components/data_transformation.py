# main purpose of data transformation is to do feature engineering, data cleaning, filling missing data, categorical to numerical conversion on our data set

import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
 
@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join("artifact","preprocess.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            num_column=["writing_score","reading_score"]
            cat_column=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))

                ]
            )
            logging.info("num and cat pipeline created")

            preprocessor=ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, num_column),
                    ("cat", cat_pipeline, cat_column)
                ]
            )
            return preprocessor
            logging.info("transformer object created")

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df= pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("data loaded")

            logging.info("obtaining transformer object")
            preprocessing_object= self.get_data_transformer_object()

            target_column= "math_score"
            num_column= ["writing_score","reading_score"]

            #here we are basically doing x_train,x_test,y_train,y_test
            input_feature_train_df= train_df.drop(columns = [target_column],axis=1) #x_train
            target_feature_train_df= train_df[target_column] #y_train

            input_feature_test_df=test_df.drop(columns=[target_column],axis=1) #x_test
            target_feature_test_df=test_df[target_column] #y_test

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_object.transform(input_feature_test_df)

            # np.c_[] is a NumPy shortcut for concatenating arrays column-wise (i.e., horizontally).
            # np.c_[A, B] means Combining A and B side by side
            # this is useful for training a model because we need to pass both input and output to the model.
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            logging.info("input feature created")

            save_object(
                file_path=self.transformation_config.preprocess_obj_file_path,
                obj=preprocessing_object
            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocess_obj_file_path
                

            )

        except Exception as e:
            raise CustomException(e,sys) from e
        
            
               