from sklearn.impute import SimpleImputer # handling missing values
from sklearn.preprocessing import StandardScaler # handling feature scaling
from sklearn.preprocessing import OrdinalEncoder # performing ordinal encoding 

from sklearn.pipeline import Pipeline # create pipelines
from sklearn.compose import ColumnTransformer # combine pipelines

import os, sys
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

# Data transformation config
@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


# Data transformation class

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initiated.")
            # defining categorical and numerical features 
            categorical_cols =['cut','color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x','y','z']

            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Pipeline Initaited')
            # numerical pipeline 
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            # categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ('scaler', StandardScaler())
                ]
            )

            # combining pipelines
            preprocessor =ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            return preprocessor

        except Exception as e:
            logging.info("Error in data transformation")
            raise CustomException(e, sys)

        
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading test and train data completed')
            logging.info(f'Train DataFrame Head : \n {train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head : \n {test_df.head().to_string()}')

            preprocessor_obj = self.get_data_transformation_object()

            target_column_name ='price'
            drop_columns =[target_column_name,'id']

            # dependent and independent features
            input_feature_train_df = train_df.drop(columns= drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns= drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # apply the transformation

            input_feature_train_arr= preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessor_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessor object on training and testing datasets.') 

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            logging.info('Preprocessor pickle is created and saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occured in the initiate_transformation")

            raise CustomException(e, sys)

        
