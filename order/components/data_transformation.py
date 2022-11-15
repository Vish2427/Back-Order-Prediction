import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,LabelEncoder
from sklearn.pipeline import Pipeline
import miceforest as mf
from order.constant.training_pipeline import SCHEMA_FILE_PATH
from sklearn.compose import ColumnTransformer
from order.constant.training_pipeline import TARGET_COLUMN
from order.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from order.entity.config_entity import DataTransformationConfig
from order.exception import OrderException
from order.logger import logging
from order.ml.model.estimator import TargetValueMapping
from order.utils.main_utils import save_numpy_array_data, save_object,read_yaml_file
from sklearn.base import BaseEstimator,TransformerMixin


class MICEimputer(BaseEstimator, TransformerMixin):

    def __init__(self,columns=None):
        self.columns = columns  

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            kernel = mf.ImputationKernel(X,save_all_iterations=True,random_state=1989)
            X_mice = kernel.complete_data()
            return X_mice
        except Exception as e:
            raise OrderException(e, sys) from e

class label_encoder(BaseEstimator, TransformerMixin):

    def __init__(self,columns=None):
        self.columns = columns  

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            label= LabelEncoder()
            t = [['Yes'], ['No']]
            label.fit(t)
            for col in self.columns:
                if col!='sku':
                    X[col]=label.transform(X[col])
            return X
        except Exception as e:
            raise OrderException(e, sys) from e
    
    
class DataTransformation:
    def __init__(self,data_validation_artifact: DataValidationArtifact, 
                    data_transformation_config: DataTransformationConfig,):
        """

        :param data_validation_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            logging.info('Starting data transformation')
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise OrderException(e, sys) from e


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise OrderException(e, sys) from e


    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            dataset_schema = read_yaml_file(SCHEMA_FILE_PATH)

            numerical_columns = dataset_schema['numerical_columns']
            categorical_columns = dataset_schema['categorical_columns']

            num_pipeline = Pipeline(steps=[
                ('imputer', MICEimputer()), 
                
            ])
            cat_pipeline = Pipeline(steps=[
                ('Label',label_encoder(columns=categorical_columns)) ,
                ('impute', SimpleImputer(strategy="most_frequent"))
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            def column_concat():
                concat = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns),
                        ])
                return concat

            preprocessor = Pipeline(steps=[
                ('column_transform',column_concat()),
                ('scalar',RobustScaler())
            ])
            
            return preprocessor

        except Exception as e:
            raise OrderException(e, sys) from e

    
    def initiate_data_transformation(self,) -> DataTransformationArtifact:
        try:
            logging.info('Reading train and test dataframe')
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            
            logging.info("Initilizing preprocessing object")
            preprocessor = self.get_data_transformer_object()

            logging.info(("seprating target and feature columns from train and test"))
            #training dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            
            logging.info('Encoing train target values')
            target_feature_train_df= target_feature_train_df.replace({'Yes': 1, 'No': 0})
            #target_feature_train_df = target_feature_train_df.replace( TargetValueMapping().to_dict())
            
            #target_feature_train_df = np.array(target_feature_train_df).astype('int')
            #testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            

            logging.info('Encoing test target values')
            target_feature_test_df= target_feature_test_df.replace({'Yes': 1, 'No': 0})
            #target_feature_test_df = target_feature_test_df.replace(TargetValueMapping().to_dict())
            #target_feature_test_df = np.array(target_feature_test_df).astype('int')

            logging.info('Fit and transform train and test files')
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature =preprocessor_object.transform(input_feature_test_df)

            logging.info('balancing train and test using SMOTETomek strategy = minority')
            smt = SMOTETomek(sampling_strategy="minority")

            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, np.ravel(target_feature_train_df)
            )

            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                transformed_input_test_feature, np.ravel(target_feature_test_df)
            )
            logging.info('Concat target and feature column of train and test')
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final) ]
            test_arr = np.c_[ input_feature_test_final, np.array(target_feature_test_final) ]

            logging.info(f'Saving numpy array of train at {self.data_transformation_config.transformed_train_file_path} \
                                        and test at {self.data_transformation_config.transformed_test_file_path}')
            #save numpy array data
            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            logging.info(f'saving preprocessor object at {self.data_transformation_config.transformed_object_file_path}')
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)
            
            
            #preparing artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise OrderException(e, sys) from e

