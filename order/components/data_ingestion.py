from order.exception import OrderException
from order.logger import logging
from order.entity.config_entity import DataIngestionConfig
from order.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
import os,sys
from pandas import DataFrame
from order.data_access.order_data import OrderData
from order.utils.main_utils import read_yaml_file
from order.constant.training_pipeline import SCHEMA_FILE_PATH
class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise OrderException(e,sys) from e

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Export mongo db collection record as data frame into feature
        """
        try:
            logging.info("Exporting data from mongodb to feature store")
            order_data = OrderData()
            dataframe = order_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path            

            #creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
        except  Exception as e:
            raise  OrderException(e,sys) from e

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Feature store dataset will be split into train and test file
        """

        try:
            split = StratifiedShuffleSplit(n_splits=1,
                     test_size=self.data_ingestion_config.train_test_split_ratio, 
                     random_state=42
            )
            for train_index,test_index in split.split(dataframe, dataframe["went_on_backorder"]):
                train_set = dataframe.loc[train_index]
                test_set = dataframe.loc[test_index]


            logging.info("Performed train test split on the dataframe")

            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)

            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train and test file path.")

            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )

            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise OrderData(e,sys) from e
    

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_data_into_feature_store()
            dataframe = dataframe.drop(self._schema_config["drop_columns"],axis=1)
            self.split_data_as_train_test(dataframe=dataframe)
            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.testing_file_path)
            return data_ingestion_artifact
        except Exception as e:
            raise OrderException(e,sys) from e