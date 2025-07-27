# data ingestion
import pandas as pd
import numpy as np
pd.set_option("future.no_silent_downcasting",True)

import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.logger import logging



def load_params(params_path:str)->dict:
    """
        Load parameters from the yaml file
    """
    try:
        with open(params_path,'r') as file:
            params=yaml.safe_load(file)
        logging.debug("Parameters received from the %s",params_path)
        return params
    except FileNotFoundError:
        logging.error("File is not found %s",params_path)
    except yaml.YAMLError as e:
        logging.error("Yaml Error: %s",e)
    except Exception as e:
        logging.error("Unexpected error: %s",e)
        raise

def load_data(data_url:str)->pd.DataFrame:
    """
        Load the data from the csv file
    """
    try:
        df=pd.read_csv(data_url)
        logging.info("Data loaded from %s",data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error("Failed to parse the csv file : %s",e)
    except Exception as e:
        logging.error("Unexpected error occured while loading the data : %s",e)
        raise

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    """Pre process the data"""
    try:
        logging.info("Pre-Processing....")
        final_df=df[df['sentiment'].isin(['positive','negative'])]
        final_df['sentiment']=final_df["sentiment"].replace({'positive':1,'negative':0})
        logging.info("Data Pre Processing completed")
        return final_df
    except KeyError as e:
        logging.error("Missing the column in the dataFrame : %s",e)
    except Exception as e:
        logging.error("Unexpected error during preprocessing : %s",e)
        raise

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)-> None:
    """Save the train and test dataset....."""
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)
        logging.info("Train and the Test data saved to : %s",data_path)
    except Exception as e:
        logging.error("Unexpected error occured while saving the data : %s",e)
        raise
def main():
    try:
        params=load_params(params_path="params.yaml")
        test_size=params['data_ingestion']['test_size']

        df=load_data(data_url="https://raw.githubusercontent.com/Roopendra-M/DataSets/refs/heads/main/data.csv")

        final_df=preprocess_data(df)
        train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=42)
        save_data(train_data=train_data,test_data=test_data,data_path="./data")
    except Exception as e:
        logging.error("Failed to complete the data ingestion process : %s",e)
        print(f"Error: {e}")
    
if __name__=="__main__":
    main()
