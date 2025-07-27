# Feature Engineering
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
import yaml
from src.logger import logging
import pickle

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

def load_data(file_path:str)->pd.DataFrame:
    """
        Load the data from the csv file
    """
    try:
        df=pd.read_csv(file_path)
        logging.info("Data loaded from %s",file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error("Failed to parse the csv file : %s",e)
    except Exception as e:
        logging.error("Unexpected error occured while loading the data : %s",e)
        raise

def apply_bow(train_data:pd.DataFrame,test_data:pd.DataFrame,max_features:int)-> tuple:
    """Apply count vectorizer to the data"""
    try:
        logging.info("Applying BOW....")
        vectorizer=CountVectorizer(max_features=max_features)

        x_train=train_data['review'].values
        y_train=train_data['sentiment'].values
        x_test=test_data['review'].values
        y_test=test_data['sentiment'].values

        x_train_bow=vectorizer.fit_transform(x_train)
        x_test_bow=vectorizer.transform(x_test)

        train_df=pd.DataFrame(x_train_bow.toarray())
        train_df['label']=y_train

        test_df=pd.DataFrame(x_test_bow.toarray())
        test_df['label']=y_test

        pickle.dump(vectorizer,open('models/vectorizer.pkl','wb'))
        logging.info("Bag of words applied and data transformed.....")

        return train_df,test_df
    except Exception as e:
        logging.error("Error during Bag of Words transformation : %s",e)
        raise

def save_data(df:pd.DataFrame,file_path:str)-> None:
    """Save the data frame to a csv file..."""
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logging.info("Data saved to %s ",file_path)
    except Exception as e:
        logging.error("Unexpected error occured while saving the data : %s",e)
        raise

def main():
    try:
        params=load_params('params.yaml')
        max_features=params['feature_engineering']['max_features']

        train_data=load_data('./data/interim/train_processed.csv')
        test_data=load_data('./data/interim/test_processed.csv')

        train_df,test_df=apply_bow(train_data,test_data,max_features)

        save_data(train_df,os.path.join("./data","processed","train_bow.csv"))
        save_data(test_df,os.path.join("./data","processed","test_bow.csv"))
    except Exception as e:
        logging.error("Failed to complete the feature engineering process : %s",e)
        print(f"Error : {e}")


if __name__=="__main__":
    main()


