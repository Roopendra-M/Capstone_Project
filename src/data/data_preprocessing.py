# Data Pre Processing

import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logging
nltk.download("wordnet")
nltk.download("stopwords")

def preprocess_dataframe(df,col="text"):
    """
        Pre process the data frame by applying text preprocessing to a specific column...

        Args:
            df(pd.DataFrame):The dataframe to preprocess
            col(str):The name of the column containing text..
        Returns:
            pd.DataFrame: The pre processed DataFrame..
    """
    lemmatizer=WordNetLemmatizer()
    stop_words=set(stopwords.words("english"))

    def preprocess_text(text):
        """Helper function to preprocess a single line text string...."""
        # helps to remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove numbers..
        text=''.join([char for char in text if not char.isdigit()])
        # Convert to lower case
        text=text.lower()
        # Remove punctuations...
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text).strip()
        # Remove stop words
        text = " ".join([word for word in text.split() if word not in stop_words])
        # Lemmatization
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
        return text
    # Apply preprocessing to the specified column
    df[col]=df[col].apply(preprocess_text)

    # Drop word with NaN values...
    df=df.dropna(subset=[col])
    logging.info("Data Pre-Processing completed....")
    return df

def main():
    try:
        #Fetch the data from data/raw
        train_data=pd.read_csv('./data/raw/train.csv')
        test_data=pd.read_csv('./data/raw/test.csv')
        logging.info("Data Loaded Properly.....")

        # Transform the data
        train_processed_data=preprocess_dataframe(train_data,'review')
        test_processed_data=preprocess_dataframe(test_data,'review')

        data_path=os.path.join("./data","interim")
        os.makedirs(data_path,exist_ok=True)
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)


        logging.info("Processed data saved to %s",data_path)
    except Exception as e:
        logging.error("Failed to complete the data transformation process : %s",e)
        raise


if __name__=="__main__":
        main()