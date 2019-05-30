# DO NOT RUN
# for splitting the test data and training daat


import json
import io
import numpy as np
import pandas as pd


def train_test_split(n = 0.7):
    # import files
    review_json_file = 'review.json'

    review = []
    for line in open(review_json_file, 'r'):
        review.append(json.loads(line))

    # convert to dataframe
    review_df = pd.DataFrame.from_records(review)
    # extract the userful column
    review_df = review_df.loc[:,['business_id','user_id','text','stars']]
    # split the test and training dataset
    length = len(review_df) * n
    
    review_df_training = review_df.iloc[:n*length,]
    review_df_test = review_df.iloc[:length,]
    review_df_training.to_csv('training.csv')
    review_df_test.to_csv('test.csv')
    print('SUCCESS!!!  train_test_split')
    print('The training set has ', length, 'rows data')
    print('The training set has ', length, 'rows data')

