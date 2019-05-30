# ''' please run it main '''

# import basic packages
import numpy as np
import pandas as pd
import nltk
import gensim
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
import json
import io

import re
import sys
from pyspark import SparkConf, SparkContext

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# import other packages
from train_test_split import train_test_split
from preprocessing import textProcessing,rating_proportion
from lda import lda
from cf import cf

# split training and test data, n means how much proportion is extrated from entire dataset
train_test_split(n=0.001)

# convert it as RDD
conf = SparkConf()
sc = SparkContext(conf=conf)
lines = sc.textFile('training.csv')
# preprocessing the data(remove stopword,keep noun,extract stem),multiply by rating
review_set = lines.map(lambda x: x.split(sep=',')).map(lambda x: [x[0],x[1],textProcessing(x[2]),x[3]]).map(lambda x:[x[0],x[1],rating_proportion(x[2],x[3])])
# aggregate by user or business
review_by_user = review_set.map(lambda x:(x[1],x[2])).reduceByKey(lambda x1,x2: x1+x2)
review_by_business = review_set.map(lambda x:(x[0],x[2])).reduceByKey(lambda x1,x2: x1+x2)

# run lda topic model
review = review_set.map(lambda x:x[2]).collect()
topic_word_matrix,feature_dict = lda(review,n_topic=10,n_top_words=20)

# generate the userid / businessid topic matrix
def topic_probability(document,feature_dict,topic_word_matrix):
    word_list = document.split(' ')
    topic_num = len(topic_word_matrix)
    topic_probability = {k:0 for k in range(topic_num)}
    for topic_idx in len(topic_word_matrix):
        for word in word_list:
            topic_probability[topic_idx] += topic_word_matrix[topic_idx,feature_dict[word]]
    return topic_probability

user_topic = review_by_user.map(lambda x: [x[0],topic_probability(x[1],feature_dict,topic_word_matrix)])
user_topic_matrix = user_topic.flatMap(lambda x: [(x[0],i,x[1][i]) for i in len(topic_word_matrix)])
business_topic = review_by_business.map(lambda x: [x[0],topic_probability(x[1],feature_dict,topic_word_matrix)])
business_topic_matrix = business_topic.flatMap(lambda x: [(x[0],i,x[1][i]) for i in len(topic_word_matrix)])

# generate the R: user - topic   *     topic - business
# transpose business- topic
topic_business_matrix = business_topic_matrix.map(lambda x: (x[1],x[0],x[2]))
user_business_matrix_dict = user_topic_matrix.map(lambda x:((x[0],x[1]),x[2])).join(topic_business_matrix.map(lambda x:((x[0],x[1]),x[2]))).map(lambda x:(x[0],x[1][0]*x[1][1])).collectAsMap()

# generate the businss list, user list
business_list = []
user_list = []
for key,value in user_business_matrix_dict.item():
    user_list.append(key[0])
    business_list.append(key[1])

user_business_probability_matrix = np.zeros((len(user_list),len(business_list)))
useridx = 0
businessidx = 0
for key,value in user_business_matrix_dict.item():
    user_business_probability_matrix[useridx,businessidx] = value
    useridx += 1
    businessidx += 1
# collaborative filtering
cf(user_business_probability_matrix,business_list)


 






