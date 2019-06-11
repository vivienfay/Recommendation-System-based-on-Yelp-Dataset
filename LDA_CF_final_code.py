# import packages

# file reading & writing
import json
import io
import time
import sys

## data structure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## text preprocessing & LDA
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

## Cosine smimilariy & KL divergence & Measure
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import correlation, cosine
import sklearn.metrics as metrics
import re
import math
from random import sample
from scipy import sparse
from scipy import stats


# import data
review_json_file = 'review.json'
review = []
for line in open(review_json_file, 'r'):
    review.append(json.loads(line))
review_df = pd.DataFrame.from_records(review)
review = review_df[['user_id','business_id','stars','text']]
## split data
def split_data(n=0.7):
    training = review.iloc[:int(n*len(review)),:]
    testing = review.iloc[int(n*len(review)):,:]
    print(int(n*len(review))," rows training data")
    print(len(review) - int(n*len(review))," rows testing data")
    return training,testing



# build LDA function 

## Text preprocessing, remove stop word, extract word stem,keep noun
def textProcessing(text):
    # lower words  
    text = text.lower()
    # remove punctuation
    for c in string.punctuation:
        text = text.replace(c, ' ')
    # tokenize
    wordLst = nltk.word_tokenize(text)
    # stop word
    filtered = [w for w in wordLst if w not in stopwords.words('english')]
    # keep noun  
    refiltered =nltk.pos_tag(filtered)
    filtered = [w for w, pos in refiltered if pos.startswith('NN')]
    # extract the stem
    ps = PorterStemmer()
    filtered = [ps.stem(w) for w in filtered]
    return " ".join(filtered) 

## strengthen the sentiment influence in matrix
def rating_proportion(text,rate):
    return text * int(rate)

## LDA model 
def lda(review,n_topic = 10,n_top_words=20):
# vectorization
# generate the word-docu matrix
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,stop_words='english')
    tf = tf_vectorizer.fit_transform(review)
# train the lda model
    lda = LatentDirichletAllocation(n_topics=n_topic, 
                                max_iter=50,
                                learning_method='batch')
    lda.fit(tf)
# print the performance
    print('perplexity is: ',lda.perplexity(tf))
# generate the top word list for every topic
    tf_feature_names = tf_vectorizer.get_feature_names()
    feature_dict = {k: v for v, k in enumerate(tf_feature_names)}  
# return the topic*word distribution matrix
    return lda.components_,feature_dict,lda.perplexity(tf)

# sythnize the topic probability
def topic_probability(document,feature_dict,topic_word_matrix):
    word_list = document.split()
    topic_num = len(topic_word_matrix)
    topic_probability = {k:0 for k in range(topic_num)}
    for topic_idx in range(topic_num):
        for word in word_list:
            if word in feature_dict.keys():
                topic_probability[topic_idx] += topic_word_matrix[topic_idx,feature_dict[word]]
    return topic_probability

# normalize the matrix
def normalize(x):
    normalized_list = []
    new_dict = {}
    for key,value in x.items():
        normalized_list.append(value**2)
    for key, value in x.items():
        if sum(normalized_list) == 0:
            new_dict[key] = 0
        else:
            new_dict[key] = value / math.sqrt(sum(normalized_list))
    return new_dict


# Collaborative Filtering
def KL_sim(a,b):
    KL_ab = stats.entropy(a,b)
    KL_ba = stats.entropy(b,a)
    return np.exp(-(KL_ab+KL_ba)/2)

def cosine_sim(a,b):
    vector_a = np.mat(a)
    vector_b = np.mat(b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos  #normalize
    return sim

def get_top_k_similar_user(user_id = None,k= 15, biz_id = None,method ='KL'):
    #get the similar user_id who have rated the item
    user_rated_item_id = [id_ for id_ in range(user_num) if rating_matrix[id_,biz_id]!=0]
    #find the most similar user
    if len(user_rated_item_id)< k:
        return 'not enough similar users'
    
    else:
        #get sample user_biz_probability
        user_biz_probability = {}        
        sample_user_id = [sample(set(user_rated_item_id), 300) if len(user_rated_item_id)>300 else user_rated_item_id][0]     
        for user in (sample_user_id + [user_id]):
            user_biz_probability[user] = np.dot((user_topic[user].reshape(1,-1)), business_topic.T)
        
        #calculate similairity
        user_similarity_dic = {} 
        for user in sample_user_id:
            if user_id != user and method ='KL':
                user_similarity_dic[user] =  KL_sim(user_biz_probability[user][0], user_biz_probability[user_id][0])
            if user_id != user and method ='cos':

        similar_user_list= sorted(user_similarity_dic.items(), key=lambda d:d[1], reverse = True)
        similar_user_list = [u[0] for u in similar_user_list if not u[0] == user_id ]

        return user_similarity_dic, similar_user_list


def user_user_rating_prediction(user_id=1, biz_id=1, top_k=10,method = 'KL'):
    user_i_mean_rating = user_mean_rating_dic[user_id]

    if get_top_k_similar_user(user_id = user_id, k = top_k, biz_id = biz_id, method) == 'not enough similar users':
        return user_i_mean_rating

    else:
        user_similarity_dic, top_similar_user_list = get_top_k_similar_user(user_id =user_id , k = top_k, biz_id = biz_id,method)
        # list to store similar user info for calculation: mean rating, rating given biz_id, similarity
        u_info = []

        for similar_user in top_similar_user_list:
            mean_rating = user_mean_rating_dic[similar_user]
            rating_u_i = rating_matrix[similar_user, biz_id]
            similarity = user_similarity_dic[similar_user]
            u_info.append([mean_rating, rating_u_i, similarity])

        similar_user_rating = np.sum([(u_info[i][1] - u_info[i][0]) * u_info[i][2] for i in range(top_k)])
        sum_of_similarity = np.sum([u_info[i][2] for i in range(top_k)])
        predicted_rating = user_i_mean_rating + similar_user_rating / sum_of_similarity
        return np.abs(predicted_rating)

def item_item_rating_prediction(user_id=1, biz_id=1, top_k=3,method = 'KL'):
    item_i_mean_rating = biz_mean_rating_dic[biz_id]
    if get_top_k_similar_items(user_id = user_id, k = top_k, biz_id =biz_id,method ) == 'not enough similar items':
        return item_i_mean_rating
    
    else:
        biz_similarity_dic, similar_item_list = get_top_k_similar_items(user_id = user_id, k = top_k, biz_id =biz_id,method )
        biz_info = []
        
        for similar_item in similar_item_list:
            mean_rating = biz_mean_rating_dic[similar_item]
            rating_u_i = rating_matrix[user_id, similar_item]
            similarity = biz_similarity_dic[similar_item]
            biz_info.append([mean_rating,rating_u_i,similarity])
        similar_item_rating = np.sum([(biz_info[i][1] - biz_info[i][0]) * biz_info[i][2] for i in range(top_k-1)])
        sum_of_similarity = np.sum([biz_info[i][2] for i in range(top_k-1)])
        predicted_rating = item_i_mean_rating + similar_item_rating / sum_of_similarity    
        return np.abs(predicted_rating)



############ main ##############
training,testing = split_data(n=0.7)

# Text Preprocssing
time1 = time.time()
training['text'] = training['text'].apply(textProcessing)
training['text'] = training['text'] * training['stars'].apply(int)
review_by_user = training.groupby('user_id').text.sum()
review_by_business = training.groupby('business_id').text.sum()
time2 = time.time()
print("Success!!!")
print("Finish preprocessing the data!")
print("Using the time:   ",time2-time1)
# build LDA

def LDA(TOPIC_NUM = 20):
    time1 = time.time()
    topic_word_matrix,feature_dict,perplexity = lda(review,n_topic=TOPIC_NUM,n_top_words=20)
    time2 = time.time()
    print("Success!!!")
    print("Finish LDA model!")
    print("LDA time:  ",time2 - time1)
    print("perplexity:   ",perplexity)
    return time2-time1,topic_word_matrix,feature_dict,perplexity

computation_time_list = []
topic_word_matrix_list = []
feature_dict_list = []
perplexity_list = []
user_topic_list = []
business_topic_list = []

for topic_num in [20,40,60,80,100]:
    computation_time, topic_word_matrix,feature_dict,perplexity = lda(topic_num)
    computation_time_list.append(computation_time)
    topic_word_matrix_list.append(topic_word_matrix)
    feature_dict_list.append(feature_dict)
    perplexity_list.append(perplexity)
    #User-Topic/Business-Topic Numpy Matrix
    user_topic = np.zeros([user_num, TOPIC_NUM])
    for i in range(user_num):
        for j in range(0,TOPIC_NUM):
            user_topic[i][j] = review_by_user['topic_pro_list'][i][j] # row: user; column: topic

    business_topic =  np.zeros([biz_num, TOPIC_NUM])
    for i in range(biz_num):
        for j in range(0,TOPIC_NUM):
            business_topic[i][j]=review_by_business['topic_pro_list'][i][j]  #row; business_id; column:topic
    
    user_topic_list.append(user_topic)
    business_topic_list.append(business_topic)

# transfer the user_id & business_id
user_raw_int_id_dic = {v: k for k,v in enumerate(training['user_id'].unique())}
biz_raw_int_id_dic = {v: k for k,v in enumerate(training['business_id'].unique())}

# transform str_id to int_id and keep them in the dataframe
training['user_int_id'] = training['user_id'].apply(lambda x: user_raw_int_id_dic[x])
training['biz_int_id'] = training['business_id'].apply(lambda x: biz_raw_int_id_dic[x])


user_mean_rating_dic = training.groupby('user_int_id').stars.mean().to_dict()
biz_mean_rating_dic = training.groupby('biz_int_id').stars.mean().to_dict()

user_row = training['user_int_id'].values
biz_column = training['biz_int_id'].values

user_num = len(user_raw_int_id_dic)
biz_num = len(biz_raw_int_id_dic)

rating_data = training['stars'].values
rating_matrix = sparse.csr_matrix((rating_data, (user_row, biz_column)), shape=(user_num, biz_num))


# testing
rmse_kl_user = []
rmse_kl_item = []
rmse_cos_user = []
rmse_cos_item = []

for i in range(5):
    user_topic = user_topic_list[i]
    business_topic = business_topic_list[i]

    acutal_rating = []
    user_lda_kl_prediction_rating = []

    for index, row in testing.iterrows():
        user = row['user_id']
        business = row['business_id']
        if user in user_raw_int_id_dic and business in biz_raw_int_id_dic and row['stars']!=0:
            acutal_rating.append(row['stars'])
            user_idx = user_raw_int_id_dic[user]
            business_idx = biz_raw_int_id_dic[business]
            item_lda_kl_prediction_rating.append(item_item_rating_prediction(user_id=user_idx, biz_id=business_idx, top_k=3,method = 'KL'))
            user_lda_kl_prediction_rating.append(user_user_rating_prediction(user_id=user_idx, biz_id=business_idx, top_k=10,method = 'KL'))
            item_lda_cos_prediction_rating.append(item_item_rating_prediction(user_id=user_idx, biz_id=business_idx, top_k=3,method = 'cos'))
            user_lda_cos_prediction_rating.append(user_user_rating_prediction(user_id=user_idx, biz_id=business_idx, top_k=10,method = 'cos'))

    item_lda_kl_rms = math.sqrt(mean_squared_error(acutal_rating, item_lda_kl_prediction_rating))
    user_lda_kl_rms = math.sqrt(mean_squared_error(acutal_rating, user_lda_kl_prediction_rating))
    item_lda_cos_rms = math.sqrt(mean_squared_error(acutal_rating, item_lda_cos_prediction_rating))
    user_lda_cos_rms = math.sqrt(mean_squared_error(acutal_rating, user_lda_cos_prediction_rating))

rmse_kl_user.append(item_lda_kl_rms)
rmse_kl_item.append(user_lda_kl_rms)
rmse_cos_user.append(item_lda_cos_rms)
rmse_cos_item.append(user_lda_cos_rms)
    