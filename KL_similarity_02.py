import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats
import math
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import json
import io
import time
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

start = time.time()

training = pd.read_csv('/Users/mjy/Desktop/training.csv')

def lda(review,n_topic = 10):
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,stop_words='english')
    tf = tf_vectorizer.fit_transform(review)
    lda = LatentDirichletAllocation(n_topics=n_topic,
                                max_iter=50,
                                learning_method='batch')
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    feature_dict = {k: v for v, k in enumerate(tf_feature_names)}
    return lda.components_,feature_dict

def textProcessing(text):
    text = text.lower()
    for c in string.punctuation:
        text = text.replace(c, ' ')
    wordLst = nltk.word_tokenize(text)
    filtered = [w for w in wordLst if w not in stopwords.words('english')]
    refiltered =nltk.pos_tag(filtered)
    filtered = [w for w, pos in refiltered if pos.startswith('NN')]
    ps = PorterStemmer()
    filtered = [ps.stem(w) for w in filtered]
    return " ".join(filtered)

def rating_proportion(text,rate):
    return text * int(rate)

def topic_probability(document,feature_dict,topic_word_matrix):
    word_list = document.split()
    topic_num = len(topic_word_matrix)
    topic_probability = {k:0 for k in range(topic_num)}
    for topic_idx in range(topic_num):
        for word in word_list:
            if word in feature_dict.keys():
                topic_probability[topic_idx] += topic_word_matrix[topic_idx,feature_dict[word]]
    return topic_probability

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

def KL_sim(a,b):
    KL_ab = stats.entropy(a,b)
    KL_ba = stats.entropy(b,a)
    return np.exp(-(KL_ab+KL_ba)/2)

def get_top_k_similar_user(user_id = None,k= 15, biz_id = None):
    #get the similar user_id who have rated the ite
    user_rated_item_id = [id_ for id_ in range(user_num) if rating_matrix[id_,biz_id]!=0]
    #find the most similar user
    if len(user_rated_item_id)< k:
        return 'not enough similar users'
    else:
        index_list = np.argsort(user_similarity[user_id])[-k-1:-1]
    return index_list

#parameters
TOPIC_NUM = 10

#load data
training = pd.read_csv('/Users/mjy/Desktop/training.csv')

#generate user_id, biz_id dict
user_raw_int_id_dic = {v: k for k,v in enumerate(training['user_id'].unique())}
biz_raw_int_id_dic = {v: k for k,v in enumerate(training['business_id'].unique())}

user_int_raw_id_dic = {k:v for k,v in enumerate(training['user_id'].unique())}
biz_int_raw_id_dic = {k:v for k,v in enumerate(training['business_id'].unique())}

#transform str_id to int_id and keep them in the dataframe
training['user_int_id'] = training['user_id'].apply(lambda x: user_raw_int_id_dic[x])
training['biz_int_id'] = training['business_id'].apply(lambda x: biz_raw_int_id_dic[x])

#Generate Rating Sparse Matrix
user_row = training['user_int_id'].values
biz_column = training['biz_int_id'].values

user_num = len(user_raw_int_id_dic)
biz_num = len(biz_raw_int_id_dic)

rating_data = training['stars'].values
rating_matrix = sparse.csr_matrix((rating_data, (user_row, biz_column)), shape=(user_num, biz_num))
#c.todense()
#c.toarray()

'''
lda 
'''

#preprocess the text
training['text'] = training['text'].apply(textProcessing)
training['text'] = training['text'] * training['stars'].apply(int)

#user/business - text Dataframe
review_by_user = pd.DataFrame(training.groupby('user_int_id').text.sum())  #index: user_int_id, column='text'
review_by_business = pd.DataFrame(training.groupby('biz_int_id').text.sum())

#applying lda on the entire review corpus
review = training['text']
topic_word_matrix,feature_dict = lda(review, n_topic= TOPIC_NUM)

# generate user-topic matrix
review_by_user['topic_pro_list'] = (review_by_user['text'].
                                    apply(lambda x: topic_probability(x,feature_dict,topic_word_matrix)).
                                    apply(normalize))
review_by_business['topic_pro_list'] = (review_by_business['text'].
                                        apply(lambda x: topic_probability(x,feature_dict,topic_word_matrix)).
                                        apply(normalize))

#User-Topic/Business-Topic Numpy Matrix
user_topic = np.zeros([user_num, TOPIC_NUM])
for i in range(user_num):
    for j in range(0,TOPIC_NUM):
        user_topic[i][j] = review_by_user['topic_pro_list'][i][j] # row: user; column: topic

business_topic =  np.zeros([biz_num, TOPIC_NUM])
for i in range(biz_num):
    for j in range(0,TOPIC_NUM):
        business_topic[i][j]=review_by_business['topic_pro_list'][i][j]  #row; business_id; column:topic

# Calculate User-Business Probability Matrix
user_business_pro = np.dot(user_topic,business_topic.T)

#Calculate User-User Similarity and Business-Business Similarity¶
user_similarity = np.zeros([user_num, user_num])
for i in range(user_num):
    for j in range(user_num):
        if i== j:
            user_similarity[i][j] = 1
        else:
            user_similarity[i][j] = KL_sim(user_business_pro[i],user_business_pro[j])


business_similarity = np.zeros([biz_num,biz_num])
for i in range(biz_num):
    for j in range(biz_num):
        if i==j:
            business_similarity[i][j] = 1
        else:
            business_similarity[i][j] = KL_sim(user_business_pro[:,i], user_business_pro[:,j])


# User-User recommender
user_mean_rating_dic = training.groupby('user_int_id').stars.mean().to_dict()

def user_user_rating_prediction(user_id=1, biz_id=1, top_k=15):
    user_i_mean_rating = user_mean_rating_dic[user_id]

    if get_top_k_similar_user(user_id, k=top_k, biz_id =biz_id ) == 'not enough similar users':
        print('not enough data! return mean rating of this user!')
        return user_i_mean_rating

    else:
        top_similar_user_list = get_top_k_similar_user(user_id, k=top_k)
        # list to store similar user info for calculation: mean rating, rating given biz_id, similarity
        u_info = []

        for similar_user in top_similar_user_list:
            mean_rating = user_mean_rating_dic[similar_user]
            rating_u_i = rating_matrix[similar_user, biz_id]
            similarity = user_similarity[user_id][similar_user]
            u_info.append([mean_rating, rating_u_i, similarity])

        similar_user_rating = np.sum([(u_info[i][1] - u_info[i][0]) * u_info[i][2] for i in range(top_k)])
        sum_of_similarity = np.sum([u_info[i][2] for i in range(top_k)])
        predicted_rating = user_i_mean_rating + similar_user_rating / sum_of_similarity
        return predicted_rating

print(user_user_rating_prediction(user_id=1, biz_id=5, top_k=15))

def user_recommendation(user_id =None, biz_id = None, top_k = 15):
    #get the unrated business
    unrated = [biz_id for biz_id in range(biz_num) if rating_matrix[user_id,biz_id] == 0]
    if rating_matrix[user_id, biz_id]!= 0:
        return 'the item has been rated!'
    else:
        predicted_rating = user_user_rating_prediction(user_id=1, biz_id= biz_id, top_k= top_k)
    return ["Recommended with predicted rating: "+str(predicted_rating) if predicted_rating>3 else "Not Recommened"]

print(user_recommendation(user_id = 1, biz_id = 3, top_k= 15))

#end = time.time()
#print(end - start)


#TODO: Item - Item recommender

# get_top_similar_items
def get_top_k_similar_items(user_id = None, k= 5, biz_id = None):
    #get the items that the given users has rated
    rated_item_id = [id_ for id_ in range(biz_num) if rating_matrix[user_id,id_]!=0]
    if len(rated_item_id)< k:
        return 'not enough similar items'

    #find the most similar user
    else:
        dic_ = {i: business_similarity[i, user_id] for i in rated_item_id if i != user_id}
        index_list = pd.DataFrame.from_dict(dic_, orient='index').sort_values(by=0, ascending=False)[:k].index()
    return index_list

def biz_biz_rating_predation(user_id=1, biz_id=1, top_k=5):
    user_i_mean_rating = user_mean_rating_dic[user_id]

    if get_top_k_similar_items(user_id = user_id, k = top_k, biz_id = biz_id) == 'not enough similar items':
        print('not enough data! return mean rating of this user!')
        return user_i_mean_rating

    else:
        top_similar_item_list = get_top_k_similar_items(user_id = user_id, k = top_k, biz_id = biz_id)
        ''' 
        # Need to be revised, must have bug
        # list to store similar item info for calculation: mean rating, rating given biz_id, similarity
        item_info = []

        for similar_item in top_similar_item_list:
            mean_rating = user_mean_rating_dic[similar_item]
            rating_u_i = rating_matrix[similar_item, biz_id]
            similarity = user_similarity[user_id][similar_item]
            item_info.append([mean_rating, rating_u_i, similarity])

        similar_user_rating = np.sum([(item_info[i][1] - item_info[i][0]) * item_info[i][2] for i in range(top_k)])
        sum_of_similarity = np.sum([item_info[i][2] for i in range(top_k)])
        predicted_rating = user_i_mean_rating + similar_user_rating / sum_of_similarity
        '''
        return None


#TODO: given user i, 对所有未评分的biz的预测评分，算RMSE（？是对所有的算还是前K个算）
