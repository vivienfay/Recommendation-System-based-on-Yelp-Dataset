import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import json
import io
import time
import re
import sys
from scipy import sparse
from scipy import stats


def train_test_split(n = 0.7):
    time1 = time.time()
    # import files
    review_json_file = '/Users/wenxianfei/Desktop/lda_cf/review.json'

    review = []
    for line in open(review_json_file, 'r'):
        review.append(json.loads(line))

    # convert to dataframe
    review_df = pd.DataFrame.from_records(review)
    # extract the userful column
    review_df = review_df.loc[:,['business_id','user_id','stars','text']]
    # split the test and training dataset
    length = int(len(review_df) * n)

    review_df_training = review_df.iloc[:length,]
    review_df_test = review_df.iloc[length:,]
    review_df_training.to_csv('training.csv')
    review_df_test.to_csv('test.csv')
    time2 = time.time()
    print('SUCCESS!!!  train_test_split')
    print('The training set has ', length, 'rows data')
    print('The testing set has ', len(review_df) - length, 'rows data')
    print('Time: ', time2 - time1)

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
    # xtract the stem
    ps = PorterStemmer()
    filtered = [ps.stem(w) for w in filtered]
    return " ".join(filtered) 

def rating_proportion(text,rate):
    return text * int(rate)

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

#     for topic_idx, topic in enumerate(lda.components_):       
#         print ("Topic #%d:" % topic_idx)
#         print (" ".join([tf_feature_names[i]
#                         for i in topic.argsort()[:-n_top_words - 1:-1]]))    

# return the topic*word distribution matrix
    return lda.components_,feature_dict,lda.perplexity(tf)

def topic_probability(document,feature_dict,topic_word_matrix):
    word_list = document.split()
    topic_num = len(topic_word_matrix)
    topic_probability = {k:0 for k in range(topic_num)}
    for topic_idx in range(topic_num):
        for word in word_list:
            if word in feature_dict.keys():
                topic_probability[topic_idx] += topic_word_matrix[topic_idx,feature_dict[word]]
    return topic_probability

def KL_sim(a,b):
    KL_ab = stats.entropy(a,b)
    KL_ba = stats.entropy(b,a)
    return np.exp(-(KL_ab+KL_ba)/2)

def get_top_k_similar_user(user_id = None,k= 15, biz_id = None):
    #get the similar user_id who have rated the item
    user_rated_item_id = [id_ for id_ in range(user_num) if rating_matrix[id_,biz_id]!=0]
    #find the most similar user
    if len(user_rated_item_id)< k:
        return 'not enough similar users'
    else:
        index_list = np.argsort(user_similarity[user_id])[-k-1:-1]
    return index_list    

train_test_split(n = 0.0001)
training = pd.read_csv('training.csv')
training['text'] = training['text'].apply(textProcessing)
training['text'] = training['text'] * training['stars'].apply(int)
review_by_user = training.groupby('user_id').text.sum()
review_by_business = training.groupby('business_id').text.sum()
review_by_user.to_csv('review_by_user.csv')
review_by_business.to_csv('review_by_business.csv')

# LDA model
review = training['text'] 
topic_word_matrix_set = []
feature_dict_set = []
perplexity_set = []
# lda_set = []
# for i in [10,20,30,40,50,60,70,80,90,100]:
i = 20
topic_word_matrix,feature_dict,perplexity = lda(review,n_topic=i,n_top_words=20)
# topic_word_matrix_set.append(topic_word_matrix)
# feature_dict_set.append(feature_dict)
# perplexity_set.append(perplexity)
#     lda_set.append(lda)
time2 = time.time()
print("LDA time:  ",time2 - time1)
print("perplexity:   ",perplexity)
# plt.xlabel("Topic Number")
# plt.ylabel("Perplexity")
# plt.plot([i for i in range(10,40)],perplexity_set)
# plt.show()

# generate the R matrix
review_by_user = pd.read_csv('review_by_user.csv',header=None)
review_by_business  = pd.read_csv('review_by_business',header=None)
review_by_user[2] = review_by_user[1].apply(lambda x: topic_probability(x,feature_dict,topic_word_matrix))
review_by_business[2] = review_by_business[1].apply(lambda x: topic_probability(x,feature_dict,topic_word_matrix))
# normalize
review_by_user[2] = review_by_user[2].apply(normalize)
review_by_business[2] = review_by_business[2].apply(normalize)
# generate the user_business R matrix
user_business = {}
time1 = time.time()
for i in range(len(review_by_user)):
    for j in range(len(review_by_business)):
        user_rating = review_by_user.iloc[i,2]
        business_rating = review_by_business.iloc[j,2]
        rating = 0
        for k in range(len(user_rating)):
            rating += user_rating[k]*business_rating[k]
        user_business[(review_by_user.iloc[i,0],review_by_business.iloc[j,0])] = rating
time2 = time.time()
print("Success!!!")
print("Already generated the big R matrix")
print("Big R generation time:   ",time2-time1)

# KL similarity

#generate user_id, biz_id dict
user_raw_int_id_dic = {v: k for k,v in enumerate(training['user_id'].unique())}
biz_raw_int_id_dic = {v: k for k,v in enumerate(training['business_id'].unique())}
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









