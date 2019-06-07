import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import correlation, cosine
import sklearn.metrics as metrics
import json
import io
import time
import re
import sys
import math
from scipy import sparse
from scipy import stats

# generate R matrix
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

# prediction
# KL
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

def get_top_k_similar_user(user_id = None,k= 15, biz_id = None):
    #get the similar user_id who have rated the ite
    user_rated_item_id = [id_ for id_ in range(user_num) if rating_matrix[id_,biz_id]!=0]
    #find the most similar user
    if len(user_rated_item_id)< k:
        return 'not enough similar users'
    else:
        index_list = np.argsort(user_similarity[user_id])[-k-1:-1]
    return index_list

def KL_sim(a,b):
    KL_ab = stats.entropy(a,b)
    KL_ba = stats.entropy(b,a)
    return np.exp(-(KL_ab+KL_ba)/2)

# Cosine

def findksimilarusers(user_id, ratings, metric = 'cosine', k=1):
    similarities=[]
    indices=[]
    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute') 
    model_knn.fit(ratings)

    distances, indices = model_knn.kneighbors(ratings.iloc[user_id-1, :].values.reshape(1, -1), n_neighbors = k+1)
    similarities = 1-distances.flatten()
    print('{0} most similar users for User {1}:\n'.format(k,user_id))
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == user_id:
            continue
        else:
            print('{0}: User {1}, with similarity of {2}'.format(i, indices.flatten()[i]+1, similarities.flatten()[i]))
            
    return similarities,indices

def cos_predict_userbased(user_id, item_id, ratings, metric = 'cosine', k=5):
    prediction=0
    similarities, indices=findksimilarusers(user_id, ratings,metric, k) #similar users based on cosine similarity
    mean_rating = ratings.loc[user_id,:].mean() #to adjust for zero based indexing
    sum_wt = np.sum(similarities)-1
    product=1
    wtd_sum = 0 
    
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == user_id:
            continue
        else: 
            ratings_diff = ratings.iloc[indices.flatten()[i],item_id]-np.mean(ratings.iloc[indices.flatten()[i],:])
            product = ratings_diff * (similarities[i])
            wtd_sum = wtd_sum + product
    
    prediction = int(round(mean_rating + (wtd_sum/sum_wt)))
    print('\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id,item_id,prediction))

    return prediction

def cos_predict_itembased(user_id, item_id, ratings, metric = 'cosine', k=5):
    prediction= wtd_sum =0
    similarities, indices=findksimilarusers(user_id, ratings,metric, k) #similar users based on correlation coefficients
    mean_rating = ratings.loc[:,item_id].mean()
    sum_wt = np.sum(similarities)-1
    product = 1
    wtd_sum = 0
    
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == item_id:
            continue
        else:
            product = ratings.iloc[user_id,indices.flatten()[i]] * (similarities[i])
            wtd_sum = wtd_sum + product    
            
    prediction = int(round(wtd_sum/sum_wt))
    print ('\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id,item_id,prediction)      
)
    return prediction

##### run
# import data
time1 = time.time()
review1 = pd.read_csv('review1.csv')
review2 = pd.read_csv('review2.csv')
review3 = pd.read_csv('review3.csv')
review4 = pd.read_csv('review4.csv')
review5 = pd.read_csv('review5.csv')
review = pd.concat([review1,review2,review3,review4,review5],ignore_index=True)

#split training set and test set
n = 0.7
training = review.iloc[:int(n*len(review)),:]
testing = review.iloc[int(n*len(review)):,:]
print(int(n*len(review))," rows training data")
print(len(review) - int(n*len(review))," rows training data")


#generate user_id, biz_id dict
user_raw_int_id_dic = {v: k for k,v in enumerate(training['user_id'].unique())}
biz_raw_int_id_dic = {v: k for k,v in enumerate(training['business_id'].unique())}

#transform str_id to int_id and keep them in the dataframe
training['user_int_id'] = training['user_id'].apply(lambda x: user_raw_int_id_dic[x])
training['biz_int_id'] = training['business_id'].apply(lambda x: biz_raw_int_id_dic[x])
time2 = time.time()
print("Success!!!")
print("Finish splitting data and format adjust!")
print("Using the time:   ",time2-time1)

#preprocessing the data 
time1 = time.time()
training['text'] = training['text'].apply(textProcessing)
training['text'] = training['text'] * training['stars'].apply(int)
review_by_user = training.groupby('user_id').text.sum()
review_by_business = training.groupby('business_id').text.sum()
review_by_user.to_csv('review_by_user.csv')
review_by_business.to_csv('review_by_business.csv')
time2 = time.time()
print("Success!!!")
print("Finish preprocessing the data!")
print("Using the time:   ",time2-time1)


# building the lda model
# def lda_iterate(i):
# LDA model
review = training['text'] 
topic_word_matrix_set = []
feature_dict_set = []
perplexity_set = []
# lda_set = []
user_lda_kl_rms_list = []
item_lda_kl_rms_list = []
user_lda_cos_rms_list = []
item_lda_cos_rms_list = []


for TOPIC_NUM in [20,40,60,80,100]:
    time1 = time.time()
    TOPIC_NUM = 20
    topic_word_matrix,feature_dict,perplexity = lda(review,n_topic=TOPIC_NUM,n_top_words=20)
    # topic_word_matrix_set.append(topic_word_matrix)
    # feature_dict_set.append(feature_dict)
    # perplexity_set.append(perplexity)
    #     lda_set.append(lda)
    time2 = time.time()
    perplexity_set.append(perplexity)
    print("Success!!!")
    print("Finish LDA model!")
    print("LDA time:  ",time2 - time1)
    print("perplexity:   ",perplexity)

    # Generate Rating Sparse Matrix
    user_row = training['user_int_id'].values
    biz_column = training['biz_int_id'].values

    user_num = len(user_raw_int_id_dic)
    biz_num = len(biz_raw_int_id_dic)

    rating_data = training['stars'].values
    rating_matrix = sparse.csr_matrix((rating_data, (user_row, biz_column)), shape=(user_num, biz_num))

    # generate the R matrix
    review_by_user = pd.DataFrame(training.groupby('user_int_id').text.sum())  #index: user_int_id, column='text'
    review_by_business = pd.DataFrame(training.groupby('biz_int_id').text.sum())
    review_by_user['topic_pro_list'] = (review_by_user['text'].
                                        apply(lambda x: topic_probability(x,feature_dict,topic_word_matrix)).
                                        apply(normalize))
    review_by_business['topic_pro_list'] = (review_by_business['text'].
                                            apply(lambda x: topic_probability(x,feature_dict,topic_word_matrix)).
                                            apply(normalize))

    # 

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
    # KL similarity
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
    
    user_mean_rating_dic = training.groupby('user_int_id').stars.mean().to_dict()

    acutal_rating = []
    user_lda_cos_prediction_rating = []
    item_lda_cos_prediction_rating = []
    user_lda_kl_prediction_rating = []
    item_lda_kl_prediction_rating = []

    for index, row in training.iterrows():
        user = row['user_id']
        business = row['business_id']
        if user in user_raw_int_id_dic and business in biz_raw_int_id_dic and row['stars']!=0:
            acutal_rating.append(row['stars'])
            user_idx = user_raw_int_id_dic[user]
            business_idx = biz_raw_int_id_dic[business]
            user_lda_kl_prediction_rating.append(user_user_rating_prediction(user_id=user_idx, biz_id=business_idx, top_k=15))
    #         item_lda_kl_prediction_rating.append(predict_userbased(user_idx,business_idx,pd.DataFrame(user_business_pro)))
            item_lda_cos_prediction_rating.append(cos_predict_itembased(user_idx, business_idx, pd.DataFrame(user_business_pro), metric = 'cosine', k=5))
            user_lda_cos_prediction_rating.append(cos_predict_userbased(user_idx, business_idx, pd.DataFrame(user_business_pro), metric = 'cosine', k=5))

    user_lda_kl_rms = math.sqrt(mean_squared_error(acutal_rating, user_lda_kl_prediction_rating))
    # item_lda_kl_rms = math.sqrt(mean_squared_error(acutal_rating, item_lda_kl_prediction_rating))
    item_lda_cos_rms = math.sqrt(mean_squared_error(acutal_rating, item_lda_cos_prediction_rating))
    user_lda_cos_rms = math.sqrt(mean_squared_error(acutal_rating, user_lda_cos_prediction_rating))

    user_lda_kl_rms_list.append(user_lda_kl_rms)
    # item_lda_kl_rms_list.append(item_lda_kl_rms)
    user_lda_cos_rms_list.append(user_lda_cos_rms)
    item_lda_cos_rms_list.append(item_lda_cos_rms)

print(user_lda_kl_rms_list)
print(item_lda_kl_rms_list)
print(user_lda_cos_rms_list)
print(item_lda_cos_rms_list)

plt.xlabel("Topic Number")
plt.ylabel("Perplexity")
plt.plot([20,40,60,80,100],perplexity_set)
plt.show()
