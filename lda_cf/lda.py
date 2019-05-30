# DO NOT RUN
# for extracting topic from user and business


import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

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

    for topic_idx, topic in enumerate(lda.components_):       
        print ("Topic #%d:" % topic_idx)
        print (" ".join([tf_feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))    

# return the topic*word distribution matrix
    return lda.components_,feature_dict


