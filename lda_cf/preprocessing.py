# DO NOT RUN
# for extracting topic from user and business


import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


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