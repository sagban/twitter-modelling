import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk
import numpy as np
import re



def lemmatize_stemming(text):
    # stemmer = PorterStemmer()
    # return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    return WordNetLemmatizer().lemmatize(text, pos='v')

def clean_data(text):
	# Remove URL
    noURL = re.sub(r"http\S+", "", text)
    # Remove User
    noUser = re.sub('@[^\s]+','', noURL)
    result = re.sub('#[^\s]+','', noUser)
    return result

    

def preprocess(text):

    # Cleaning the data
    text = clean_data(text)
    # print(text)
    result = []

    # Tokenisation and Removing STOPWORDS
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            token = re.sub(r'[^a-z]', '', token)
            lem = lemmatize_stemming(token)
            if len(lem) > 2:
                result.append(lem)
    return result


def preprocess_data(fileName):

	np.random.seed(2018)
	nltk.download('wordnet')
	data = pd.read_csv(fileName, error_bad_lines=False, encoding='latin-1')
	data_text = data[['text']]
	data_text["index"] = data_text.index
	document = data_text
	return document['text'].map(preprocess)


