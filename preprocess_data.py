import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk
import numpy as np
import re

np.random.seed(2018)
nltk.download('wordnet')


def lemmatize_stemming(text):
    stemmer = PorterStemmer()
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def clean_data(text):
	# Remove URL
    result = re.sub(r"http\S+", "", text)
    # Remove User
    result = re.sub('@[^\s]+','',text)
    #Fix Misspelled
    result = re.sub(r'[^a-z]', '', text)

    return result

def preprocess(text):

	# Cleaning the data
    text = clean_data(text)
    result = []

    # Tokenisation and Removing STOPWORDS
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            lem = lemmatize_stemming(token)
            result.append(lem)
    return result


def preprocess_data(fileName):

	data = pd.read_csv(fileName, error_bad_lines=False, encoding='latin-1')
	data_text = data[['text']]
	data_text["index"] = data_text.index
	document = data_text

	processed_docs = document['text'].map(preprocess)
	return preprocess_docs


