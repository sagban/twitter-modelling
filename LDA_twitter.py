{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 152,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import gensim\n",
    "from gensim.utils import simple_preprocess\n",
    "from gensim.parsing.preprocessing import STOPWORDS\n",
    "from nltk.stem import WordNetLemmatizer, SnowballStemmer\n",
    "from nltk.stem.porter import *\n",
    "import nltk\n",
    "import numpy as np\n",
    "import re\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "metadata": {},
   "outputs": [],
   "source": [
    "def lemmatize_stemming(text):\n",
    "    # stemmer = PorterStemmer()\n",
    "    # return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))\n",
    "    return WordNetLemmatizer().lemmatize(text, pos='v')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 154,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_data(text):\n",
    "\t# Remove URL\n",
    "    noURL = re.sub(r\"http\\S+\", \"\", text)\n",
    "    # Remove User\n",
    "    noUser = re.sub('@[^\\s]+','', noURL)\n",
    "    result = re.sub('#[^\\s]+','', noUser)\n",
    "    return result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 165,
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess(text):\n",
    "\n",
    "    # Cleaning the data\n",
    "    text = clean_data(text)\n",
    "    # print(text)\n",
    "    result = []\n",
    "\n",
    "    # Tokenisation and Removing STOPWORDS\n",
    "    for token in gensim.utils.simple_preprocess(text):\n",
    "        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:\n",
    "            token = re.sub(r'[^a-z]', '', token)\n",
    "            lem = lemmatize_stemming(token)\n",
    "            if len(lem) > 2:\n",
    "                result.append(lem)\n",
    "    return result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess_data(fileName):\n",
    "\n",
    "    np.random.seed(2018)\n",
    "    nltk.download('wordnet')\n",
    "    data = pd.read_csv(fileName, error_bad_lines=False, encoding='latin-1')\n",
    "    data_text = data[['text']]\n",
    "    data_text[\"index\"] = data_text.index\n",
    "    document = data_text\n",
    "    return document['text'].map(preprocess)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 167,
   "metadata": {},
   "outputs": [],
   "source": [
    "fileName = \"./CSV/data1.csv\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package wordnet to /Users/sagban/nltk_data...\n",
      "[nltk_data]   Package wordnet is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "processed_docs = preprocess_data(fileName)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 178,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0 ayushman\n",
      "1 bear\n",
      "2 bharat\n",
      "3 caesarean\n",
      "4 economic\n",
      "5 modi\n",
      "6 section\n",
      "7 time\n",
      "8 birth\n",
      "9 claim\n",
      "10 fittingly\n"
     ]
    }
   ],
   "source": [
    "dictionary = gensim.corpora.Dictionary(processed_docs)\n",
    "count = 0\n",
    "for k, v in dictionary.iteritems():\n",
    "    print(k, v)\n",
    "    count += 1\n",
    "    if count > 10:\n",
    "        break"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    [modi, ayushman, bharat, bear, caesarean, sect...\n",
       "1        [claim, great, start, fittingly, birth, girl]\n",
       "2    [claim, raise, baby, girl, bear, caesarean, se...\n",
       "3        [claim, great, start, fittingly, birth, girl]\n",
       "4        [claim, great, start, fittingly, birth, girl]\n",
       "5    [claim, raise, ayushman, bharat, birth, girl, ...\n",
       "6    [ayushman, bharat, bear, caesarean, section, e...\n",
       "7        [claim, great, start, fittingly, birth, girl]\n",
       "8    [claim, raise, baby, girl, bear, caesarean, se...\n",
       "9    [put, ayushman, bharat, delhi, jeopardy, deman...\n",
       "Name: text, dtype: object"
      ]
     },
     "execution_count": 179,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "processed_docs[:10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 172,
   "metadata": {},
   "outputs": [],
   "source": [
    "dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 180,
   "metadata": {},
   "outputs": [],
   "source": [
    "bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 181,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[(8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1)]"
      ]
     },
     "execution_count": 181,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bow_corpus[4]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 187,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Word 0 (\"ayushman\") appears 1 time.\n",
      "Word 2 (\"bharat\") appears 1 time.\n",
      "Word 11 (\"girl\") appears 1 time.\n",
      "Word 20 (\"haryana\") appears 1 time.\n",
      "Word 61 (\"beneficiary\") appears 1 time.\n",
      "Word 106 (\"yojana\") appears 1 time.\n",
      "Word 108 (\"india\") appears 1 time.\n",
      "Word 312 (\"newborn\") appears 1 time.\n",
      "Word 313 (\"namo\") appears 1 time.\n"
     ]
    }
   ],
   "source": [
    "bow_doc_43 = bow_corpus[1220]\n",
    "for i in range(len(bow_doc_43)):\n",
    "    print(\"Word {} (\\\"{}\\\") appears {} time.\".format(bow_doc_43[i][0], \n",
    "                                               dictionary[bow_doc_43[i][0]], \n",
    "bow_doc_43[i][1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 188,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[(0, 0.12378404805943978),\n",
      " (1, 0.14415817792103724),\n",
      " (2, 0.12582356736846856),\n",
      " (3, 0.18190466523509546),\n",
      " (4, 0.6101535537975377),\n",
      " (5, 0.371754057277885),\n",
      " (6, 0.17944150667776626),\n",
      " (7, 0.6101535537975377)]\n"
     ]
    }
   ],
   "source": [
    "from gensim import corpora, models\n",
    "\n",
    "\n",
    "tfidf = models.TfidfModel(bow_corpus)\n",
    "\n",
    "\n",
    "corpus_tfidf = tfidf[bow_corpus]\n",
    "\n",
    "from pprint import pprint\n",
    "\n",
    "\n",
    "for doc in corpus_tfidf:\n",
    "    pprint(doc)\n",
    "    break"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 189,
   "metadata": {},
   "outputs": [],
   "source": [
    "lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 190,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Topic: 0 \n",
      "Words: 0.034*\"bharat\" + 0.034*\"ayushman\" + 0.025*\"healthcare\" + 0.025*\"review\" + 0.024*\"price\" + 0.024*\"scheme\" + 0.024*\"modi\" + 0.024*\"think\" + 0.024*\"sure\" + 0.023*\"doctor\"\n",
      "Topic: 1 \n",
      "Words: 0.064*\"namo\" + 0.058*\"girl\" + 0.050*\"ayushman\" + 0.049*\"bharat\" + 0.037*\"claim\" + 0.034*\"baby\" + 0.027*\"beneficiary\" + 0.026*\"haryana\" + 0.026*\"section\" + 0.026*\"hospital\"\n",
      "Topic: 2 \n",
      "Words: 0.079*\"bharat\" + 0.078*\"ayushman\" + 0.049*\"claim\" + 0.043*\"baby\" + 0.040*\"haryana\" + 0.040*\"beneficiary\" + 0.038*\"karishma\" + 0.031*\"scheme\" + 0.029*\"karnal\" + 0.026*\"arrive\"\n",
      "Topic: 3 \n",
      "Words: 0.110*\"claim\" + 0.109*\"girl\" + 0.062*\"bear\" + 0.062*\"baby\" + 0.062*\"section\" + 0.062*\"caesarean\" + 0.062*\"raise\" + 0.061*\"hospital\" + 0.061*\"chawla\" + 0.061*\"kalpana\"\n",
      "Topic: 4 \n",
      "Words: 0.074*\"bear\" + 0.058*\"section\" + 0.054*\"caesarean\" + 0.045*\"hospital\" + 0.044*\"girl\" + 0.039*\"ayushman\" + 0.039*\"bharat\" + 0.038*\"chawla\" + 0.037*\"kalpana\" + 0.035*\"claim\"\n",
      "Topic: 5 \n",
      "Words: 0.098*\"haryana\" + 0.098*\"beneficiary\" + 0.096*\"ayushman\" + 0.096*\"bharat\" + 0.082*\"india\" + 0.082*\"girl\" + 0.081*\"yojana\" + 0.080*\"newborn\" + 0.056*\"namo\" + 0.031*\"baby\"\n",
      "Topic: 6 \n",
      "Words: 0.127*\"bear\" + 0.064*\"hospital\" + 0.064*\"kalpana\" + 0.064*\"chawla\" + 0.057*\"beneficiary\" + 0.056*\"karishma\" + 0.053*\"haryana\" + 0.053*\"govt\" + 0.052*\"karnal\" + 0.048*\"days\"\n",
      "Topic: 7 \n",
      "Words: 0.112*\"india\" + 0.111*\"beneficiary\" + 0.091*\"karishma\" + 0.088*\"karnal\" + 0.040*\"ayushman\" + 0.032*\"girl\" + 0.031*\"health\" + 0.031*\"start\" + 0.023*\"haryana\" + 0.022*\"newborn\"\n",
      "Topic: 8 \n",
      "Words: 0.087*\"ayushman\" + 0.087*\"bharat\" + 0.071*\"hospital\" + 0.070*\"yojana\" + 0.069*\"pilot\" + 0.069*\"inaugurate\" + 0.069*\"tomorrow\" + 0.069*\"balrampur\" + 0.031*\"scheme\" + 0.029*\"benefit\"\n",
      "Topic: 9 \n",
      "Words: 0.070*\"girl\" + 0.056*\"namo\" + 0.051*\"ayushman\" + 0.050*\"bharat\" + 0.039*\"india\" + 0.037*\"baby\" + 0.036*\"claim\" + 0.035*\"beneficiary\" + 0.034*\"yojana\" + 0.033*\"haryana\"\n"
     ]
    }
   ],
   "source": [
    "for idx, topic in lda_model.print_topics(-1):\n",
    "    print('Topic: {} \\nWords: {}'.format(idx, topic))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 191,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Topic: 0 Word: 0.040*\"scheme\" + 0.038*\"demand\" + 0.038*\"jeopardy\" + 0.038*\"party\" + 0.038*\"put\" + 0.034*\"delhi\" + 0.025*\"ayushman\" + 0.025*\"benefit\" + 0.023*\"bharat\" + 0.020*\"namo\"\n",
      "Topic: 1 Word: 0.254*\"namo\" + 0.043*\"angel\" + 0.041*\"yojna\" + 0.041*\"pradhan\" + 0.040*\"mantri\" + 0.029*\"baby\" + 0.023*\"karishma\" + 0.018*\"haryana\" + 0.018*\"beneficiary\" + 0.010*\"payment\"\n",
      "Topic: 2 Word: 0.095*\"india\" + 0.086*\"newborn\" + 0.075*\"yojana\" + 0.074*\"beneficiary\" + 0.065*\"haryana\" + 0.061*\"namo\" + 0.055*\"bharat\" + 0.054*\"ayushman\" + 0.038*\"girl\" + 0.030*\"karnal\"\n",
      "Topic: 3 Word: 0.049*\"fittingly\" + 0.049*\"great\" + 0.048*\"birth\" + 0.047*\"claim\" + 0.047*\"start\" + 0.041*\"caesarean\" + 0.041*\"section\" + 0.039*\"hary\" + 0.039*\"raise\" + 0.035*\"bear\"\n",
      "Topic: 4 Word: 0.050*\"haryana\" + 0.049*\"bear\" + 0.046*\"beneficiary\" + 0.046*\"karnal\" + 0.044*\"karishma\" + 0.036*\"days\" + 0.036*\"august\" + 0.035*\"kalpana\" + 0.035*\"chawla\" + 0.034*\"namo\"\n",
      "Topic: 5 Word: 0.052*\"pilot\" + 0.051*\"tomorrow\" + 0.051*\"balrampur\" + 0.051*\"inaugurate\" + 0.030*\"bharat\" + 0.030*\"ayushman\" + 0.028*\"yojana\" + 0.028*\"modi\" + 0.028*\"hospital\" + 0.019*\"overnight\"\n",
      "Topic: 6 Word: 0.091*\"hary\" + 0.091*\"raise\" + 0.088*\"caesarean\" + 0.087*\"section\" + 0.079*\"chawla\" + 0.079*\"kalpana\" + 0.072*\"baby\" + 0.071*\"bear\" + 0.070*\"hospital\" + 0.055*\"claim\"\n",
      "Topic: 7 Word: 0.045*\"benefit\" + 0.035*\"bear\" + 0.032*\"govt\" + 0.031*\"sure\" + 0.030*\"doctor\" + 0.030*\"earful\" + 0.030*\"slash\" + 0.030*\"business\" + 0.030*\"price\" + 0.030*\"think\"\n",
      "Topic: 8 Word: 0.140*\"fittingly\" + 0.140*\"great\" + 0.137*\"birth\" + 0.136*\"start\" + 0.071*\"claim\" + 0.048*\"girl\" + 0.011*\"balrampur\" + 0.011*\"tomorrow\" + 0.011*\"pilot\" + 0.011*\"inaugurate\"\n",
      "Topic: 9 Word: 0.040*\"review\" + 0.027*\"status\" + 0.027*\"attain\" + 0.026*\"celebrity\" + 0.023*\"economic\" + 0.023*\"time\" + 0.021*\"bharat\" + 0.020*\"ayushman\" + 0.020*\"infrastructure\" + 0.020*\"pragati\"\n"
     ]
    }
   ],
   "source": [
    "lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)\n",
    "for idx, topic in lda_model_tfidf.print_topics(-1):\n",
    "    print('Topic: {} Word: {}'.format(idx, topic))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 192,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Score: 0.8714197278022766\t \n",
      "Topic: 0.110*\"claim\" + 0.109*\"girl\" + 0.062*\"bear\" + 0.062*\"baby\" + 0.062*\"section\" + 0.062*\"caesarean\" + 0.062*\"raise\" + 0.061*\"hospital\" + 0.061*\"chawla\" + 0.061*\"kalpana\"\n",
      "\n",
      "Score: 0.014287575148046017\t \n",
      "Topic: 0.064*\"namo\" + 0.058*\"girl\" + 0.050*\"ayushman\" + 0.049*\"bharat\" + 0.037*\"claim\" + 0.034*\"baby\" + 0.027*\"beneficiary\" + 0.026*\"haryana\" + 0.026*\"section\" + 0.026*\"hospital\"\n",
      "\n",
      "Score: 0.014287104830145836\t \n",
      "Topic: 0.079*\"bharat\" + 0.078*\"ayushman\" + 0.049*\"claim\" + 0.043*\"baby\" + 0.040*\"haryana\" + 0.040*\"beneficiary\" + 0.038*\"karishma\" + 0.031*\"scheme\" + 0.029*\"karnal\" + 0.026*\"arrive\"\n",
      "\n",
      "Score: 0.014286909252405167\t \n",
      "Topic: 0.112*\"india\" + 0.111*\"beneficiary\" + 0.091*\"karishma\" + 0.088*\"karnal\" + 0.040*\"ayushman\" + 0.032*\"girl\" + 0.031*\"health\" + 0.031*\"start\" + 0.023*\"haryana\" + 0.022*\"newborn\"\n",
      "\n",
      "Score: 0.014286840334534645\t \n",
      "Topic: 0.070*\"girl\" + 0.056*\"namo\" + 0.051*\"ayushman\" + 0.050*\"bharat\" + 0.039*\"india\" + 0.037*\"baby\" + 0.036*\"claim\" + 0.035*\"beneficiary\" + 0.034*\"yojana\" + 0.033*\"haryana\"\n",
      "\n",
      "Score: 0.014286789111793041\t \n",
      "Topic: 0.127*\"bear\" + 0.064*\"hospital\" + 0.064*\"kalpana\" + 0.064*\"chawla\" + 0.057*\"beneficiary\" + 0.056*\"karishma\" + 0.053*\"haryana\" + 0.053*\"govt\" + 0.052*\"karnal\" + 0.048*\"days\"\n",
      "\n",
      "Score: 0.014286471530795097\t \n",
      "Topic: 0.074*\"bear\" + 0.058*\"section\" + 0.054*\"caesarean\" + 0.045*\"hospital\" + 0.044*\"girl\" + 0.039*\"ayushman\" + 0.039*\"bharat\" + 0.038*\"chawla\" + 0.037*\"kalpana\" + 0.035*\"claim\"\n",
      "\n",
      "Score: 0.0142863430082798\t \n",
      "Topic: 0.098*\"haryana\" + 0.098*\"beneficiary\" + 0.096*\"ayushman\" + 0.096*\"bharat\" + 0.082*\"india\" + 0.082*\"girl\" + 0.081*\"yojana\" + 0.080*\"newborn\" + 0.056*\"namo\" + 0.031*\"baby\"\n",
      "\n",
      "Score: 0.014286255463957787\t \n",
      "Topic: 0.034*\"bharat\" + 0.034*\"ayushman\" + 0.025*\"healthcare\" + 0.025*\"review\" + 0.024*\"price\" + 0.024*\"scheme\" + 0.024*\"modi\" + 0.024*\"think\" + 0.024*\"sure\" + 0.023*\"doctor\"\n",
      "\n",
      "Score: 0.014285965822637081\t \n",
      "Topic: 0.087*\"ayushman\" + 0.087*\"bharat\" + 0.071*\"hospital\" + 0.070*\"yojana\" + 0.069*\"pilot\" + 0.069*\"inaugurate\" + 0.069*\"tomorrow\" + 0.069*\"balrampur\" + 0.031*\"scheme\" + 0.029*\"benefit\"\n"
     ]
    }
   ],
   "source": [
    "for index, score in sorted(lda_model[bow_corpus[405]], key=lambda tup: -1*tup[1]):\n",
    "    print(\"\\nScore: {}\\t \\nTopic: {}\".format(score, lda_model.print_topic(index, 10)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
