import os
import re
import fnmatch
import json
import nltk

import numpy as np
import pandas as pd
import chardet
import gc
import matplotlib.pyplot as plt

from pprint import pprint
from time import time, ctime
from datetime import datetime
from scipy.stats import randint as sp_randint

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from sklearn.lda import LDA
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

import glob
import csv
from dateutil import relativedelta
from datetime import datetime

import pandas as pd

################################################
#### SCIKIT-LEARN TRANSFORMATORS
################################################

class VerifiedTransformer(BaseEstimator, TransformerMixin):

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    X.verified.fillna(False, inplace=True)
    X.verified = LabelEncoder().fit_transform(X.verified)
    return X


class LangOneHotEncoding(BaseEstimator, TransformerMixin):

  def fit(self, X, y=None):
    valid_langs = list(set(X.lang) - set([None, np.nan, 'Select Language...']))
    self.feature_names_ = ["lang_"+str(l) for l in valid_langs if type(l) == str]
    return self

  def transform(self, X, y=None):
    check_is_fitted(self, 'feature_names_')
    
    X["lang"].fillna("", inplace=True)
    for lang_feature in self.feature_names_:
        X[lang_feature] = [(1 if lang_feature == "lang_"+v else 0) for v in X["lang"].values]
    
    X.drop(["lang"], axis=1, inplace=True)
    return X
    

class FillTextNA(BaseEstimator, TransformerMixin):

  def __init__(self, cols, replace_by=""):
    self.cols = cols
    self.replace_by = replace_by

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
        if c in X:
            X[c].fillna(self.replace_by, inplace=True)
    return X


class DataFrameTfidfVectorizer(TfidfVectorizer):

  def __init__(self, col, prefix=None, input='content', encoding='utf-8',
               decode_error='strict', strip_accents=None, lowercase=True,
               preprocessor=None, tokenizer=None, analyzer='word',
               stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
               ngram_range=(1, 1), max_df=1.0, min_df=1,
               max_features=None, vocabulary=None, binary=False,
               dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
               sublinear_tf=False):
      super(DataFrameTfidfVectorizer, self).__init__(
          input=input, encoding=encoding, decode_error=decode_error,
          strip_accents=strip_accents, lowercase=lowercase,
          preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
          stop_words=stop_words, token_pattern=token_pattern,
          ngram_range=ngram_range, max_df=max_df, min_df=min_df,
          max_features=max_features, vocabulary=vocabulary, binary=binary,
          dtype=dtype)

      self.col = col
      self.prefix = prefix or col
      
  def treat_special_char(self, c):
    try:
      encoding = chardet.detect(str(c))['encoding'] or "KOI8-R"
      return '0' if c.isdigit() else c.decode(encoding)
    except:        
      return '9'

  def treat_special_chars(self, col):
    col.fillna("null", inplace=True)
    col = [''.join([self.treat_special_char(c) for c in list(n)]) 
           for n in col.values]
    return col

  def fit(self, dataframe, y=None):
    dataframe = dataframe.copy()
    dataframe[self.col] = self.treat_special_chars(dataframe[self.col])
    super(DataFrameTfidfVectorizer, self).fit(dataframe[self.col])
    return self

  def fit_transform(self, dataframe, y=None):
    dataframe = dataframe.copy()
    dataframe[self.col] = self.treat_special_chars(dataframe[self.col])
    field_matrix = super(DataFrameTfidfVectorizer, self).fit_transform(dataframe[self.col])
    features_names = map(lambda f: "_".join([self.prefix,f]), super(DataFrameTfidfVectorizer, self).get_feature_names())
    field_df = pd.DataFrame(field_matrix.A, columns=features_names)

    dataframe = dataframe.join(field_df)

    return dataframe

  def transform(self, dataframe, copy=True):
    dataframe = dataframe.copy()
    dataframe[self.col] = self.treat_special_chars(dataframe[self.col])
    field_matrix = super(DataFrameTfidfVectorizer, self).transform(dataframe[self.col])
    features_names = map(lambda f: "_".join([self.prefix,f]), super(DataFrameTfidfVectorizer, self).get_feature_names())
    field_df = pd.DataFrame(field_matrix.A, columns=features_names)

    dataframe = dataframe.join(field_df)

    return dataframe


class TextToLowerCase(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X[c] = [t.lower() for t in X[c].values]
    return X


class NumberOfWords(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X["number_of_words_in_"+c] = [len(t.split(' ')) for t in X[c].values]
    return X


class NumberNonAlphaNumChars(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X["number_of_non_alphanum_in_"+c] = [len(re.sub(r"[\w\d]","", t)) for t in X[c].values]
    return X


class NumberUpperCaseChars(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X["number_of_upper_case_chars_in_"+c] = [len(re.sub(r"[^A-Z]","", t)) for t in X[c].values]
    return X


class NumberCamelCaseWords(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X["number_of_camel_case_words_in_"+c] = [len(re.findall(r"^[A-Z][a-z]|\s[A-Z][a-z]", t)) 
                                                 for t in X[c].values]
    return X


class NumberOfMentions(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X["number_of_mentions_in_"+c] = [len(re.findall(r"\s@[a-zA-Z]",t)) 
                                                 for t in X[c].values]
    return X


class NumberOfPeriods(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X["number_of_periods_in_"+c] = [len(t.split(". ")) 
                                        for t in X[c].values]
    return X


class AvgWordsPerPeriod(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X["avg_words_per_period_in_"+c] = [np.mean([len(p.split(" ")) for p in t.split(". ")]) 
                                            for t in X[c].values]
    return X


class MentionToFamilyRelation(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self
  
  def count_mentions(self, t):
    count = 0
    tokenizer = nltk.RegexpTokenizer(r'[a-z]+')
    for tkn in tokenizer.tokenize(t):
          if tkn in ["husband","wife","father","mother","daddy","mommy",
                     "grandfather","grandmother","grandpa","grandma"]:
                count += 1
    return count

  def transform(self, X, y=None):
    for c in self.cols:
        if c in X:
            X["mention_to_family_relation_in_"+c] = [self.count_mentions(t) 
                                                     for t in X[c].values]
    return X


class MentionToOccupation(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    occupations = pd.read_csv("https://raw.githubusercontent.com/johnlsheridan/occupations/master/occupations.csv")
    occupations = [o.lower().split(' ')[-1] for o in occupations.Occupations.values]
    self.occupations_ = dict.fromkeys(occupations, 1)
    return self
  
  def count_mentions(self, t):
    count = 0
    tokenizer = nltk.RegexpTokenizer(r'[a-z]+')
    for name in tokenizer.tokenize(t):
        count += self.occupations_.get(name, 0)
    return count

  def transform(self, X, y=None):
    check_is_fitted(self, 'occupations_')
    for c in self.cols:
        if c in X:
            X["mention_to_occupation_in_" + c] = [self.count_mentions(t) 
                                                 for t in X[c].values]
    return X
    

class PersonNames(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    female_names = pd.read_csv("http://deron.meranda.us/data/census-dist-female-first.txt", names=["name"])
    male_names   = pd.read_csv("http://deron.meranda.us/data/census-dist-male-first.txt", names=["name"])
    female_names = [re.sub(r"[^a-z]","",n.lower()) for n in female_names.name.values]
    male_names   = [re.sub(r"[^a-z]","",n.lower()) for n in male_names.name.values]        
    self.person_names_ = dict.fromkeys(set(male_names + female_names), 1)
    return self
  
  def count_mentions(self, t):
    count = 0
    tokenizer = nltk.RegexpTokenizer(r'[a-z]+')
    for name in tokenizer.tokenize(t):
        count += self.person_names_.get(name, 0)
    return count

  def transform(self, X, y=None):
    check_is_fitted(self, 'person_names_')
    for c in self.cols:
        if c in X:
            X["person_names_in_" + c] = [self.count_mentions(t) 
                                        for t in X[c].values]
    return X   

class DropColumnsTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    X = X.copy()
    for c in self.cols:
      if c in X:
        X.drop([c], axis=1, inplace=True)
    return X


class NumpyArrayTransformer(BaseEstimator, TransformerMixin):

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    X = X.copy()
    X = X.reindex_axis(sorted(X.columns), axis=1)
    X.fillna(0, inplace=True)
    return np.asarray(X)


class Debugger(BaseEstimator, TransformerMixin):

  def __init__(self, name=""):
    self.name = name

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    try:
      print X.loc[0]["screen_name"], 'step:', self.name, '-', ctime(), X.shape
    except:
      print 'step:', self.name, '-', ctime(), X.shape
    return X


class ActorClassification:

  def __init__(self):
    self.load()

  def reload(self):
    d = relativedelta.relativedelta(datetime.now(), self.last_loaded)
    if d.days > 0:
      self.load()

  def load(self):
    self.last_loaded = datetime.now()

    model_path = sorted(glob.glob('../encore-luigi/data/actor_classification/deploy/actor_classification_trained_model_*.pkl'))
    model_path += sorted(glob.glob('/mnt/encore-luigi/data/actor_classification/deploy/actor_classification_trained_model_*.pkl'))

    model_path = model_path[-1]
    self.model = joblib.load(model_path)

  def predict(self, raw_data):
    df = pd.DataFrame([pd.Series(raw_data)])
    
    result = self.model.predict_proba(df)[0]

    return {
      'business': result[0],
      'person': result[1]
    }
