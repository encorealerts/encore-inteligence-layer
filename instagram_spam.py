import gensim
import glob
import pandas as pd

import matplotlib as plt
import numpy as np

import re

from sklearn.externals import joblib

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

# - - - - - - - - -
# +++===========+++
# - - - - - - - - -

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
#     dataframe[self.col] = self.treat_special_chars(dataframe[self.col])
    field_matrix = super(DataFrameTfidfVectorizer, self).fit_transform(dataframe[self.col])
    features_names = map(lambda f: "_".join([self.prefix,f]), super(DataFrameTfidfVectorizer, self).get_feature_names())
    field_df = pd.DataFrame(field_matrix.A, columns=features_names)

    dataframe = dataframe.join(field_df)

    return dataframe

  def transform(self, dataframe, copy=True):
    dataframe = dataframe.copy()
#     dataframe[self.col] = self.treat_special_chars(dataframe[self.col])
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
        X.loc[:, c] = [t.lower() for t in X[c].values]
    return X


class NumberOfWords(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X.loc[:, "number_of_words_in_"+c] = [len(t.split(' ')) for t in X[c].values]
    return X


class NumberNonAlphaNumChars(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X.loc[:, "number_of_non_alphanum_in_"+c] = [len(re.sub(r"[\w\d]","", t)) for t in X[c].values]
    return X


class NumberUpperCaseChars(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X.loc[:, "number_of_upper_case_chars_in_"+c] = [len(re.sub(r"[^A-Z]","", t)) for t in X[c].values]
    return X


class NumberCamelCaseWords(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X.loc[:, "number_of_camel_case_words_in_"+c] = [len(re.findall(r"^[A-Z][a-z]|\s[A-Z][a-z]", t)) 
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
        X.loc[:, "number_of_mentions_in_"+c] = [len(re.findall(r"\s@[a-zA-Z]",t)) 
                                                 for t in X[c].values]
    return X

class NumberOfHashtags(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X.loc[:, "number_of_mentions_in_"+c] = [len(re.findall(r"#[a-zA-Z\d]+",t)) 
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
        X.loc[:, "number_of_periods_in_"+c] = [len(t.split(". ")) 
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
        X.loc[:, "avg_words_per_period_in_"+c] = [np.mean([len(p.split(" ")) for p in t.split(". ")]) 
                                            for t in X[c].values]
    return X

class DropColumnsTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X.drop([c], axis=1, inplace=True)
    return X


class NumpyArrayTransformer(BaseEstimator, TransformerMixin):

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    X = X.reindex_axis(sorted(X.columns), axis=1)
    X.fillna(0, inplace=True)
    return np.asarray(X)
  
class HashtagDistanceTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, model, cols):
    self.model = model
    self.cols = cols
  
  def fit(self, model, X, y=None):
    return self
  
  def transform(self, X, y=None):
    for c in self.cols:
      if c in X:
        X = X.join(X[c].apply(lambda value: self.hashtags_distance(self.model, value)))
        print X.columns
    return X
  
  def hashtags_distance(self, model, value):
    if value is None: return {'min_distance': -999, 'max_distance': -999, 'avg_distance': -999}
    
    hashtags = re.findall(r"#[a-zA-Z\d]+", value)
                          
    _min, _max, _avg = self.distance(model, hashtags)
                            
    return pd.Series({'min_distance': _min, 'max_distance': _max, 'avg_distance': _avg})

  def distance(self, model, array):
    _itens = 0
    _sum = 0
    _min = 1000
    _max = 0

    if len(array) <= 1:
        return 0,0,0

    for i1 in range(0, len(array)):
        for i2 in range(0, len(array)):
            if i1 == i2: 
                continue

            r = 0
            try:
              r = model.similarity(array[i1], array[i2])
              _sum += r
              if r > _max:
                  _max = r
              if r < _min: 
                  _min = r

              _itens += 1
            except:
              continue
            
    return _min, _max, _sum/_itens

class Debugger(BaseEstimator, TransformerMixin):
  def __init__(self, name=""):
      self.name = name

  def fit(self, X, y=None):
      return self

  def transform(self, X, y=None):
      print self.name, '-', ctime(), X.shape
      return X

# - - - - - - - - -
# +++===========+++
# - - - - - - - - -

class InstagramSpam:

  def __init__(self):
    self.load()

  def load(self):
    model_path = sorted(glob.glob('../encore-inteligence-layer/models/instagram_spam_model_*.pkl'))
    model_path += sorted(glob.glob('/mnt/encore-inteligence-layer/models/instagram_spam_model_*.pkl'))

    print model_path

    instagram_spam_model = model_path[-1]
    self.model = joblib.load(instagram_spam_model)

  def predict(self, json):
    data = pd.DataFrame([{'body': json['text']}])
    
    print 'data:', data
    result = self.model.predict_proba(data)[0]
    result = {'ham_proba': float(result[0]), 'spam_proba': float(result[1]), 'spam': repr(result[1] > result[0]).lower()}
    print 'result:', result

    return result