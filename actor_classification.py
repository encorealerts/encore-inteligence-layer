import glob

import nltk
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

from datetime import timedelta

from dateutil import relativedelta
from datetime import datetime

import numpy as np

class ActorClassification:

  def __init__(self):
    self.load()

  def reload(self):
    d = relativedelta.relativedelta(datetime.now(), self.last_loaded)
    if d.days > 0:
      self.load()

  def load(self):
    self.last_loaded = datetime.now()

    forest_path = sorted(glob.glob('../encore-luigi/data/actor_classification/deploy/actor_classification_random_forest_*.pkl'))
    forest_path += sorted(glob.glob('/mnt/encore-luigi/data/actor_classification/deploy/actor_classification_random_forest_*.pkl'))

    forest_path = forest_path[-1]
    self.model = joblib.load(forest_path)

    model_features_path = sorted(glob.glob('../encore-luigi/data/actor_classification/deploy/actor_classification_random_forest_features_*.pkl'))
    model_features_path += sorted(glob.glob('/mnt/encore-luigi/data/actor_classification/deploy/actor_classification_random_forest_features_*.pkl'))    
    model_features_path = model_features_path[-1]
    self.model_features = joblib.load(model_features_path)

  def perform_feature_engineering(self, data):
    # Remove non-relevant columns
    data = data.drop(["segment"], axis=1)
    data = data.drop(["link"], axis=1)

    # Transform boolean 'verified' to 0/1
    data.ix[data.verified.isnull(), 'verified'] = False
    data.ix[data.verified == True,  'verified'] = 1
    data.ix[data.verified == False, 'verified'] = 0

    # OneHotEncoding for 'lang'
    if "lang" in data:
      data.ix[(data.lang == 'Select Language...') | (data.lang.isnull()), 'lang'] = None
      for lang in list(set(data.lang)):
        if lang != None:
          data.ix[data.lang == lang, "lang_"+lang] = 1
          data.ix[data.lang != lang, "lang_"+lang] = 0
      data = data.drop(["lang"], axis=1)

    # Treat special characters
    text_fields = ["name", "screen_name","summary"]

    def treat_special_char(c):
      try:
        return '0' if c.isdigit() else c.decode().encode("utf-8")
      except UnicodeDecodeError:
        return '9'

    for field in text_fields:
      data.ix[data[field].isnull(), field] = "null"
      data[field] = map(lambda n: ''.join(map(lambda c: treat_special_char(c), list(n))), data[field].values)

    # CountVectorizer for 'screen_name' and 'name'
    def num_char_tokenizer(text):
      return list(text)

    for field in ["screen_name","name"]:
      if field in data:
        field_countvect = CountVectorizer(tokenizer=num_char_tokenizer,
                                          ngram_range=(3, 5), 
                                          analyzer="char",
                                          min_df = 8,
                                          vocabulary = filter(lambda f: f.startswith(field+"_"), 
                                                              self.model_features))

        field_matrix = field_countvect.fit_transform(data[field])
        features_names = map(lambda f: "_".join([field,f]), field_countvect.get_feature_names())
        field_df = pd.DataFrame(field_matrix.A, columns=features_names)

        data = pd.concat([data, field_df], axis=1, join='inner').drop([field], axis=1)

    # CountVectorizer for 'summary'
    def num_word_tokenizer(text):
      tokenizer = nltk.RegexpTokenizer(r'\w+')
      return tokenizer.tokenize(text)

    if "summary" in data:
      summary_countvect = CountVectorizer(tokenizer=num_word_tokenizer,
                                          ngram_range=(2, 4), 
                                          analyzer="word",
                                          min_df = 5,
                                          vocabulary = filter(lambda f: f.startswith("summary_"), 
                                                              self.model_features))

      summary_matrix = summary_countvect.fit_transform(data.summary)
      features_names = map(lambda f: "_".join(["summary",f]), summary_countvect.get_feature_names())
      summary_df = pd.DataFrame(summary_matrix.A, columns=features_names)
      data = pd.concat([data, summary_df], axis=1, join='inner').drop(["summary"], axis=1)

    # Treat remaining null values
    data = data.fillna(0)

    return data   

  def predict(self, raw_data):
    df = pd.DataFrame([pd.Series(raw_data)])
    df = self.perform_feature_engineering(df)
    
    result = self.model.predict_proba(df)[0]

    return {
      'business': result[0],
      'person': result[1]
    }
