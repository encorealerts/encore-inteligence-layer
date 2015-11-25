import json

import glob
import gc

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
import pandas as pd

class ActorClassification:

  def __init__(self):
    self.load()

  def reload(self):
    d = relativedelta.relativedelta(datetime.now(), self.last_loaded)
    if d.days > 0:
      self.load()

  def load(self):
    self.last_loaded = datetime.now()

    forest_path = sorted(glob.glob('../encore-luigi/data/actor_classification/deploy/actor_classification_random_forest_20*.pkl'))
    forest_path += sorted(glob.glob('/mnt/encore-luigi/data/actor_classification/deploy/actor_classification_random_forest_20*.pkl'))

    forest_path = forest_path[-1]

    self.model = joblib.load(forest_path)

    model_features_path = sorted(glob.glob('../encore-luigi/data/actor_classification/deploy/actor_classification_random_forest_features_*.pkl'))
    model_features_path += sorted(glob.glob('/mnt/encore-luigi/data/actor_classification/deploy/actor_classification_random_forest_features_*.pkl'))    
    model_features_path = model_features_path[-1]
    self.model_features = joblib.load(model_features_path)

  def perform_feature_engineering(self, data):
    # Transform boolean 'verified' to 0/1
    data.ix[data.verified.isnull(), 'verified'] = False
    data.ix[data.verified == True,  'verified'] = 1
    data.ix[data.verified == False, 'verified'] = 0

    # 'lang'
    for lang_field in filter(lambda f: f.startswith("lang_"), self.model_features):
      data[lang_field] = (1 if lang_field == "lang_"+(data["lang"]).values[0] else 0)
    del data["lang"]

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
        vocabulary = filter(lambda f: f.startswith(field+"_"), self.model_features)
        vocabulary = map(lambda f: f.replace(field+"_", ""), vocabulary)
        field_countvect = CountVectorizer(tokenizer=num_char_tokenizer,
                                          ngram_range=(3, 5), 
                                          analyzer="char",
                                          min_df = 50, #8
                                          vocabulary = vocabulary)

        field_matrix = field_countvect.fit_transform(data[field])
        features_names = map(lambda f: "_".join([field,f]), field_countvect.get_feature_names())
        field_df = pd.DataFrame(field_matrix.A, columns=features_names)
        gc.collect()
        data = pd.concat([data, field_df], axis=1, join='inner')
        del data[field]
        gc.collect()

    # CountVectorizer for 'summary'
    def num_word_tokenizer(text):
      tokenizer = nltk.RegexpTokenizer(r'\w+')
      return tokenizer.tokenize(text)

    if "summary" in data:
      vocabulary = filter(lambda f: f.startswith("summary_"), self.model_features)
      vocabulary = map(lambda f: f.replace("summary_", ""), vocabulary)
      summary_countvect = CountVectorizer(tokenizer=num_word_tokenizer,
                                          ngram_range=(2, 4), 
                                          analyzer="word",
                                          min_df = 50, #5
                                          vocabulary = vocabulary)

      summary_matrix = summary_countvect.fit_transform(data.summary)
      features_names = map(lambda f: "_".join(["summary",f]), summary_countvect.get_feature_names())
      summary_df = pd.DataFrame(summary_matrix.A, columns=features_names)
      gc.collect()
      data = pd.concat([data, summary_df], axis=1, join='inner')
      del data["summary"]
      gc.collect()

    # Treat remaining null values
    data.fillna(0, inplace=True)
    gc.collect()

    return data   

  def predict(self, raw_data):
    df = pd.DataFrame([pd.Series(raw_data)])
    df = self.perform_feature_engineering(df)

    ########################################
    self_model_features = map(lambda d: d.decode(), self.model_features)
    df_columns = map(lambda d: d.decode(), df.columns)

    print len(list(set(self_model_features)))
    print len(list(set(df_columns)))
    print len(list(set(self_model_features) - set(df_columns)))

    print self_model_features[2:11]
    print df_columns[2:11]
    
    print self_model_features[-10:]
    print df_columns[-10:]

    print self_model_features[11:20]
    print df_columns[11:20]
    
    print self_model_features[-20:-10]
    print df_columns[-20:-10]

    # print list(set(filter(lambda d: not d.startswith("screen_name_") and not d.startswith("name_") and not d.startswith("summary_"), self.model_features)) -
    #            set(filter(lambda d: not d.startswith("screen_name_") and not d.startswith("name_") and not d.startswith("summary_"), df.columns)))

    ########################################
    
    result = self.model.predict_proba(df)[0]

    return {
      'business': result[0],
      'person': result[1]
    }



classifier = ActorClassification()
classifier.predict(json.loads("{\"lang\":\"en\",\"summary\":\"Artist, Writer, Designer. Tweets on tech, culture, art, animals, \\u0026 the socioeconomy.\",\"verified\":0,\"followers_count\":175,\"friends_count\":397,\"favourites_count\":228,\"statuses_count\":410,\"listed_count\":12,\"name\":\"Daniel Adornes\",\"screen_name\":\"daniel_adornes\"}"))
