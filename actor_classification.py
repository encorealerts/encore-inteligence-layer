import glob
import gc
import chardet

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.externals import joblib

from datetime import timedelta

from dateutil import relativedelta
from datetime import datetime

import pandas as pd
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

    forest_path = sorted(glob.glob('../encore-luigi/data/actor_classification/deploy/actor_classification_random_forest_2*.pkl'))
    forest_path += sorted(glob.glob('/mnt/encore-luigi/data/actor_classification/deploy/actor_classification_random_forest_2*.pkl'))

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
        encoding = chardet.detect(str(c))['encoding'] or "KOI8-R"
        return '0' if c.isdigit() else c.decode(encoding)
      except:
        return '9'

    for field in text_fields:
      data.ix[data[field].isnull(), field] = "null"
      data[field] = map(lambda n: ''.join(map(lambda c: treat_special_char(c), list(n))), data[field].values)

    # TfidfVectorizer for 'screen_name' and 'name'
    def num_char_tokenizer(text):
      return list(text)

    for field in ["screen_name","name"]:
      if field in data:
        vocabulary = filter(lambda f: f.startswith(field+"_"), self.model_features)
        vocabulary = map(lambda f: f.replace(field+"_", ""), vocabulary)
        field_tfidf = TfidfVectorizer(tokenizer=num_char_tokenizer,
                                      ngram_range=(3, 5), 
                                      analyzer="char",
                                      min_df = 1,
                                      vocabulary = vocabulary)

        field_matrix = field_tfidf.fit_transform(data[field])
        features_names = map(lambda f: "_".join([field,f]), field_tfidf.get_feature_names())
        field_df = pd.DataFrame(field_matrix.A, columns=features_names)
        gc.collect()
        data = pd.concat([data, field_df], axis=1, join='inner')
        del data[field]
        gc.collect()

    # TfidfVectorizer for 'summary'
    if "summary" in data:
      vocabulary = filter(lambda f: f.startswith("summary_"), self.model_features)
      vocabulary = map(lambda f: f.replace("summary_", ""), vocabulary)
      summary_tfidf = TfidfVectorizer(token_pattern=r'\w+',
                                      ngram_range=(1, 4), 
                                      analyzer="word",
                                      min_df = 1,
                                      vocabulary = vocabulary)

      summary_matrix = summary_tfidf.fit_transform(data.summary)
      features_names = map(lambda f: "_".join(["summary",f]), summary_tfidf.get_feature_names())
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
    
    result = self.model.predict_proba(df)[0]

    return {
      'business': result[0],
      'person': result[1]
    }
