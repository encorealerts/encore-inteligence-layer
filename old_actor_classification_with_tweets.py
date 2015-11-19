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
from dateutil import parser
from datetime import datetime

import numpy as np

class ActorClassification:
  manual_generators = ['Twitter Web Client', 
                     'Twitter for iPhone', 
                     'Twitter for Android', 
                     'Twitter for BlackBerry', 
                     'Twitter for Windows Phone', 
                     'Twitter for iPad', 
                     'Twitter for BlackBerry\xc2\xae', 
                     'Twitter for Mac', 
                     'Twitter for Android Tablets', 
                     'Twitter for Windows', 
                     'Twitter for Apple Watch', 
                     'Twitter for  Android']

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

    counter_path = sorted(glob.glob('../encore-luigi/data/actor_classification/deploy/bio_count_vectorizer_*.pkl'))
    counter_path += sorted(glob.glob('/mnt/encore-luigi/data/actor_classification/deploy/bio_count_vectorizer_*.pkl'))    
    counter_path = counter_path[-1]
    vocabulary = joblib.load(counter_path) 
    self.counter = CountVectorizer(tokenizer=self.tokenize, ngram_range=(1,3), vocabulary = vocabulary)

  def tokenize(self, text):
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    tokens = [('NUM' if word.isdigit() else word) for word in tokens]
    return tokens

  def predict(self, raw_data):
    actor_summary = raw_data.get('actor_summary', '')
    actor_summary = '' if actor_summary is None else actor_summary
    transformed = self.counter.transform([actor_summary])
    transformed_features = transformed.toarray()[0]

    actor_verified = int(raw_data['actor_verified'])

    actor_followers_count = int(1 if raw_data['actor_followers_count'] is None else raw_data['actor_followers_count'])
    actor_friends_count = int(1 if raw_data['actor_friends_count'] is None else raw_data['actor_friends_count'])
    actor_favorites_count = int(1 if raw_data['actor_favorites_count'] is None else raw_data['actor_favorites_count'])
    actor_statuses_count = int(1 if raw_data['actor_statuses_count'] is None else raw_data['actor_statuses_count'])
    actor_listed_count = int(1 if raw_data['actor_listed_count'] is None else raw_data['actor_listed_count'])

    manually_tweeting = int(raw_data['tweet_generator'] in self.manual_generators)
    registration_from_now = self.actor_registration_from_now(raw_data['actor_created_at'])
    followers_friends_ratio = actor_followers_count/float(actor_friends_count)
    favourites_friends_ratio = actor_favorites_count/float(actor_friends_count)
    favourites_followers_ratio = actor_favorites_count/float(actor_followers_count)
    favourites_status_ratio = actor_favorites_count/float(actor_statuses_count)

    calculated_features = [ 
        actor_favorites_count, 
        actor_followers_count, 
        actor_friends_count, 
        actor_listed_count, 
        registration_from_now, 
        actor_statuses_count, 
        actor_verified, 
        favourites_followers_ratio, 
        favourites_friends_ratio,
        favourites_status_ratio,
        followers_friends_ratio, 
        manually_tweeting
    ]
    all_features = np.concatenate([calculated_features, transformed_features])
    
    result = self.model.predict_proba(all_features)[0]

    return {
      'business': result[0],
      'person': result[1]
    }

  def actor_registration_from_now(self, registration):
      r = parser.parse(registration)
      d = relativedelta.relativedelta(datetime.now(), r)
      return d.years * 12 + d.months