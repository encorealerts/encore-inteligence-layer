import glob

from sklearn.externals import joblib

from datetime import timedelta

from dateutil import relativedelta
from dateutil import parser
from datetime import datetime

import numpy as np

class PredictivePostRegression:

  def __init__(self):
    self.load()

  def reload(self):
    d = relativedelta.relativedelta(datetime.now(), self.last_loaded)
    if d.days > 0:
      self.load()

  def load(self):
    self.last_loaded = datetime.now()

    lr_model_path = sorted(glob.glob('../encore-luigi/data/predictive_post_regression/deploy/predictive_post_regression_lr_model_*.pkl'))
    lr_model_path += sorted(glob.glob('/mnt/encore-luigi/data/predictive_post_regression/deploy/predictive_post_regression_lr_model_*.pkl'))

    lr_model_path = lr_model_path[-1]
    self.model = joblib.load(lr_model_path)

  def predict(self, raw_data):
    actor_summary = raw_data.get('actor_summary', '')
    transformed = self.counter.transform([actor_summary])
    transformed_features = transformed.toarray()[0]

    actor_verified = int(raw_data['actor_verified'])

    actor_followers_count = int(raw_data['actor_followers_count'])
    actor_friends_count = int(raw_data['actor_friends_count'])
    actor_favorites_count = int(raw_data['actor_favorites_count'])
    actor_statuses_count = int(raw_data['actor_statuses_count'])
    actor_listed_count = int(raw_data['actor_listed_count'])

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
        followers_friends_ratio
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