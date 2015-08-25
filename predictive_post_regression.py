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
    native_id            = raw_data['native_id']
    time_series_retweets = raw_data['time_series_retweets']
    
    result = self.model.predict_proba(time_series_retweets)[0]

    return {
      'engagement': result[0]
    }