import glob

from sklearn.externals import joblib

from dateutil import relativedelta
from datetime import datetime

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

    model_path = sorted(glob.glob('../encore-luigi/data/actor_classification/deploy/actor_classification_random_forest_2*.pkl'))
    model_path += sorted(glob.glob('/mnt/encore-luigi/data/actor_classification/deploy/actor_classification_random_forest_2*.pkl'))

    model_path = model_path[-1]
    self.model = joblib.load(model_path)

  def predict(self, raw_data):
    df = pd.DataFrame([pd.Series(raw_data)])
    df = self.perform_feature_engineering(df)
    
    result = self.model.predict_proba(df)[0]

    return {
      'business': result[0],
      'person': result[1]
    }
