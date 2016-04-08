import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler

import glob
from dateutil import relativedelta
from datetime import datetime

import pandas as pd

from meltwater_smart_alerts.ml.pipeline import *

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
