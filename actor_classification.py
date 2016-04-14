import os

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

from boto.s3.connection import S3Connection

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

    conn = S3Connection(os.environ['AWS_ACCESS_KEY_ID'], 
                        os.environ['AWS_SECRET_ACCESS_KEY'])

    bucket = conn.get_bucket(os.environ['LUIGI_S3_BUCKET'])

    models = [k for k in bucket.list() if str(k.key).startswith("actor_classification/models/actor_classification_trained_model_")]

    MODELS_PATH = "models/"

    for k in models:
      model_path = MODELS_PATH + str(k.key)
      model_dir_path = os.path.dirname(model_path)
      if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
      if not os.path.exists(model_path):
        k.get_contents_to_filename(model_path)

    if not model_path:
      raise Exception("No model found on S3 under '%s/actor_classification/models'" % bucket.name)

    self.model = joblib.load(model_path)

  def predict(self, raw_data):
    df = pd.DataFrame([pd.Series(raw_data)])
    
    result = self.model.predict_proba(df)[0]

    return {
      'business': result[0],
      'person': result[1]
    }
