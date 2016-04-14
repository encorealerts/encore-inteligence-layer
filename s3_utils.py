import os
import gensim

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler

from sklearn.externals import joblib
from boto.s3.connection import S3Connection

from meltwater_smart_alerts.ml.pipeline import *

def load_model_from_s3(s3_path):
  conn = S3Connection(os.environ['AWS_ACCESS_KEY_ID'], 
                      os.environ['AWS_SECRET_ACCESS_KEY'])

  bucket = conn.get_bucket(os.environ['LUIGI_S3_BUCKET'])

  models = [k for k in bucket.list() if str(k.key).startswith(s3_path)]

  MODELS_PATH = "models/"

  model_path = None
  for k in models:
    model_path = MODELS_PATH + str(k.key)
    model_dir_path = os.path.dirname(model_path)
    if not os.path.exists(model_dir_path):
      os.makedirs(model_dir_path)
    if not os.path.exists(model_path):
      k.get_contents_to_filename(model_path)

  if not model_path:
    raise Exception("No model found on S3 under '%s/%s*'" % [bucket.name, s3_path])

  return joblib.load(model_path)
