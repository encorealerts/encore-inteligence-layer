import pandas as pd

from dateutil import relativedelta
from datetime import datetime

from s3_utils import load_model_from_s3

class ActorClassification:

  def __init__(self):
    self.load()

  def reload(self):
    d = relativedelta.relativedelta(datetime.now(), self.last_loaded)
    if d.days > 0:
      self.load()

  def load(self):
    self.last_loaded = datetime.now()
    self.model = load_model_from_s3("actor_classification/models/actor_classification_trained_model_")

  def predict(self, raw_data):
    df = pd.DataFrame([pd.Series(raw_data)])
    
    result = self.model.predict_proba(df)[0]

    return {
      'business': result[0],
      'person': result[1]
    }
