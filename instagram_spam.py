import gensim
import glob
import pandas as pd

import matplotlib as plt
import numpy as np

import re

from time import ctime

from sklearn.externals import joblib

from meltwater_smart_alerts.ml.pipeline import *

class InstagramSpam:

  def __init__(self):
    self.load()

  def load(self):
    model_path = sorted(glob.glob('../encore-inteligence-layer/models/instagram_spam_model_*.pkl'))
    model_path += sorted(glob.glob('/mnt/encore-inteligence-layer/models/instagram_spam_model_*.pkl'))

    print model_path

    instagram_spam_model = model_path[-1]
    self.model = joblib.load(instagram_spam_model)

  def predict(self, json):
    data = pd.DataFrame([{'body': json['text']}])
    
    print 'data:', data
    result = self.model.predict_proba(data)[0]
    result = {'ham_proba': float(result[0]), 'spam_proba': float(result[1]), 'spam': repr(result[1] > result[0]).lower()}
    print 'result:', result

    return result