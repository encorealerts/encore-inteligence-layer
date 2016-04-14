import pandas as pd

from s3_utils import load_model_from_s3

class InstagramSpam:

  def __init__(self):
    self.load()

  def load(self):
    self.model = load_model_from_s3("spam_classification/models/instagram_spam_classification_trained_model_")

  def predict(self, json):
    data = pd.DataFrame([{'body': json['text']}])
    
    print 'data:', data
    result = self.model.predict_proba(data)[0]
    result = {'ham_proba': float(result[0]), 'spam_proba': float(result[1]), 'spam': repr(result[1] > result[0]).lower()}
    print 'result:', result

    return result