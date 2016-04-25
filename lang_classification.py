import glob
from ldig.ldig_meltwater import lang_probabilities

class LangClassification:

  def __init__(self):
    self.load()

  def load(self):
    self.model_path = "ldig/models/model.small"

  def predict(self, text):
    text = text.get('text', '')
    return lang_probabilities(text, self.model_path)