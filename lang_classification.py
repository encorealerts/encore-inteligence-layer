import glob
from ldig.ldig_meltwater import lang_probabilities

class LangClassification:

  def __init__(self):
    self.load()

  def load(self):
    model_path = sorted(glob.glob('../encore-inteligence-layer/ldig/models/model.small'))
    model_path += sorted(glob.glob('/mnt/encore-inteligence-layer/ldig/models/model.small'))

    self.model_path = model_path[-1]

  def predict(self, text):
  	text = text.get('text', '')
    return lang_probabilities(text, self.model_path)