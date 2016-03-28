#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ldig server
# This code is available under the MIT License.
# (c)2011 Nakatani Shuyo / Cybozu Labs Inc.

import json
import numpy
import ldig
import codecs

class Detector(object):
  def __init__(self, modeldir):
    self.ldig = ldig.ldig(modeldir)
    self.features = self.ldig.load_features()
    self.trie = self.ldig.load_da()
    self.labels = self.ldig.load_labels()
    self.param = numpy.load(self.ldig.param)

  def detect(self, st):
    label, text, org_text = ldig.normalize_text(st)
    events = self.trie.extract_features(u"\u0001" + text + u"\u0001")
    sum = numpy.zeros(len(self.labels))

    data = []
    for id in sorted(events, key=lambda id:self.features[id][0]):
      phi = self.param[id,]
      sum += phi * events[id]
      data.append({"id":int(id), "feature":self.features[id][0], "phi":["%0.3f" % x for x in phi]})
    exp_w = numpy.exp(sum - sum.max())
    prob = exp_w / exp_w.sum()
    return {"labels":self.labels, "data":data, "prob":["%0.3f" % x for x in prob]}


def lang_probabilities(text, model):
  detector = Detector(model)
  result = detector.detect(text)
  result = zip(result["labels"], result["prob"])
  result = filter(lambda (lb, pb): float(pb) > 0.001, result)
  return dict(result)


# JUST FOR TESTING
if __name__ == '__main__':
  with codecs.open("data/instagram_activities_bodies.csv", 'rb',  'utf-8') as f:
    for text in f:
  # for text in ["Beautiful sight from the Chateau de la Fondue!",
  #              "I really loved the sight from La Casa de Los Hermanos!",
  #              "Could you tell me where is the Chateau de la Fondue? Je suis Charlie!",
  #              "Could you tell me where is La Casa de Los Hermanos?"]:
      print text
      print lang_probabilities(text, 'models/model.small')
