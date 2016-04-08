import sys
import time

from cherrypy import wsgiserver

from actor_classification import *
from instagram_spam import *
from lang_classification import LangClassification

from flask import Flask, jsonify, request

import traceback

# =========================
# Flask application 
# =========================

resources = {}
def create_app():
  app = Flask(__name__)

  print 'Inteligence Layer - Initializing ...'
  
  resources['actor_classification'] = ActorClassification()
  print '...'
  resources['instagram_spam'] = InstagramSpam()
  print '...'
  resources['language_classification'] = LangClassification()

  print 'Inteligence Layer - App initialized ...'

  return app

app = create_app()

class Timer(object):
  def __init__(self, verbose=False):
    self.verbose = verbose  

  def __enter__(self):
    self.start = time()
    return self

  def __exit__(self, *args):
    self.end = time()
    self.secs = self.end - self.start
    self.msecs = self.secs * 1000  # millisecs
    if self.verbose:
      print 'elapsed time: %f ms' % self.msecs

@app.route("/")
def hello():
    return "Hello World!\n"

@app.route("/<modelname>/predict", methods=['POST', 'OPTIONS'])
def predict(modelname):
  try:
    json_request = request.get_json(force=True)
    json_predict = None 

    with Timer() as t:
      json_predict = jsonify(resources[modelname].predict(json_request))
    print json_request.get('screen_name', modelname), "predict:", t.secs

    return json_predict
  except Exception as e:
    traceback.print_exc()

# =========================
# Hosting 
# =========================

if __name__ == "__main__":
  cherry_port = (5001 if len(sys.argv) == 1 else int(sys.argv[1]))
  cherry_dispatcher = wsgiserver.WSGIPathInfoDispatcher({'/': app})
  cherry_server = wsgiserver.CherryPyWSGIServer(('0.0.0.0', cherry_port), cherry_dispatcher, numthreads=100)
  cherry_server.thread_pool = 100

  try:
    cherry_server.start()
  except KeyboardInterrupt:
    cherry_server.stop()
