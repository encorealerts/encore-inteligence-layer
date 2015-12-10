import sys

from cherrypy import wsgiserver

from actor_classification import *

from flask import Flask, jsonify, request

import traceback

# =========================
# Flask application 
# =========================

resources = {}
def create_app():
  app = Flask(__name__)
  
  resources['actor_classification'] = ActorClassification()

  return app

@app.route("/")
def hello():
    return "Hello World!\n"

@app.route("/<modelname>/predict", methods=['POST', 'OPTIONS'])
def predict(modelname):
	try:
		json_request = request.get_json(force=True)
		return jsonify(resources[modelname].predict(json_request))
	except Exception as e:
		traceback.print_exc()

# =========================
# Hosting 
# =========================

app = create_app()

if __name__ == "__main__":
  cherry_port = (5001 if len(sys.argv) == 1 else int(sys.argv[1]))
  cherry_server = wsgiserver.CherryPyWSGIServer(('0.0.0.0', cherry_port), cherry_dispatcher)
  cherry_dispatcher = wsgiserver.WSGIPathInfoDispatcher({'/': app})

  try:
    server.start()
  except KeyboardInterrupt:
    server.stop()