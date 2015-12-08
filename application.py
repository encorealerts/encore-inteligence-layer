import sys
import cherrypy as cp
from cherrypy.wsgiserver import CherryPyWSGIServer
from cherrypy.process.servers import ServerAdapter

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

app = create_app()

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

def run_decoupled(app, port, host='0.0.0.0', **config):
  server = CherryPyWSGIServer((host, port), app, **config)
  try:
      server.start()
  except KeyboardInterrupt:
      server.stop()

def run_in_cp_tree(app, port, host='0.0.0.0', **config):
  cp.tree.graft(app, '/')
  cp.config.update(config)
  cp.config.update({
      'server.socket_port': port,
      'server.socket_host': host
  })
  cp.engine.signals.subscribe() # optional
  cp.engine.start()
  cp.engine.block()

def run_with_adapter(app, port, host='0.0.0.0', config=None, **kwargs):
  cp.server.unsubscribe()
  bind_addr = (host, port)
  cp.server = ServerAdapter(cp.engine,
                            CherryPyWSGIServer(bind_addr, app, **kwargs),
                            bind_addr).subscribe()
  if config:
    cp.config.update(config)
  cp.engine.signals.subscribe() # optional
  cp.engine.start()
  cp.engine.block()

if __name__ == "__main__":
  port = (5001 if len(sys.argv) == 1 else int(sys.argv[1]))
  run_with_adapter(app, port=port)
