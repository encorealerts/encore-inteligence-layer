from actor_classification import ActorClassification

from flask import Flask, jsonify, request

import traceback

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


if __name__ == "__main__":
    app.run()