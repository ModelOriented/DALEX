import logging
import sys
import json
import numpy as np
from flask import Flask, request, abort
from flask_cors import CORS
import requests
import random

def convert(o):
    if isinstance(o, np.generic): return o.item()  
    raise TypeError

def start_server(arena, host, port):
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None
    app = Flask(__name__)
    CORS(app)
    shutdown_token = str(random.randrange(2<<63))
    plots = arena.plots

    log = logging.getLogger('werkzeug')
    log.disabled = True
    app.logger.disabled = True

    arena._stop_server = lambda: requests.get('http://' + host + ':' + str(port) + '/shutdown?token=' + shutdown_token)

    @app.route("/", methods=['GET'])
    def main():
        return {
            'version': '1.1.0',
            'api': 'arenar_api',
            'timestamp': arena.timestamp*1000,
            'availableParams': {
                'observation': arena.list_observations(),
                'variable': arena.list_variables(),
                'model': arena.list_models(),
                'dataset': arena.list_datasets()
            },
            'availablePlots': [plot.info for plot in plots]
        }

    def get_explainer(request):
        name = request.args.get('model')
        return next((x for x in arena.get_models() if x.label == name), None)

    def get_observation(request):
        name = request.args.get('observation')
        observation = next((df.loc[name] for df in arena.get_observations() if len(df.index.intersection([name])) > 0), None)
        return observation

    def get_variable(request):
        variable = request.args.get('variable')
        if variable in arena.list_variables():
            return variable
        return None

    def get_params(request):
        return {
            'model': get_explainer(request),
            'observation': get_observation(request),
            'variable': get_variable(request)
        }

    @app.route("/<string:plot_type>", methods=['GET'])
    def get_plot(plot_type):
        if plot_type == 'timestamp':
            return {'timestamp': arena.timestamp * 1000}
        elif plot_type == 'shutdown':
            if request.args.get('token') != shutdown_token:
                abort(403)
                return
            shutdown = request.environ.get('werkzeug.server.shutdown')
            if shutdown is None:
                raise Exception('Failed to stop the server.')
            shutdown()
            return ''
        plot_class = next((c for c in plots if c.info.get('plotType') == plot_type), None)
        if plot_class is None:
            abort(404)
            return
        params = get_params(request)
        kwargs = {'arena': arena}
        for p in plot_class.info.get('requiredParams'):
            if params.get(p) is None:
                abort(404)
                return
            kwargs[p] = params.get(p)
        result = plot_class(**kwargs).serialize()
        return json.dumps(result, default=convert)

    app.run(host=host, port=port)
