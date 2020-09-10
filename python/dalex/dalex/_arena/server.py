import logging
import sys
import json
import numpy as np
from flask import Flask, request, abort
from flask_cors import CORS
import requests
import random
import traceback

def convert(o):
    if isinstance(o, np.generic): return o.item()  
    raise TypeError

def start_server(arena, host, port, disable_logs):
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None
    app = Flask(__name__)
    CORS(app)
    shutdown_token = str(random.randrange(2<<63))
    plots = arena.plots
    
    log = logging.getLogger('werkzeug')
    log.disabled = disable_logs
    app.logger.disabled = disable_logs

    arena._stop_server = lambda: requests.get('http://' + host + ':' + str(port) + '/shutdown?token=' + shutdown_token)

    @app.route("/", methods=['GET'])
    def main():
        return {
            'version': '1.1.0',
            'api': 'arenar_api',
            'timestamp': arena.timestamp*1000,
            'availableParams': arena.list_available_params(),
            'availablePlots': [plot.info for plot in plots]
        }

    def get_params(request):
        result = {}
        for param_type in ['model', 'observation', 'variable', 'dataset']:
            param_value = arena.find_param_value(param_type, request.args.get(param_type))
            if not param_value is None:
                result[param_type] = param_value
        return result

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
        params = get_params(request)
        try:
            result = arena.get_plot(plot_type, params)
        except Exception:
            abort(404)
            return
        return json.dumps(result.serialize(), default=convert)

    app.run(host=host, port=port)
