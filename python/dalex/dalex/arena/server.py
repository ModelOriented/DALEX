import logging
import sys
import json
import numpy as np
import random
import pandas as pd
from .params import ObservationParam

def convert(o):
    if isinstance(o, np.generic): return o.item()  
    raise TypeError

def start_server(arena, host, port, disable_logs):
    from flask import Flask, request, abort, Response
    from werkzeug.serving import ThreadedWSGIServer
    from flask_cors import CORS
    import requests

    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None
    app = Flask(__name__)
    CORS(app)
    wsgi_server = ThreadedWSGIServer(host=host, port=port, app=app)
    shutdown_token = str(random.randrange(2<<63))
    
    log = logging.getLogger('werkzeug')
    log.disabled = disable_logs
    app.logger.disabled = disable_logs

    arena._stop_server = lambda: requests.get('http://' + host + ':' + str(port) + '/shutdown?token=' + shutdown_token)

    @app.route("/", methods=['GET'])
    def main():
        result = {
            'version': '1.2.0',
            'api': 'arenar_api',
            'timestamp': arena.timestamp*1000,
            'availableParams': arena.list_available_params(),
            'availablePlots': [plot.info for plot in arena.get_supported_plots()],
            'options': { 'attributes': arena.enable_attributes, 'customParams': arena.enable_custom_params }
        }
        return Response(json.dumps(result, default=convert), content_type='application/json')

    def get_params(request):
        result = {}
        custom_params = False
        for param_type in ['model', 'observation', 'variable', 'dataset']:
            param_label = request.args.get(param_type)
            if (not param_label is None) and param_label.startswith('{') and param_label.endswith('}') and param_type == 'observation':
                custom_params = True
                try:
                    obj = json.loads(param_label)
                    data = result['model'].explainer.data if not result['model'] is None else None
                    if data is None:
                        raise Exception('Custom observation requires model param')
                    df = pd.concat([
                        data.head(n=0), 
                        pd.DataFrame({ k: v for k, v in obj.items() if k in data.columns }, index=[0])
                    ], ignore_index=True).tail(n=1).reset_index(drop=True)
                    param = ObservationParam(df, 0)
                    result[param_type] = param
                except:
                    raise Exception('Invalid custom observation')
            else:
                param_value = arena.find_param_value(param_type, request.args.get(param_type))
                if not param_value is None:
                    result[param_type] = param_value
        return (result, custom_params)

    @app.route("/<string:plot_type>", methods=['GET'])
    def get_plot(plot_type):
        if plot_type == 'timestamp':
            return Response(json.dumps({'timestamp': arena.timestamp * 1000}, default=convert), content_type='application/json')
        elif plot_type == 'shutdown':
            if request.args.get('token') != shutdown_token:
                abort(403)
                return
            wsgi_server.shutdown()
            return ''
        params, custom_params = get_params(request)
        if not arena.enable_custom_params and custom_params:
            abort(403)
            return
        try:
            result = arena.plots_manager.get_plot(plot_type, params, cache=not custom_params)
            return Response(json.dumps(result.serialize(), default=convert), content_type='application/json')
        except Exception as e:
            abort(404)
            return

    @app.route("/attribute/<string:param_type>/<string:param_label>", methods=['GET'])
    def get_attribute(param_type, param_label):
        if not param_type in ['model', 'variable', 'observation', 'dataset']:
            abort(404)
            return
        result = arena.get_param_attributes(param_type, param_label)
        return Response(json.dumps(result, default=convert), content_type='application/json')

    wsgi_server.serve_forever()
