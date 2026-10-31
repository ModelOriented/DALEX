from .._dataframe import column_names, frame_from_values
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


def serialize_json(value):
    """Represent nonfinite display values as JSON null without editing sources."""
    def normalize(item):
        import datetime
        if item is pd.NA or item is pd.NaT:
            return None
        if isinstance(item, (np.datetime64, np.timedelta64)):
            return None if np.isnat(item) else str(item)
        if isinstance(item, (datetime.datetime, datetime.date)):
            return item.isoformat()
        if isinstance(item, datetime.timedelta):
            return str(item)
        if isinstance(item, np.ndarray):
            item = item.tolist()
        elif isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: normalize(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(value) for value in item]
        return item
    return json.dumps(normalize(value), default=convert, allow_nan=False)

def start_server(arena, host, port, disable_logs, ready=None):
    from flask import Flask, request, abort, Response
    from werkzeug.serving import ThreadedWSGIServer
    from flask_cors import CORS
    import requests
    from socketserver import TCPServer

    class ArenaWSGIServer(ThreadedWSGIServer):
        def server_bind(self):
            # HTTPServer's default reverse DNS lookup can block local startup
            # when the host resolver is unavailable. WSGI accepts the bound IP.
            TCPServer.server_bind(self)
            self.server_name = self.server_address[0]
            self.server_port = self.server_address[1]

    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None
    app = Flask(__name__)
    CORS(app)
    wsgi_server = ArenaWSGIServer(host=host, port=port, app=app)
    shutdown_token = str(random.randrange(2<<63))
    
    log = logging.getLogger('werkzeug')
    log.disabled = disable_logs
    app.logger.disabled = disable_logs

    arena._stop_server = lambda: requests.get('http://' + host + ':' + str(port) + '/shutdown?token=' + shutdown_token, timeout=10)

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
        return Response(serialize_json(result), content_type='application/json')

    def get_params(request):
        result = {}
        custom_params = False
        for param_type in ['model', 'observation', 'variable', 'dataset']:
            param_label = request.args.get(param_type)
            if (not param_label is None) and param_label.startswith('{') and param_label.endswith('}') and param_type == 'observation':
                custom_params = True
                try:
                    obj = json.loads(param_label)
                    reference = result['model'].explainer.data if not result['model'] is None else None
                    if reference is None:
                        raise Exception('Custom observation requires model param')
                    if isinstance(reference, pd.DataFrame):
                        df = pd.concat([
                            reference.head(n=0),
                            pd.DataFrame({k: v for k, v in obj.items() if k in reference.columns}, index=[0])
                        ], ignore_index=True).tail(n=1).reset_index(drop=True)
                        param = ObservationParam(df, 0)
                    else:
                        names = list(column_names(reference))
                        values = np.empty((1, len(names)), dtype=object)
                        for i, name in enumerate(names):
                            values[0, i] = obj.get(name)
                        native = frame_from_values(values, reference)
                        from .params import ObservationDataset
                        param = ObservationParam(ObservationDataset(native, index=[0]), 0, native_dataset=native, position=0)
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
            return Response(serialize_json({'timestamp': arena.timestamp * 1000}), content_type='application/json')
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
            return Response(serialize_json(result.serialize()), content_type='application/json')
        except Exception as e:
            abort(404)
            return

    @app.route("/attribute/<string:param_type>/<string:param_label>", methods=['GET'])
    def get_attribute(param_type, param_label):
        if not param_type in ['model', 'variable', 'observation', 'dataset']:
            abort(404)
            return
        result = arena.get_param_attributes(param_type, param_label)
        return Response(serialize_json(result), content_type='application/json')

    if ready is not None:
        ready.set()
    try:
        wsgi_server.serve_forever()
    finally:
        wsgi_server.server_close()
