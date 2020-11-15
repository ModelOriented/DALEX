from .params import ModelParam, DatasetParam, VariableParam, ObservationParam

class PlotContainer:
    def __init__(self, arena):
        if type(arena).__name__ != 'Arena' or type(arena).__module__ != 'dalex._arena.object':
            raise Exception('Invalid Arena argument')
        info = self.__class__.info
        self.name = info.get('name')
        self.plot_type = info.get('plotType')
        self.plot_component = info.get('plotType')
        self.plot_category = info.get('plotCategory')
        self.params = {}
        self.data = {}
        self.arena = arena
    def fit(self, params):
        required_params = {}
        for p in self.__class__.info.get('requiredParams'):
            self.check_param(p, params.get(p))
        for p in self.__class__.info.get('requiredParams'):
            required_params[p] = params.get(p)
            self.params[p] = params.get(p).get_label()
        self._fit(**required_params)
        return self
    def serialize(self):
        return {
            'name': self.name,
            'plotType': self.plot_type,
            'plotComponent': self.plot_component,
            'plotCategory': self.plot_category,
            'params': self.params,
            'data': self.data
        }
    def set_message(self, msg, msg_type='info'):
        if msg_type != 'info' and msg_type != 'error':
            raise Exception('Invalid message type')
        self.plot_component = 'Message'
        self.data = {'message': msg, 'type': msg_type}

    def check_param(self, param_type, value):
        correct_class = {'model': ModelParam, 'variable': VariableParam, 'observation': ObservationParam, 'dataset': DatasetParam}.get(param_type)
        if not isinstance(value, correct_class):
            raise Exception('Invalid param ' + str(param_type))
    @staticmethod
    def test_arena(arena):
        """Tests if plot can be created for at least one combination of params

        This method searches for params, that can produce valid chart. Displaying
        error messages are not counted as valid. One example of usage are charts for
        classification models. Such charts should override this method and check if
        there is at least one classification model in arena.

        Parameters
        -----------
        arena : dalex.Arena
            Object of class dalex.Arena

        Returns
        -----------
        bool
        """
        return True
