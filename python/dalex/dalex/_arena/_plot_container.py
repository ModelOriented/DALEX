class PlotContainer:
    def __init__(self,
                 arena,
                 name='',
                 plot_type='',
                 plot_component='',
                 plot_category='',
                 params={},
                 data={}):
        if type(arena).__name__ != 'Arena' or type(arena).__module__ != 'dalex._arena.object':
            raise Exception('Invalid Arena argument')
        self.name = name
        self.plot_type = plot_type
        self.plot_component = plot_component
        self.plot_category = plot_category
        self.params = params
        self.data = data
        self.options = arena.options.get(plot_type)
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
