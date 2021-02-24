class OptionBase:
    """
    Base class providing methods for options. The goal of it is to
    create common interface for PlotContainer and Resource.
    """
    options_category = "base"
    options = {}
    def __init__(self, arena):
        if type(arena).__name__ != 'Arena' or type(arena).__module__ != 'dalex.arena.object':
            raise Exception('Invalid Arena argument')
        self.arena = arena

    def get_option(self, name):
        return self.arena.get_option(self.__class__.options_category, name)

    def set_option(self, name, value):
        return self.arena.set_option(self.__class__.options_category, name, value)
