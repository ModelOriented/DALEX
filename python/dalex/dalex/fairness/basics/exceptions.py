class ParameterCheckError(Exception):

    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):

        if self.message:
            return f'Parameter Check Error, {self.message}'
        else:
            return 'Parameter Check Error has been raised'


class ModelTypeNotSupportedError(Exception):

    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):

        if self.message:
            return f'Model Type Not Supported Error, {self.message}'
        else:
            return 'Model Type Not Supported Error has been raised'


class FairnessObjectsDifferenceError(Exception):

    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):

        if self.message:
            return f'FairnessObject difference error has been raised, {self.message}'
        else:
            return 'FairnessObject difference error has been raised'
