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