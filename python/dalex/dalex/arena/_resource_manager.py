from ._resource import Resource
from . import resources


class ResourceManager:
    """Creates ResourceManager object

    This class should be only created by arena instance to manage its resources.

    Parameters
    ----------
    arena : dalex.Arena
        Instance of Arena.

    Attributes
    --------
    arena : Arena
        Instance of dalex.Arena
    cache : list of Resource objects
        List of already requested resources (both calculated and in progress)
    mutex : _thread.lock
        Mutex for params, plots and resources cache. Should be common with mutex from Arena instance.
    resources : list of classes extending PlotContainer
        List of available resource classes
    """
    def __init__(self, arena):
        if type(arena).__name__ != 'Arena' or type(arena).__module__ != 'dalex.arena.object':
            raise Exception('Invalid Arena argument')
        self.arena = arena
        self.cache = []
        self.mutex = arena.mutex
        self.resources = [vars(resources)[res] for res in getattr(resources, '__all__')]

    def find_in_cache(self, resource_type, params):
        """Function searches for cached resource

        Parameters
        -----------
        resource_type : str
            Value of resource_type field, that requested resource must have
        params : dict
            Keys of this dict are params types (model, observation, variable, dataset)
            and values are corresponding params labels. Requested resource must have equal
            params field.

        Returns
        --------
        Resource or None
        """

        def _filter(p):
            return p.resource_type == resource_type and params == p.params
        with self.mutex:
            return next(filter(_filter, self.cache), None)

    def put_to_cache(self, resource):
        """Puts new resource to cache

        Parameters
        -----------
        resource : Resource
        """
        if not isinstance(resource, Resource):
            raise Exception('Invalid resource')
        with self.mutex:
            self.cache.append(resource)

    def clear_cache(self, resource_type=None):
        """Clears cache
        
        Parameters
        -----------
        resource_type : str or None
            If None all cache is cleared. Otherwise only resources with
            provided resource_type are removed.

        Notes
        -------
        This function must be called from mutex context
        """
        if resource_type is None:
            for r in self.cache:
                r.cancel()
            self.cache = []
        else:
            for r in self.cache:
                if r.resource_type == resource_type:
                    r.cancel()
            self.cache = list(filter(lambda r: r.resource_type != resource_type, self.cache))
        self.arena.update_timestamp()

    def get_resource(self, resource_type, params_values, cache=True):
        """Returns resource for specified type and params

        Function serches for resource in cache, when not present creates
        requested resource and put it to cache.

        Parameters
        -----------
        resource_type : str
            Type of resource to be generated
        params_values : dict
            Dict for param types as keys and Param objects as values
        cache : bool
            If search for resource in cache and put calculated resource into cache.

        Returns
        --------
        PlotContainer
        """
        resource_class = next((c for c in self.resources if c.resource_type == resource_type), None)
        if resource_class is None:
            raise Exception('Not supported resource type')
        required_params_values = {}
        required_params_labels = {}
        for p in resource_class.required_params:
            if params_values.get(p) is None:
                raise Exception('Required param is missing')
            required_params_values[p] = params_values.get(p)
            required_params_labels[p] = params_values.get(p).get_label()
        result = self.find_in_cache(resource_type, required_params_labels) if cache else None
        if result is None:
            result = resource_class(self.arena).fit(required_params_values)
            if cache:
                self.put_to_cache(result)
            else:
                # while is_done is False
                while not result.get_result()[2]:
                    result.wait_for_update()
        return result
