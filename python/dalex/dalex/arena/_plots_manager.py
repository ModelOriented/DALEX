import time
from . import plots
from ._plot_container import PlotContainer


class PlotsManager:
    """Creates PlotsManager object

    This class should be only created by arena instance to manage its plots.

    Parameters
    ----------
    arena : dalex.Arena
        Instance of Arena.

    Attributes
    --------
    arena : Arena
        Instance of dalex.Arena
    cache : list of PlotContainer objects
        List of already calculated plots
    mutex : _thread.lock
        Mutex for params, plots and resources cache. Copied from Arena instance.
    plots : list of classes extending PlotContainer
        List of enabled plots
    """
    def __init__(self, arena):
        if type(arena).__name__ != 'Arena' or type(arena).__module__ != 'dalex.arena.object':
            raise Exception('Invalid Arena argument')
        self.arena = arena
        self.cache = []
        self.mutex = arena.mutex
        self.plots = [vars(plots)[res] for res in getattr(plots, '__all__')]

    def get_supported_plots(self):
        """Returns plots classes that can produce at least one valid chart for parent arena.

        Returns
        -----------
        List of classes extending PlotContainer
        """
        return [plot for plot in self.plots if plot.test_arena(self.arena)]

    def clear_cache(self, plot_type=None):
        """Clears cache

        Parameters
        -----------
        plot_type : str or None
            If None all cache is cleared. Otherwise only plots with
            provided plot_type are removed.

        Notes
        -------
        This function must be called from mutex context
        """
        if plot_type is None:
            self.cache = []
        else:
            self.cache = list(filter(lambda p: p.plot_type != plot_type, self.cache))
        self.arena.update_timestamp()

    def find_in_cache(self, plot_type, params):
        """Function searches for cached plot

        Parameters
        -----------
        plot_type : str
            Value of plot_type field, that requested plot must have
        params : dict
            Keys of this dict are params types (model, observation, variable, dataset)
            and values are corresponding params labels. Requested plot must have equal
            params field.

        Returns
        --------
        PlotContainer or None
        """

        def _filter(p):
            return p.plot_type == plot_type and params == p.params
        with self.mutex:
            return next(filter(_filter, self.cache), None)

    def put_to_cache(self, plot_container):
        """Puts new plot to cache

        Parameters
        -----------
        plot_container : PlotContainer
        """
        if not isinstance(plot_container, PlotContainer):
            raise Exception('Invalid plot container')
        with self.mutex:
            self.cache.append(plot_container)

    def fill_cache(self, fixed_params={}):
        """Generates all available plots and cache them

        This function tries to generate all plots that are not cached already and
        put them to cache. Range of generated plots can be narrow using `fixed_params`

        Parameters
        -----------
        fixed_params : dict
            This dict specifies which plots should be generated. Only those with
            all keys from `fixed_params` present and having the same value will be
            calculated.
        """
        if not isinstance(fixed_params, dict):
            raise Exception('Params argument must be a dict')
        for plot_class in self.get_supported_plots():
            required_params = plot_class.info.get('requiredParams')
            # Test if all params fixed by user are used in this plot. If not, then skip it.
            # This list contains fixed params' types, that are not required by plot.
            # Loop will be skipped if this list is not empty.
            if len([k for k in fixed_params.keys() if k not in required_params]) > 0:
                continue
            available_params = self.arena.get_available_params()
            iteration_pools = map(lambda p: available_params.get(p) if fixed_params.get(p) is None else [fixed_params.get(p)], required_params)
            combinations = [[]]
            for pool in iteration_pools:
                combinations = [x + [y] for x in combinations for y in pool]
            for params_values in combinations:
                params = dict(zip(required_params, params_values))
                self.get_plot(plot_type=plot_class.info.get('plotType'), params_values=params, wait=True)

    def get_plot(self, plot_type, params_values, cache=True, wait=False):
        """Returns plot for specified type and params

        Function serches for plot in cache, when not present creates
        requested plot and put it to cache.

        Parameters
        -----------
        plot_type : str
            Type of plot to be generated
        params_values : dict
            Dict for param types as keys and Param objects as values
        cache : bool
            If serach for plot in cache and put calculated plot into cache.

        Returns
        --------
        PlotContainer
        """
        plot_class = next((c for c in self.plots if c.info.get('plotType') == plot_type), None)
        if plot_class is None:
            raise Exception('Not supported plot type')
        plot_type = plot_class.info.get('plotType')
        required_params_values = {}
        required_params_labels = {}
        for p in plot_class.info.get('requiredParams'):
            if params_values.get(p) is None:
                raise Exception('Required param is missing')
            required_params_values[p] = params_values.get(p)
            required_params_labels[p] = params_values.get(p).get_label()
        result = self.find_in_cache(plot_type, required_params_labels) if cache else None
        if result is None:
            result = plot_class(self.arena, cache=cache).fit(required_params_values)
            while wait and result.is_done == False:
                time.sleep(0.5)
                result = plot_class(self.arena, cache=cache).fit(required_params_values)
            if cache and result.is_done != False:
                self.put_to_cache(result)
        return result
