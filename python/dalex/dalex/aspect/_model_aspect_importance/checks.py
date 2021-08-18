def check_method_depend(depend_method, corr_method, agg_method):
    depend_method_types = ('assoc', 'pps')
    depend_method_aliases = {'association': 'assoc', "PPS": 'pps', 'stats': 'assoc'}
    corr_method_types = ('spearman', 'pearson', 'kendall')
    agg_method_types = ('max', 'min', 'mean')
    agg_method_aliases = {'maximum': 'max', 'minimum': 'min', 'avg': 'mean', 'average': 'mean'}
    if isinstance(depend_method, str):
        if depend_method not in depend_method_types:
            if depend_method not in depend_method_aliases:
                raise ValueError("'depend_method' must be one of: {}".format(', '.join(depend_method_types+tuple(depend_method_aliases))))
            else:
                depend_method = depend_method_aliases[depend_method]
        if depend_method == "assoc":
            if corr_method not in corr_method_types:
                raise ValueError("'corr_method' must be one of: {}".format(', '.join(corr_method_types)))
        if depend_method == "pps":
            if agg_method not in agg_method_types:
                if agg_method not in agg_method_aliases:
                    raise ValueError("'agg_method' must be one of: {}".format(', '.join(agg_method_types)))
                else: 
                    agg_method = agg_method_aliases[agg_method]
    return depend_method, corr_method, agg_method

