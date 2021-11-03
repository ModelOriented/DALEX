import copy

import numpy as np

import dalex.datasets
from . import checks
from .._basics import checks as basic_checks
from .._basics.exceptions import ParameterCheckError
from ..._explainer import helper


def reweight(protected, y, verbose = True):
    """Obtain weights for model training and mitigate bias in Statistical Parity.

    Method produces weights for each subgroup for each class.
    Firstly, it assumes that protected variable and class are
    independent and calculates expected probability of this
    certain event (that subgroup == a and class = c).
    Than it calculates the actual probability of this event
    based on empirical data. Finally the weight is quotient
    of those probabilities.

    Parameters
    -----------
    protected : np.ndarray (1d)
        Vector, preferably 1-dimensional np.ndarray containing strings,
        which denotes the membership to a subgroup.
        NOTE: List and pd.Series are also supported; however, if provided,
        they will be transformed into a np.ndarray (1d) with dtype 'U'.
    y : pd.Series or pd.DataFrame or np.ndarray (1d)
        Target variable with outputs / scores. It shall have the same length as `protected`
    verbose : bool
        Print messages about changes of types in 'y' and 'protected' (default is `True`).

    Returns
    -----------
    numpy.ndarray (1d)
        Array with sample (case) weights

    Notes
    -----------
    - https://link.springer.com/content/pdf/10.1007/s10115-011-0463-8.pdf
    """
    y = basic_checks.check_y(y, verbose)
    protected = basic_checks.check_protected(protected, verbose)

    if not len(y) == len(protected):
        raise ParameterCheckError("protected and target (y) must have the same length")

    weights = np.repeat(None, len(y))

    for subgroup in np.unique(protected):
        for c in np.unique(y):

            Xs = np.sum(protected == subgroup)
            Xc = np.sum(y == c)
            Xsc = np.sum((protected == subgroup) & (c == y))
            Wsc = (Xs * Xc) / (len(y) * Xsc)

            weights[(protected == subgroup) & (y == c)] = Wsc

    return weights

def roc_pivot(explainer, protected, privileged, cutoff = 0.5, theta = 0.05, verbose = True):
    """Reject Option based Classification pivot

    Reject Option based Classifier is post-processing bias
    mitigation method. Method changes the predictions of model
    (probabilities) and returns new explainer with modified 'y_hat'.
    Probabilities that are made for privileged subgroup and are
    favorable, and close to cutoff are pivoted to the other side
    of the cutoff. The opposite happens for unprivileged observations
    (changing unfavorable and close to cutoff observations to favorable
    by pivoting probabilities from left of the cutoff to right).
    By this potentially wrongfully labeled observations are
    assigned different labels. Note that 1 in y in Explainer
    should indicate favorable outcome.

    Parameters
    -----------
    explainer: Explainer
        Explainer made from classification model.

    protected : np.ndarray (1d)
        Vector, preferably 1-dimensional np.ndarray containing strings,
        which denotes the membership to a subgroup.
        NOTE: List and pd.Series are also supported; however, if provided,
        they will be transformed into a np.ndarray (1d) with dtype 'U'.
    privileged : str
        Subgroup that is suspected to have the most privilege.
        It needs to be a string present in `protected`.
    cutoff: float
        Threshold for probabilistic output of a classifier.
    theta: float
        Value that indicates the radius of the area where values
        are pivoted. The default is (0.05) which means that
        the probabilities of privileged class within
        (cutoff, cutoff+ theta)  will be pivoted to the other
        side of the cutoff. The opposite thing will happen for
        unprivileged subgroup.
    verbose : bool
        Print messages about changes of types in 'y' and 'protected' (default is `True`).

    Returns
    -----------
    Explainer class object
        Explainer with changed 'y_hat'

    Notes
    -----------
    - https://ieeexplore.ieee.org/document/6413831/
    """



    if not isinstance(explainer, dalex.Explainer):
        raise ParameterCheckError("explainer must be of type 'Explainer'")

    if explainer.model_type != 'classification':
        raise ParameterCheckError("model in explainer must be binary classification type")

    # same checking as in epsilon
    theta = checks.check_epsilon(theta, 'theta')
    cutoff = checks.check_epsilon(cutoff, 'cutoff')

    protected = basic_checks.check_protected(protected, verbose)
    privileged = basic_checks.check_privileged(privileged, protected, verbose)

    exp = copy.deepcopy(explainer)
    probs = exp.y_hat

    if not len(probs) == len(protected):
        raise ParameterCheckError("protected and target (y) must have the same length")

    is_close = np.abs(probs - cutoff) < theta
    is_privileged = privileged == protected
    is_favorable = probs > cutoff

    probs[is_close & is_privileged & is_favorable] = cutoff - (probs[is_close & is_privileged & is_favorable] - cutoff)
    probs[is_close & np.logical_not(is_privileged) & np.logical_not(is_favorable)] = cutoff + (cutoff - probs[is_close & np.logical_not(is_privileged) & np.logical_not(is_favorable)])

    probs[probs < 0] = 0
    probs[probs > 1] = 1

    return exp

def resample(protected, y, type = 'uniform', probs = None,  verbose = True):
    """Returns indices of observations for data.

    Method of bias mitigation. Similarly to 'reweight'
    this method computes desired number of observations just
    if the protected variable was independent from y and
    on this basis decides if this subgroup with certain class
    (favorable or not) should be more or less numerous. Than performs
    oversampling or undersampling depending on the case.
    If type of sampling is set to 'preferential' and probs
    are provided than instead of uniform sampling preferential
    sampling will be performed. Preferential sampling depending on the case
    will sample observations close to border or far from border.

    Parameters
    -----------
    protected : np.ndarray (1d)
        Vector, preferably 1-dimensional np.ndarray containing strings,
        which denotes the membership to a subgroup.
        NOTE: List and pd.Series are also supported; however, if provided,
        they will be transformed into a np.ndarray (1d) with dtype 'U'.
    y : pd.Series or pd.DataFrame or np.ndarray (1d)
        Target variable with outputs / scores. It shall have the same length as `protected`
    type : {'uniform', 'preferential'}
        Type indicates what strategy to use when choosing the samples.
        (default is 'uniform')
    probs : np.ndarray (1d)
        Vector with probabilities for each sample. Note that this should be
        probabilities for favourable outcome. For the best performance they
        should be consistent with 'y' but it is not required. This argument
        is required when using strategy of type 'preferential'
    verbose : bool
        Print messages about changes of types in 'y' and 'protected' (default is `True`).

    Returns
    -----------
    numpy.ndarray (1d)
        Array with indices for the data.

    Notes
    -----------
        - https://link.springer.com/content/pdf/10.1007/s10115-011-0463-8.pdf
    """
    if type == 'preferential' and probs is None:
        raise ParameterCheckError("when using type 'preferential' probabilities (probs) must be provided")

    if type not in set(['uniform', 'preferential']):
        raise ParameterCheckError("type must be either 'uniform' or 'preferential'")


    protected = basic_checks.check_protected(protected, verbose)
    y = basic_checks.check_y(y, verbose)

    if type == 'preferential':
        try:
            probs = np.asarray(probs)
            helper.verbose_cat("converted 'probs' to numpy array", verbose=verbose)
        except Exception:
            raise ParameterCheckError("try converting 'probs' to 1D numpy array")

        if probs.ndim != 1 or len(probs) != len(y):
            raise ParameterCheckError("probs parameter must 1D numpy array with the same length as y")


    weights = reweight(protected, y, verbose=False)

    expected_size =  dict.fromkeys(np.unique(protected))
    for key in expected_size.keys():
        expected_size[key] = dict.fromkeys(np.unique(y))

    for subgroup in expected_size.keys():
        for value in np.unique(y):
            case_weights = weights[(subgroup == protected) & (value == y)]
            case_size = len(case_weights)
            weight = case_weights[0]
            expected_size[subgroup][value] = round(case_size * weight)

    indices = []

    for subgroup in expected_size.keys():
        for value in np.unique(y):
            current_case = np.arange(len(y))[(protected == subgroup) & (y == value)]
            expected = expected_size[subgroup][value]
            actual = np.sum((protected == subgroup) & (y == value))
            if expected == actual:
                 indices += list(current_case)

            elif expected < actual:
                if type == 'uniform':
                    indices += list(np.random.choice(current_case, expected, replace=False))
                else:
                    sorted_current_case = current_case[np.argsort(probs[current_case])]
                    if value == 0:
                        indices += list(sorted_current_case[:expected])
                    if value == 1:
                        indices += list(sorted_current_case[-expected:])
            else:
                if type == 'uniform':
                    u_ind = list(np.repeat(current_case, expected // actual))
                    u_ind += list(np.random.choice(current_case, expected % actual))

                    indices += u_ind

                else:
                    sorted_current_case = current_case[np.argsort(probs[current_case])]
                    p_ind = list(np.repeat(current_case, expected // actual))

                    if expected % actual != 0:
                        if value == 0:
                            p_ind += list(sorted_current_case[-(expected % actual):])
                        if value == 1:
                            p_ind += list(sorted_current_case[:(expected % actual)])

                    indices += p_ind

    return np.array(indices)
