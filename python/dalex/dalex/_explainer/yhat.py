from warnings import warn
import numpy as np


def yhat_default(m, d):
    return m.predict(d)


def yhat_proba_default(m, d):
    return m.predict_proba(d)[:, 1]


def yhat_xgboost(m, d):
    from xgboost import DMatrix
    return m.predict(DMatrix(d))


def get_tf_yhat(model):
    if not (str(type(model)).startswith("<class 'tensorflow.python.keras.engine") or
             str(type(model)).startswith("<class 'keras.engine") or
             str(type(model)).startswith("<class 'keras.src.models")):
        return None

    if model.output_shape[1] == 1:
        return yhat_tf_regression, "regression"
    elif model.output_shape[1] == 2:
        return yhat_tf_classification, "classification"
    else:
        warn("Tensorflow: Output shape of the predict method should not be greater than 2")
        return yhat_tf_classification, "classification"


def yhat_tf_regression(m, d):
    return m.predict(np.array(d), verbose=0).reshape(-1, )


def yhat_tf_classification(m, d):
    return m.predict(np.array(d), verbose=0)[:, 1]


def get_h2o_yhat(model):
    if not str(type(model)).startswith("<class 'h2o.estimators"):
        return None
    
    if model.type == 'classifier':
        return yhat_h2o_classification, "classification"
    if model.type == 'regressor':
        return yhat_h2o_regression, "regression"
    
    
def yhat_h2o_regression(m, d):
    from h2o import H2OFrame
    return m.predict(H2OFrame(d, column_types=m._column_types)).as_data_frame().to_numpy().flatten()


def yhat_h2o_classification(m, d):
    from h2o import H2OFrame
    return m.predict(H2OFrame(d, column_types=m._column_types)).as_data_frame().to_numpy()[:, 2]


# not used currently
def yhat_pycaret_regression(m, d):
    from pycaret.regression import predict_model
    return predict_model(m, d, verbose=False)['Score'].values
def yhat_pycaret_classification(m, d):
    from pycaret.classification import predict_model
    return predict_model(m, d, verbose=False)['Score'].values

    
def get_predict_function_and_model_type(model, model_class):
    
    prep_tf = get_tf_yhat(model)
    prep_h2o = get_h2o_yhat(model)
    
    # check for exceptions
    yhat_exception_dict = {
        "xgboost.core.Booster": (yhat_xgboost, None),  # there is no way of checking for regr/classif
        "tensorflow.python.keras.engine.sequential.Sequential": prep_tf,
        "tensorflow.python.keras.engine.training.Model": prep_tf,
        "tensorflow.python.keras.engine.functional.Functional": prep_tf,
        "keras.engine.sequential.Sequential": prep_tf,
        "keras.engine.training.Model": prep_tf,
        "keras.engine.functional.Functional": prep_tf,
        "keras.src.models.sequential.Sequential": prep_tf,
        "h2o.estimators.coxph.H2OCoxProportionalHazardsEstimator": prep_h2o,
        "h2o.estimators.deeplearning.H2ODeepLearningEstimator": prep_h2o,
        "h2o.estimators.gam.H2OGeneralizedAdditiveEstimator": prep_h2o,
        "h2o.estimators.gbm.H2OGradientBoostingEstimator": prep_h2o,
        "h2o.estimators.glm.H2OGeneralizedLinearEstimator": prep_h2o,
        "h2o.estimators.naive_bayes.H2ONaiveBayesEstimator": prep_h2o,
        "h2o.estimators.psvm.H2OSupportVectorMachineEstimator": prep_h2o,
        "h2o.estimators.random_forest.H2ORandomForestEstimator": prep_h2o,
        "h2o.estimators.rulefit.H2ORuleFitEstimator": prep_h2o,
        "h2o.estimators.stackedensemble.H2OStackedEnsembleEstimator": prep_h2o,
        "h2o.estimators.targetencoder.H2OTargetEncoderEstimator": prep_h2o,
        "h2o.estimators.xgboost.H2OXGBoostEstimator": prep_h2o
    }

    if yhat_exception_dict.get(model_class, None) is not None:
        return yhat_exception_dict.get(model_class)

    model_type_ = None
    # additional extraction sklearn
    if hasattr(model, '_estimator_type'):
        if model._estimator_type == 'classifier':
            model_type_ = 'classification'
        elif model._estimator_type == 'regressor':
            model_type_ = 'regression'

    # default extraction
    if hasattr(model, 'predict_proba'):
        # check if model has predict_proba
        return yhat_proba_default, 'classification' if model_type_ is None else model_type_
    elif hasattr(model, 'predict'):
        # check if model has predict
        return yhat_default, 'regression' if model_type_ is None else model_type_
    else:
        # this path results in an error later
        return False, None
