import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import dalex as dx
from dalex._explainer.theme import get_default_colors
from dalex.fairness.group_fairness.utils import fairness_check_metrics

data = dx.datasets.load_german()

X = data.drop(columns='Risk')
y = data.Risk

categorical_features = ['Sex', 'Job', 'Housing', 'Saving_accounts', "Checking_account", 'Purpose']
numeric_features = ['Credit_amount', 'Duration', 'Age']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numeric_transformer, numeric_features)])

clf_tree = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', DecisionTreeClassifier(max_depth=7, random_state=123))]).fit(X, y)

clf_svc = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', SVC())]).fit(X,y)
clf_forest = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(random_state=123))]).fit(X,y)

clf_logreg = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(random_state=123))]).fit(X,y)


exp_tree = dx.Explainer(clf_tree, X, y)
exp_svc  = dx.Explainer(clf_svc, X,y)
exp_forest  = dx.Explainer(clf_forest, X,y)
exp_logreg  = dx.Explainer(clf_logreg, X,y)

#
protected = data.Sex + '_' + np.where(data.Age < 25, 'young', 'old')
mgf_tree = exp_tree.model_fairness(protected, 'male_old')
mgf_svc = exp_svc.model_fairness(protected, 'male_old')
mgf_forest = exp_forest.model_fairness(protected, 'male_old')
mgf_logreg = exp_logreg.model_fairness(protected, 'male_old')

mgf_svc._subgroup_confusion_matrix_metrics_object.to_horizontal_DataFrame()
#
# protected = np.where(data.Age < 25, 'young', 'old')
#
# mgf_tree = exp_tree.model_group_fairness(protected, 'old')
# mgf_svc = exp_svc.model_group_fairness(protected, 'old')
# mgf_logreg = exp_logreg.model_group_fairness(protected, 'old')


mgf_svc.plot(objects = [mgf_tree, mgf_forest])
mgf_svc.plot(objects = [mgf_tree, mgf_forest], type = 'metric_scores')


