import unittest

import numpy as np
import pandas as pd
from plotly.graph_objs import Figure
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from traitlets.traitlets import Dict

import dalex as dx


class AspectTestTitanic(unittest.TestCase):
    def setUp(self):
        data = dx.datasets.load_titanic()
        data.loc[:, "survived"] = LabelEncoder().fit_transform(data.survived)

        self.X = data.drop(columns="survived")
        self.y = data.survived.values

        numeric_features = ["age", "fare", "sibsp", "parch"]
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_features = ["gender", "class", "embarked"]
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        clf = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(50, 100, 50), max_iter=400, random_state=0
                    ),
                ),
            ]
        )

        clf.fit(self.X, self.y)

        self.exp = dx.Explainer(clf, self.X, self.y, verbose=False)
        self.exp2 = dx.Explainer(clf, self.X, self.y, label="model2", verbose=False)
        self.aspect = dx.Aspect(self.exp)
        self.aspect2 = dx.Aspect(self.exp2, depend_method="pps")

    def test(self):
        self.assertIsInstance(self.aspect.explainer, dx.Explainer)
        self.assertIsInstance(self.aspect2.explainer, dx.Explainer)

        self.assertIsInstance(self.aspect.depend_matrix, pd.DataFrame)
        self.assertIsInstance(self.aspect2.depend_matrix, pd.DataFrame)

        self.assertIsInstance(self.aspect.linkage_matrix, np.ndarray)
        self.assertIsInstance(self.aspect2.linkage_matrix, np.ndarray)

        self.assertIsInstance(self.aspect._hierarchical_clustering_dendrogram, Figure)
        self.assertIsInstance(self.aspect2._hierarchical_clustering_dendrogram, Figure)

        self.assertIsInstance(self.aspect._dendrogram_aspects_ordered, pd.DataFrame)
        self.assertIsInstance(self.aspect2._dendrogram_aspects_ordered, pd.DataFrame)

        self.assertIsInstance(self.aspect.get_aspects(h=0.3), dict)
        self.assertIsInstance(self.aspect2.get_aspects(h=0.99), dict)

        self.assertIsInstance(self.aspect.plot_clustering_dendrogram(show=False), Figure)
        self.assertIsInstance(self.aspect2.plot_clustering_dendrogram(show=False), Figure)
        
    def test_predict_parts(self):
        pai = self.aspect.predict_parts(self.X.iloc[12])
        groups = {"personal_info":["gender", "age"], "ticket_info":['class', 'embarked', 'fare'], 'other':['sibsp','parch']}
        pai2 = self.aspect.predict_parts(self.X.iloc[12], variable_groups=groups)

        self.assertIsInstance(pai, dx.aspect.PredictAspectImportance)
        self.assertIsInstance(pai2, dx.aspect.PredictAspectImportance)

        self.assertIsInstance(pai.result, pd.DataFrame)
        self.assertIsInstance(pai2.result, pd.DataFrame)

        self.assertEqual(set(pai.result.columns), set(["aspect_name", "variable_names", "variables_values", "importance", "min_depend", "vars_min_depend", "label"]))
        self.assertEqual(set(pai2.result.columns), set(["aspect_name", "variable_names", "variables_values", "importance", "min_depend", "vars_min_depend", "label"]))

        self.assertGreaterEqual(1, pai.result.min_depend.max())
        self.assertGreaterEqual(1, pai2.result.min_depend.max())

        self.assertGreaterEqual(pai.result.min_depend.min(), 0)
        self.assertGreaterEqual(pai2.result.min_depend.min(), 0)

        for i in range(len(pai.result)):
            self.assertTrue(len(pai.result.loc[:, 'variables_values'][i]) == len(pai.result.loc[:, 'variable_names'][i]))
        for i in range(len(pai2.result)):
            self.assertTrue([len(pai2.result.loc[:, 'variables_values'][i]) == len(pai2.result.loc[:, 'variable_names'][i])])

        self.assertEqual(list(groups.keys()), list(pai2.result.aspect_name))
        
        fig = pai.plot(show=False)
        self.assertIsInstance(fig, Figure)
        fig2 = pai2.plot(show=False)
        self.assertIsInstance(fig2, Figure)

    def test_model_parts(self):
        mai = self.aspect.model_parts()
        groups = {"personal_info":["gender", "age"], "ticket_info":['class', 'embarked', 'fare'], 'other':['sibsp','parch']}
        mai2 = self.aspect.model_parts(variable_groups=groups)

        self.assertIsInstance(mai, dx.aspect.ModelAspectImportance)
        self.assertIsInstance(mai2, dx.aspect.ModelAspectImportance)

        self.assertIsInstance(mai.result, pd.DataFrame)
        self.assertIsInstance(mai2.result, pd.DataFrame)

        self.assertEqual(set(mai.result.columns), set(["aspect_name","variable_names","dropout_loss","dropout_loss_change","label", "min_depend", "vars_min_depend"]))
        self.assertEqual(set(mai2.result.columns), set(["aspect_name","variable_names","dropout_loss","dropout_loss_change","label", "min_depend", "vars_min_depend"]))

        self.assertGreaterEqual(1, mai.result.min_depend.max())
        self.assertGreaterEqual(1, mai2.result.min_depend.max())

        self.assertGreaterEqual(mai.result.min_depend.min(), 0)
        self.assertGreaterEqual(mai2.result.min_depend.min(), 0)

        self.assertEqual(set(groups.keys()), set(mai2.result.aspect_name[1:-1]))
        
        fig = mai.plot(show=False)
        self.assertIsInstance(fig, Figure)
        fig2 = mai2.plot(show=False)
        self.assertIsInstance(fig2, Figure)

    def test_predict_triplot(self):
        pt = self.aspect.predict_triplot(self.X.iloc[12])
        pt2 = self.aspect.predict_triplot(self.X.iloc[12], type='shap')

        self.assertIsInstance(pt, dx.aspect.PredictTriplot)
        self.assertIsInstance(pt2, dx.aspect.PredictTriplot)

        self.assertIsInstance(pt.result, pd.DataFrame)
        self.assertIsInstance(pt2.result, pd.DataFrame)

        self.assertEqual(set(pt.result.columns), set(["variable_names", "variables_values", "importance", "min_depend", "vars_min_depend", "label"]))
        self.assertEqual(set(pt2.result.columns), set(["variable_names", "variables_values", "importance", "min_depend", "vars_min_depend", "label"]))

        self.assertGreaterEqual(1, pt.result.min_depend.max())
        self.assertGreaterEqual(1, pt2.result.min_depend.max())

        self.assertGreaterEqual(pt.result.min_depend.min(), 0)
        self.assertGreaterEqual(pt2.result.min_depend.min(), 0)

        self.assertEqual(
            set(pt.result["variable_names"].iloc[-1]),
            set(self.X.columns),
        )
        self.assertEqual(
            set(pt2.result["variable_names"].iloc[-1]),
            set(self.X.columns),
        )

        for i in range(len(pt.result)):
            self.assertTrue(len(pt.result.loc[:, 'variables_values'][i]) == len(pt.result.loc[:, 'variable_names'][i]))
        for i in range(len(pt2.result)):
            self.assertTrue([len(pt2.result.loc[:, 'variables_values'][i]) == len(pt2.result.loc[:, 'variable_names'][i])])
        
        fig = pt.plot(show=False)
        self.assertIsInstance(fig, Figure)
        fig2 = pt2.plot(show=False)
        self.assertIsInstance(fig2, Figure)


    def test_model_parts(self):
        mt = self.aspect.model_triplot()
        mt2 = self.aspect.model_triplot(type="difference")

        self.assertIsInstance(mt, dx.aspect.ModelTriplot)
        self.assertIsInstance(mt2, dx.aspect.ModelTriplot)

        self.assertIsInstance(mt.result, pd.DataFrame)
        self.assertIsInstance(mt2.result, pd.DataFrame)

        self.assertEqual(set(mt.result.columns), set(["variable_names","dropout_loss","dropout_loss_change","label", "min_depend", "vars_min_depend"]))
        self.assertEqual(set(mt2.result.columns), set(["variable_names","dropout_loss","dropout_loss_change","label", "min_depend", "vars_min_depend"]))

        self.assertGreaterEqual(1, mt.result.min_depend.max())
        self.assertGreaterEqual(1, mt2.result.min_depend.max())

        self.assertGreaterEqual(mt.result.min_depend.min(), 0)
        self.assertGreaterEqual(mt2.result.min_depend.min(), 0)

        self.assertEqual(
            set(mt.result["variable_names"].iloc[-1]),
            set(self.X.columns),
        )
        self.assertEqual(
            set(mt2.result["variable_names"].iloc[-1]),
            set(self.X.columns),
        )
        
        fig = mt.plot(show=False)
        self.assertIsInstance(fig, Figure)
        fig2 = mt2.plot(show=False)
        self.assertIsInstance(fig2, Figure)

        
