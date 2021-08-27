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
from lightgbm import LGBMRegressor


import dalex as dx
from dalex.model_explanations._variable_importance.loss_functions import *
from dalex.aspect._predict_aspect_importance.object import PredictAspectImportance
from dalex.aspect._model_triplot.object import ModelTriplot
from dalex.aspect._predict_triplot.object import PredictTriplot
from dalex.aspect._model_aspect_importance.object import ModelAspectImportance

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
        pd.testing.assert_frame_equal(self.aspect2.depend_matrix, self.aspect2.depend_matrix.T)
        for ind, row in self.aspect.depend_matrix.iterrows():
            self.assertEqual(row[ind], 1.0)
        for ind, row in self.aspect2.depend_matrix.iterrows():
            self.assertEqual(row[ind], 1.0)


        self.assertIsInstance(self.aspect.linkage_matrix, np.ndarray)
        self.assertIsInstance(self.aspect2.linkage_matrix, np.ndarray)

        self.assertIsInstance(self.aspect._hierarchical_clustering_dendrogram, Figure)
        self.assertIsInstance(self.aspect2._hierarchical_clustering_dendrogram, Figure)

        self.assertIsInstance(self.aspect._dendrogram_aspects_ordered, pd.DataFrame)
        self.assertIsInstance(self.aspect2._dendrogram_aspects_ordered, pd.DataFrame)
        
        self.assertIsInstance(self.aspect.get_aspects(h=0.3), dict)
        self.assertIsInstance(self.aspect2.get_aspects(h=0.99), dict)
        self.assertGreaterEqual(3, len(self.aspect.get_aspects(h=0.3, n=3)))
        self.assertEqual(len(self.aspect.depend_matrix), len(self.aspect2.get_aspects(h=3)))

        self.assertIsInstance(self.aspect.plot_clustering_dendrogram(show=False), Figure)
        self.assertIsInstance(self.aspect2.plot_clustering_dendrogram(show=False), Figure)
        
    def test_predict_parts(self):
        pai = self.aspect.predict_parts(self.X.iloc[12])
        groups = {"personal_info":["gender", "age"], "ticket_info":['class', 'embarked', 'fare'], 'other':['sibsp','parch']}
        pai2 = self.aspect2.predict_parts(self.X.iloc[12], variable_groups=groups)
        pai3 = self.aspect.predict_parts(self.X.iloc[19], type='shap')
        n_aspects = 4
        pai4 = self.aspect2.predict_parts(self.X.iloc[22], sample_method='binom', n_aspects=n_aspects)

        self.assertEqual(set(groups.keys()), set(pai2.result.aspect_name))
        for pai_x in [pai, pai2, pai3, pai4]:
            self.assertIsInstance(pai_x, dx.aspect.PredictAspectImportance)
            self.assertIsInstance(pai_x.result, pd.DataFrame)
            self.assertEqual(set(pai_x.result.columns), set(["aspect_name", "variable_names", "variable_values", "importance", "min_depend", "vars_min_depend", "label"]))
            self.assertGreaterEqual(1, pai_x.result.min_depend.max())
            self.assertGreaterEqual(pai_x.result.min_depend.min(), 0)
            fig = pai_x.plot(show=False)
            self.assertIsInstance(fig, Figure)
            for index, row in pai_x.result.iterrows():
                self.assertTrue(len(row['variable_values']) == len(row['variable_names']))
                self.assertTrue(set(row['vars_min_depend']).issubset(set(row['variable_names'])))

        
    
    def test_predict_aspect_importance(self):
        pai = PredictAspectImportance(self.aspect.get_aspects(h=0.2))
        groups = {"personal_info":["gender", "age"], "ticket_info":['class', 'embarked', 'fare'], 'other':['sibsp','parch']}
        pai2 = PredictAspectImportance(variable_groups=groups)
        pai3 = PredictAspectImportance(self.aspect.get_aspects(h=0.5), type='shap')
        n_aspects = 4
        pai4 = PredictAspectImportance(self.aspect2.get_aspects(h=0.1), sample_method='binom', n_aspects=n_aspects)

        pai.fit(self.exp, self.X.iloc[12])
        pai2.fit(self.exp2, self.X.iloc[13])
        pai3.fit(self.exp, self.X.iloc[14])
        pai4.fit(self.exp2, self.X.iloc[15])

        self.assertIsInstance(pai, dx.aspect.PredictAspectImportance)
        self.assertIsInstance(pai2, dx.aspect.PredictAspectImportance)
        self.assertIsInstance(pai3, dx.aspect.PredictAspectImportance)
        self.assertIsInstance(pai4, dx.aspect.PredictAspectImportance)

        self.assertIsInstance(pai.result, pd.DataFrame)
        self.assertIsInstance(pai2.result, pd.DataFrame)
        self.assertIsInstance(pai3.result, pd.DataFrame)
        self.assertIsInstance(pai4.result, pd.DataFrame)
        
        self.assertEqual(set(pai.result.columns), set(["aspect_name", "variable_names", "variable_values", "importance", "min_depend", "vars_min_depend", "label"]))
        self.assertEqual(set(pai2.result.columns), set(["aspect_name", "variable_names", "variable_values", "importance", "min_depend", "vars_min_depend", "label"]))
        self.assertEqual(set(pai3.result.columns), set(["aspect_name", "variable_names", "variable_values", "importance", "min_depend", "vars_min_depend", "label"]))
        self.assertEqual(set(pai4.result.columns), set(["aspect_name", "variable_names", "variable_values", "importance", "min_depend", "vars_min_depend", "label"]))

        self.assertGreaterEqual(1, pai.result.min_depend.max())
        self.assertGreaterEqual(1, pai2.result.min_depend.max())
        self.assertGreaterEqual(1, pai3.result.min_depend.max())
        self.assertGreaterEqual(1, pai4.result.min_depend.max())

        self.assertGreaterEqual(pai.result.min_depend.min(), 0)
        self.assertGreaterEqual(pai2.result.min_depend.min(), 0)
        self.assertGreaterEqual(pai3.result.min_depend.min(), 0)
        self.assertGreaterEqual(pai4.result.min_depend.min(), 0)

        for i in range(len(pai.result)):
            self.assertTrue(len(pai.result.loc[i, 'variable_values']) == len(pai.result.loc[i, 'variable_names']))
            self.assertTrue(set(pai.result.loc[i, 'vars_min_depend']).issubset(set(pai.result.loc[i, 'variable_names'])))
        for i in range(len(pai2.result)):
            self.assertTrue([len(pai2.result.loc[i, 'variable_values']) == len(pai2.result.loc[i, 'variable_names'])])
            self.assertTrue(set(pai2.result.loc[i, 'vars_min_depend']).issubset(set(pai2.result.loc[i, 'variable_names'])))
        for i in range(len(pai3.result)):
            self.assertTrue(len(pai3.result.loc[i, 'variable_values']) == len(pai3.result.loc[i, 'variable_names']))
            self.assertTrue(set(pai3.result.loc[i, 'vars_min_depend']).issubset(set(pai3.result.loc[i, 'variable_names'])))
        counter = 0
        for i in range(len(pai4.result)):
            self.assertTrue([len(pai4.result.loc[i, 'variable_values']) == len(pai4.result.loc[i, 'variable_names'])])
            self.assertTrue(set(pai4.result.loc[i, 'vars_min_depend']).issubset(set(pai4.result.loc[i, 'variable_names'])))
            if pai4.result.loc[i, 'importance'] != 0:
                counter +=1
        self.assertEqual(counter, n_aspects)

        self.assertEqual(set(groups.keys()), set(pai2.result.aspect_name))
        
        fig = pai.plot(show=False)
        self.assertIsInstance(fig, Figure)
        fig2 = pai2.plot(show=False)
        self.assertIsInstance(fig2, Figure)
        fig3 = pai3.plot(show=False)
        self.assertIsInstance(fig3, Figure)
        fig4 = pai4.plot(show=False)
        self.assertIsInstance(fig4, Figure)
        

    def test_model_parts(self):
        mai = self.aspect.model_parts()
        groups = {"personal_info":["gender", "age"], "ticket_info":['class', 'embarked', 'fare'], 'other':['sibsp','parch']}
        mai2 = self.aspect2.model_parts(variable_groups=groups)
        mai3 = self.aspect.model_parts(type='ratio')
        mai4 = self.aspect2.model_parts(type='difference')
        mai5 = self.aspect.model_parts(loss_function='rmse')

        self.assertEqual(mai.loss_function, loss_one_minus_auc)
        self.assertEqual(mai2.loss_function, loss_one_minus_auc)
        self.assertEqual(mai3.loss_function, loss_one_minus_auc)
        self.assertEqual(mai4.loss_function, loss_one_minus_auc)
        self.assertEqual(mai5.loss_function, loss_root_mean_square)

        for mai_x in [mai, mai2, mai3, mai4, mai5]:
            self.assertIsInstance(mai_x, dx.aspect.ModelAspectImportance)
            self.assertIsInstance(mai_x.result, pd.DataFrame)
            self.assertEqual(set(mai_x.result.columns), set(["aspect_name","variable_names","dropout_loss","dropout_loss_change","label", "min_depend", "vars_min_depend"]))
            self.assertGreaterEqual(1, mai_x.result.min_depend.max())
            self.assertGreaterEqual(mai_x.result.min_depend.min(), 0)
            for index, row in mai_x.result.iterrows():
                if row['aspect_name'] not in ['_baseline_', '_full_model_']:
                    self.assertTrue(set(row['vars_min_depend']).issubset(set(row['variable_names'])))
            fig = mai_x.plot(show=False)
            self.assertIsInstance(fig, Figure)

    def test_model_aspect_importance(self):
        mai = ModelAspectImportance(self.aspect.get_aspects(h=0.2))
        groups = {"personal_info":["gender", "age"], "ticket_info":['class', 'embarked', 'fare'], 'other':['sibsp','parch']}
        mai2 = ModelAspectImportance(variable_groups=groups)
        mai3 = ModelAspectImportance(self.aspect2.get_aspects(h=0.4), type='ratio')
        mai4 = ModelAspectImportance(self.aspect.get_aspects(h=0.3), type='difference')
        mai5 = ModelAspectImportance(self.aspect2.get_aspects(h=0.6), loss_function='rmse')

        mai.fit(self.exp)
        mai2.fit(self.exp)
        mai3.fit(self.exp2)
        mai4.fit(self.exp)
        mai5.fit(self.exp2)

        self.assertEqual(mai.loss_function, loss_one_minus_auc)
        self.assertEqual(mai2.loss_function, loss_one_minus_auc)
        self.assertEqual(mai3.loss_function, loss_one_minus_auc)
        self.assertEqual(mai4.loss_function, loss_one_minus_auc)
        self.assertEqual(mai5.loss_function, loss_root_mean_square)

        for mai_x in [mai, mai2, mai3, mai4, mai5]:
            self.assertIsInstance(mai_x, dx.aspect.ModelAspectImportance)
            self.assertIsInstance(mai_x.result, pd.DataFrame)
            self.assertEqual(set(mai_x.result.columns), set(["aspect_name","variable_names","dropout_loss","dropout_loss_change","label", "min_depend", "vars_min_depend"]))
            self.assertGreaterEqual(1, mai_x.result.min_depend.max())
            self.assertGreaterEqual(mai_x.result.min_depend.min(), 0)
            for index, row in mai_x.result.iterrows():
                if row['aspect_name'] not in ['_baseline_', '_full_model_']:
                    self.assertTrue(set(row['vars_min_depend']).issubset(set(row['variable_names'])))
            fig = mai_x.plot(show=False)
            self.assertIsInstance(fig, Figure)

    def test_predict_triplot(self):
        pt = self.aspect.predict_triplot(self.X.iloc[12], sample_method='binom')
        pt2 = self.aspect2.predict_triplot(self.X.iloc[12], type='shap')

        self.assertIsInstance(pt, dx.aspect.PredictTriplot)
        self.assertIsInstance(pt2, dx.aspect.PredictTriplot)

        self.assertIsInstance(pt.result, pd.DataFrame)
        self.assertIsInstance(pt2.result, pd.DataFrame)

        self.assertEqual(set(pt.result.columns), set(["variable_names", "variable_values", "importance", "min_depend", "vars_min_depend", "label"]))
        self.assertEqual(set(pt2.result.columns), set(["variable_names", "variable_values", "importance", "min_depend", "vars_min_depend", "label"]))

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
            self.assertTrue(len(pt.result.loc[i, 'variable_values']) == len(pt.result.loc[i, 'variable_names']))
            self.assertTrue(set(pt.result.loc[i, 'vars_min_depend']).issubset(set(pt.result.loc[i, 'variable_names'])))
        for i in range(len(pt2.result)):
            self.assertTrue([len(pt2.result.loc[i, 'variable_values']) == len(pt2.result.loc[i, 'variable_names'])])
            self.assertTrue(set(pt.result.loc[i, 'vars_min_depend']).issubset(set(pt.result.loc[i, 'variable_names'])))
        
        fig = pt.plot(show=False)
        self.assertIsInstance(fig, Figure)
        fig2 = pt2.plot(show=False)
        self.assertIsInstance(fig2, Figure)
    
    def test_predict_triplot_class(self):
        pt = PredictTriplot(sample_method='binom')
        pt2 = PredictTriplot(type='shap')
        pt.fit(self.aspect, self.X.iloc[12])
        pt2.fit(self.aspect2, self.X.iloc[12])

        self.assertIsInstance(pt, dx.aspect.PredictTriplot)
        self.assertIsInstance(pt2, dx.aspect.PredictTriplot)

        self.assertIsInstance(pt.result, pd.DataFrame)
        self.assertIsInstance(pt2.result, pd.DataFrame)

        self.assertEqual(set(pt.result.columns), set(["variable_names", "variable_values", "importance", "min_depend", "vars_min_depend", "label"]))
        self.assertEqual(set(pt2.result.columns), set(["variable_names", "variable_values", "importance", "min_depend", "vars_min_depend", "label"]))

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
            self.assertTrue(len(pt.result.loc[i, 'variable_values']) == len(pt.result.loc[i, 'variable_names']))
            self.assertTrue(set(pt.result.loc[i, 'vars_min_depend']).issubset(set(pt.result.loc[i, 'variable_names'])))
        for i in range(len(pt2.result)):
            self.assertTrue([len(pt2.result.loc[i, 'variable_values']) == len(pt2.result.loc[i, 'variable_names'])])
            self.assertTrue(set(pt.result.loc[i, 'vars_min_depend']).issubset(set(pt.result.loc[i, 'variable_names'])))
        
        fig = pt.plot(show=False)
        self.assertIsInstance(fig, Figure)
        fig2 = pt2.plot(show=False)
        self.assertIsInstance(fig2, Figure)


    def test_model_triplot(self):
        mt = self.aspect.model_triplot()
        mt2 = self.aspect.model_triplot(type="difference")

        self.assertEqual(mt.loss_function, loss_one_minus_auc)
        self.assertEqual(mt2.loss_function, loss_one_minus_auc)

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
    
    def test_model_triplot_class(self):
        mt = ModelTriplot()
        mt2 = ModelTriplot(type="difference")

        mt.fit(self.aspect)
        mt2.fit(self.aspect2)

        self.assertEqual(mt.loss_function, loss_one_minus_auc)
        self.assertEqual(mt2.loss_function, loss_one_minus_auc)

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


class AspectTestFifa(unittest.TestCase):
    def setUp(self):
        data = dx.datasets.load_fifa()

        self.X = data.drop(["overall", "potential", "value_eur", "wage_eur"], axis = 1)
        self.y = data['value_eur']

        categorical_features = ['nationality']
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical_features)], remainder='passthrough')
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LGBMRegressor(random_state=123))
        ])

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
        pd.testing.assert_frame_equal(self.aspect2.depend_matrix, self.aspect2.depend_matrix.T)
        for ind, row in self.aspect.depend_matrix.iterrows():
            self.assertEqual(row[ind], 1.0)
        for ind, row in self.aspect2.depend_matrix.iterrows():
            self.assertEqual(row[ind], 1.0)


        self.assertIsInstance(self.aspect.linkage_matrix, np.ndarray)
        self.assertIsInstance(self.aspect2.linkage_matrix, np.ndarray)

        self.assertIsInstance(self.aspect._hierarchical_clustering_dendrogram, Figure)
        self.assertIsInstance(self.aspect2._hierarchical_clustering_dendrogram, Figure)

        self.assertIsInstance(self.aspect._dendrogram_aspects_ordered, pd.DataFrame)
        self.assertIsInstance(self.aspect2._dendrogram_aspects_ordered, pd.DataFrame)
        
        self.assertIsInstance(self.aspect.get_aspects(h=0.3), dict)
        self.assertIsInstance(self.aspect2.get_aspects(h=0.99), dict)
        self.assertGreaterEqual(3, len(self.aspect.get_aspects(h=0.3, n=3)))
        self.assertEqual(len(self.aspect.depend_matrix), len(self.aspect2.get_aspects(h=3)))

        self.assertIsInstance(self.aspect.plot_clustering_dendrogram(show=False), Figure)
        self.assertIsInstance(self.aspect2.plot_clustering_dendrogram(show=False), Figure)
        
    def test_predict_parts(self):
        pai = self.aspect.predict_parts(self.X.iloc[12])
        groups = self.aspect.get_aspects(h=0.3)
        pai2 = self.aspect2.predict_parts(self.X.iloc[12], variable_groups=groups)
        pai3 = self.aspect.predict_parts(self.X.iloc[19], type='shap')
        n_aspects = 4
        pai4 = self.aspect2.predict_parts(self.X.iloc[22], sample_method='binom', n_aspects=n_aspects)
        self.assertEqual(set(groups.keys()), set(pai2.result.aspect_name))

        for pai_x in [pai, pai2, pai3, pai4]:
            self.assertIsInstance(pai_x, dx.aspect.PredictAspectImportance)
            self.assertIsInstance(pai_x.result, pd.DataFrame)
            self.assertEqual(set(pai_x.result.columns), set(["aspect_name", "variable_names", "variable_values", "importance", "min_depend", "vars_min_depend", "label"]))
            self.assertGreaterEqual(1, pai_x.result.min_depend.max())
            self.assertGreaterEqual(pai_x.result.min_depend.min(), 0)
            fig = pai_x.plot(show=False)
            self.assertIsInstance(fig, Figure)
            for index, row in pai_x.result.iterrows():
                self.assertTrue(len(row['variable_values']) == len(row['variable_names']))
                self.assertTrue(set(row['vars_min_depend']).issubset(set(row['variable_names'])))
    
    def test_predict_aspect_importance(self):
        pai = PredictAspectImportance(self.aspect.get_aspects(h=0.2))
        groups = self.aspect.get_aspects(h=0.3)
        pai2 = PredictAspectImportance(variable_groups=groups)
        pai3 = PredictAspectImportance(self.aspect.get_aspects(h=0.5), type='shap')
        n_aspects = 4
        pai4 = PredictAspectImportance(self.aspect2.get_aspects(h=0.1), sample_method='binom', n_aspects=n_aspects)

        pai.fit(self.exp, self.X.iloc[12])
        pai2.fit(self.exp2, self.X.iloc[13])
        pai3.fit(self.exp, self.X.iloc[14])
        pai4.fit(self.exp2, self.X.iloc[15])

        self.assertEqual(set(groups.keys()), set(pai2.result.aspect_name))

        for pai_x in [pai, pai2, pai3, pai4]:
            self.assertIsInstance(pai_x, dx.aspect.PredictAspectImportance)
            self.assertIsInstance(pai_x.result, pd.DataFrame)
            self.assertEqual(set(pai_x.result.columns), set(["aspect_name", "variable_names", "variable_values", "importance", "min_depend", "vars_min_depend", "label"]))
            self.assertGreaterEqual(1, pai_x.result.min_depend.max())
            self.assertGreaterEqual(pai_x.result.min_depend.min(), 0)
            fig = pai_x.plot(show=False)
            self.assertIsInstance(fig, Figure)
            for index, row in pai_x.result.iterrows():
                self.assertTrue(len(row['variable_values']) == len(row['variable_names']))
                self.assertTrue(set(row['vars_min_depend']).issubset(set(row['variable_names'])))

        

    def test_model_parts(self):
        mai = self.aspect.model_parts()
        groups = self.aspect.get_aspects(h=0.3)
        mai2 = self.aspect2.model_parts(variable_groups=groups)
        mai3 = self.aspect.model_parts(type='ratio')
        mai4 = self.aspect2.model_parts(type='difference')
        mai5 = self.aspect.model_parts(loss_function='1-auc')

        self.assertEqual(mai.loss_function, loss_root_mean_square)
        self.assertEqual(mai2.loss_function, loss_root_mean_square)
        self.assertEqual(mai3.loss_function, loss_root_mean_square)
        self.assertEqual(mai4.loss_function, loss_root_mean_square)
        self.assertEqual(mai5.loss_function, loss_one_minus_auc)

        for mai_x in [mai, mai2, mai3, mai4, mai5]:
            self.assertIsInstance(mai_x, dx.aspect.ModelAspectImportance)
            self.assertIsInstance(mai_x.result, pd.DataFrame)
            self.assertEqual(set(mai_x.result.columns), set(["aspect_name","variable_names","dropout_loss","dropout_loss_change","label", "min_depend", "vars_min_depend"]))
            self.assertGreaterEqual(1, mai_x.result.min_depend.max())
            self.assertGreaterEqual(mai_x.result.min_depend.min(), 0)
            for index, row in mai_x.result.iterrows():
                if row['aspect_name'] not in ['_baseline_', '_full_model_']:
                    self.assertTrue(set(row['vars_min_depend']).issubset(set(row['variable_names'])))
            fig = mai_x.plot(show=False)
            self.assertIsInstance(fig, Figure)

        

    def test_model_aspect_importance(self):
        mai = ModelAspectImportance(self.aspect.get_aspects(h=0.2))
        groups = self.aspect.get_aspects(h=0.3)
        mai2 = ModelAspectImportance(variable_groups=groups)
        mai3 = ModelAspectImportance(self.aspect2.get_aspects(h=0.4), type='ratio')
        mai4 = ModelAspectImportance(self.aspect.get_aspects(h=0.3), type='difference')
        mai5 = ModelAspectImportance(self.aspect2.get_aspects(h=0.6), loss_function='1-auc')

        mai.fit(self.exp)
        mai2.fit(self.exp)
        mai3.fit(self.exp2)
        mai4.fit(self.exp)
        mai5.fit(self.exp2)

        self.assertEqual(mai.loss_function, loss_root_mean_square)
        self.assertEqual(mai2.loss_function, loss_root_mean_square)
        self.assertEqual(mai3.loss_function, loss_root_mean_square)
        self.assertEqual(mai4.loss_function, loss_root_mean_square)
        self.assertEqual(mai5.loss_function, loss_one_minus_auc)

        self.assertEqual(set(groups.keys()), set(mai2.result.aspect_name[1:-1]))

        for mai_x in [mai, mai2, mai3, mai4, mai5]:
            self.assertIsInstance(mai_x, dx.aspect.ModelAspectImportance)
            self.assertIsInstance(mai_x.result, pd.DataFrame)
            self.assertEqual(set(mai_x.result.columns), set(["aspect_name","variable_names","dropout_loss","dropout_loss_change","label", "min_depend", "vars_min_depend"]))
            self.assertGreaterEqual(1, mai_x.result.min_depend.max())
            self.assertGreaterEqual(mai_x.result.min_depend.min(), 0)
            for index, row in mai_x.result.iterrows():
                if row['aspect_name'] not in ['_baseline_', '_full_model_']:
                    self.assertTrue(set(row['vars_min_depend']).issubset(set(row['variable_names'])))
            fig = mai_x.plot(show=False)
            self.assertIsInstance(fig, Figure)

    def test_predict_triplot(self):
        pt = self.aspect.predict_triplot(self.X.iloc[12], sample_method='binom')
        pt2 = self.aspect2.predict_triplot(self.X.iloc[12], type='shap')

        self.assertIsInstance(pt, dx.aspect.PredictTriplot)
        self.assertIsInstance(pt2, dx.aspect.PredictTriplot)

        self.assertIsInstance(pt.result, pd.DataFrame)
        self.assertIsInstance(pt2.result, pd.DataFrame)

        self.assertEqual(set(pt.result.columns), set(["variable_names", "variable_values", "importance", "min_depend", "vars_min_depend", "label"]))
        self.assertEqual(set(pt2.result.columns), set(["variable_names", "variable_values", "importance", "min_depend", "vars_min_depend", "label"]))

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
            self.assertTrue(len(pt.result.loc[i, 'variable_values']) == len(pt.result.loc[i, 'variable_names']))
            self.assertTrue(set(pt.result.loc[i, 'vars_min_depend']).issubset(set(pt.result.loc[i, 'variable_names'])))
        for i in range(len(pt2.result)):
            self.assertTrue([len(pt2.result.loc[i, 'variable_values']) == len(pt2.result.loc[i, 'variable_names'])])
            self.assertTrue(set(pt.result.loc[i, 'vars_min_depend']).issubset(set(pt.result.loc[i, 'variable_names'])))
        
        fig = pt.plot(show=False)
        self.assertIsInstance(fig, Figure)
        fig2 = pt2.plot(show=False)
        self.assertIsInstance(fig2, Figure)
    
    def test_predict_triplot_class(self):
        pt = PredictTriplot(sample_method='binom')
        pt2 = PredictTriplot(type='shap')
        pt.fit(self.aspect, self.X.iloc[12])
        pt2.fit(self.aspect2, self.X.iloc[12])

        self.assertIsInstance(pt, dx.aspect.PredictTriplot)
        self.assertIsInstance(pt2, dx.aspect.PredictTriplot)

        self.assertIsInstance(pt.result, pd.DataFrame)
        self.assertIsInstance(pt2.result, pd.DataFrame)

        self.assertEqual(set(pt.result.columns), set(["variable_names", "variable_values", "importance", "min_depend", "vars_min_depend", "label"]))
        self.assertEqual(set(pt2.result.columns), set(["variable_names", "variable_values", "importance", "min_depend", "vars_min_depend", "label"]))

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
            self.assertTrue(len(pt.result.loc[i, 'variable_values']) == len(pt.result.loc[i, 'variable_names']))
            self.assertTrue(set(pt.result.loc[i, 'vars_min_depend']).issubset(set(pt.result.loc[i, 'variable_names'])))
        for i in range(len(pt2.result)):
            self.assertTrue([len(pt2.result.loc[i, 'variable_values']) == len(pt2.result.loc[i, 'variable_names'])])
            self.assertTrue(set(pt.result.loc[i, 'vars_min_depend']).issubset(set(pt.result.loc[i, 'variable_names'])))
        
        fig = pt.plot(show=False)
        self.assertIsInstance(fig, Figure)
        fig2 = pt2.plot(show=False)
        self.assertIsInstance(fig2, Figure)


    def test_model_triplot(self):
        mt = self.aspect.model_triplot()
        mt2 = self.aspect.model_triplot(type="difference")

        self.assertEqual(mt.loss_function, loss_root_mean_square)
        self.assertEqual(mt2.loss_function, loss_root_mean_square)

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
    
    def test_model_triplot_class(self):
        mt = ModelTriplot()
        mt2 = ModelTriplot(type="difference")

        mt.fit(self.aspect)
        mt2.fit(self.aspect2)

        self.assertEqual(mt.loss_function, loss_root_mean_square)
        self.assertEqual(mt2.loss_function, loss_root_mean_square)

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
        
