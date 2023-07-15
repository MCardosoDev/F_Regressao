# Regressão
### Modelos de regressão avaliados com base de dados de veiculo usados a venda


- import pandas as pd
- import matplotlib.pyplot as plt
- import seaborn as sns
- import numpy as np
- import warnings
  - warnings.filterwarnings('ignore')
- import pickle
- from ipywidgets import widgets, HBox, VBox
- from IPython.display import display
- from sklearn.preprocessing import LabelEncoder, StandardScaler
- from sklearn.model_selection import train_test_split
- from sklearn.model_selection import GridSearchCV
- from sklearn.linear_model import (
  - LinearRegression,
  - Ridge,
  - Lasso,
  - ElasticNet,
  - BayesianRidge,
  - SGDRegressor,
  - LogisticRegression)
- from sklearn.tree import DecisionTreeRegressor
- from sklearn.neighbors import KNeighborsRegressor
- from sklearn.svm import SVR
- from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
- from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
- from Utils import shapiro_w, mannwhitney_u, research_py, chi_2