#%%
from sklearn.linear_model import (
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    SGDRegressor,
    LogisticRegression
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import researchpy
from scipy.stats import shapiro, mannwhitneyu, chi2, normaltest

def shapiro_w(df):
    stat, p = shapiro(df)
    print('Estatística de Teste: {:.4f}, valor p: {}'.format(stat, p))
    if p > 0.05:
        return print('Não há evidência suficiente para rejeitar a hipótese de normalidade.')
    else:
        return print('A hipótese de normalidade é rejeitada.')
#%%
def mannwhitney_u(df1, df2, alternative):
    stat, p = mannwhitneyu(df1, df2, alternative=alternative)
    print("Estatística de teste U: ", stat)
    print("Valor p: ", p)
    alpha = 0.05

    if p < alpha:
        return print("Diferença estatisticamente significante")
    else:
        return print("Não há diferença estatisticamente significante")
#%%
def chi_2(df):
    prob = 0.95
    critical = chi2.ppf(prob, len(df) - 1)
    alpha = 1.0 - prob
    stat, p = normaltest(df)
    print("Critical: ", critical)
    print("Estatística de teste U: ", stat)
    print("Valor p: ", p)

    if p < alpha:
        return print("Diferença estatisticamente significante")
    else:
        return print("Não há diferença estatisticamente significante")
#%%
def research_py(df1, df2, test):
    result = researchpy.crosstab(df1, df2, test=test)
    return print(result)
#%%
def ridge_cv(x_train, y_train):
    ridge_model = Ridge(random_state=42)

    ridge_parameters = {
        'alpha': [0.01, 0.1, 0.5, 1.0, 10.0, 90.0],
        'fit_intercept': [True, False],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga']
    }

    ridge_scoring = {
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'MSE': make_scorer(mean_squared_error, greater_is_better=False),
        'R-squared': make_scorer(r2_score)
    }

    ridge_grid_search = GridSearchCV(
                            ridge_model,
                            ridge_parameters,
                            scoring=ridge_scoring,
                            cv=9,
                            refit='R-squared',
                            n_jobs=-1
                        )
    
    return ridge_grid_search.fit(x_train, y_train)
#%%
def lasso_cv(x_train, y_train):
    lasso_model = Lasso(random_state=42)

    lasso_parameters = {
        'alpha': [0.01, 0.1, 0.5, 1.0, 10.0, 90.0, 100.0, 130.0, 140.0],
        'fit_intercept': [True, False],
        'selection': ['cyclic', 'random']
    }

    lasso_scoring = {
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'MSE': make_scorer(mean_squared_error, greater_is_better=False),
        'R-squared': make_scorer(r2_score)
    }

    lasso_grid_search = GridSearchCV(
                            lasso_model,
                            lasso_parameters,
                            scoring=lasso_scoring,
                            cv=20,
                            refit='R-squared'
                        )

    return lasso_grid_search.fit(x_train, y_train)
#%%
def elastic_net_cv(x_train, y_train):
    elasticnet_model = ElasticNet(random_state=42)

    elasticnet_parameters = {
        'alpha': [0.01, 0.1, 0.5, 1.0, 10.0, 90.0],
        'l1_ratio': [0.01, 0.1, 0.5, 0.9],
        'fit_intercept': [True, False]
    }

    elasticnet_scoring = {
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'MSE': make_scorer(mean_squared_error, greater_is_better=False),
        'R-squared': make_scorer(r2_score)
    }

    elasticnet_grid_search = GridSearchCV(
                                elasticnet_model,
                                elasticnet_parameters,
                                scoring=elasticnet_scoring,
                                cv=12,
                                refit='R-squared',
                                n_jobs=-1
                            )

    return elasticnet_grid_search.fit(x_train, y_train)
#%%
def bayesian_ridge_cv(x_train, y_train):
    bayesianridge_model = BayesianRidge()

    bayesianridge_parameters = {'alpha_1': [1e-6, 1e-4, 1e-2],
                'alpha_2': [1e-6, 1e-4, 1e-2],
                'lambda_1': [1e-6, 1e-4, 1e-2],
                'lambda_2': [1e-6, 1e-4, 1e-2],
                'fit_intercept': [True, False]}

    bayesianridge_scoring = {
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'MSE': make_scorer(mean_squared_error, greater_is_better=False),
        'R-squared': make_scorer(r2_score)
    }

    bayesianridge_grid_search = GridSearchCV(
                                    bayesianridge_model,
                                    bayesianridge_parameters,
                                    scoring=bayesianridge_scoring,
                                    cv=2,
                                    refit='R-squared',
                                    n_jobs=-1
                                )

    return bayesianridge_grid_search.fit(x_train, y_train)
#%%
def sgd_regressor_cv(x_train, y_train):
    sgd_model = SGDRegressor(random_state=42)

    sgd_parameters = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 0.9],
        'l1_ratio': [0.15, 0.5, 0.85],
        'fit_intercept': [True, False],
        'penalty': ['l1', 'l2', 'elasticnet']
    }

    sgd_scoring = {
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'MSE': make_scorer(mean_squared_error, greater_is_better=False),
        'R-squared': make_scorer(r2_score)
    }

    sgd_grid_search = GridSearchCV(
                            sgd_model,
                            sgd_parameters,
                            scoring=sgd_scoring,
                            cv=7,
                            refit='R-squared',
                            n_jobs=-1
                        )

    return sgd_grid_search.fit(x_train, y_train)
#%%
def knn_regressor_cv(x_train, y_train):
    knn_model = KNeighborsRegressor()

    knn_parameters = {
        'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'metric': ['euclidean', 'manhattan', 'chebyshev']
    }

    knn_scoring = {
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'MSE': make_scorer(mean_squared_error, greater_is_better=False),
        'R-squared': make_scorer(r2_score)
    }

    knn_grid_search = GridSearchCV(
                            knn_model,
                            knn_parameters,
                            scoring=knn_scoring,
                            cv=3,
                            refit='R-squared',
                            n_jobs=-1
                        )

    return knn_grid_search.fit(x_train, y_train)
#%%
def decision_tree_regressor_cv(x_train, y_train):
    dt_model = DecisionTreeRegressor(random_state=42)

    dt_parameters = {
        'criterion': ['poisson', 'friedman_mse', 'squared_error', 'absolute_error', 'mae'],
        'max_depth': [None, 5, 10, 15, 30],
        'min_samples_split': [2, 5, 10, 15, 20, 25],
        'min_samples_leaf': [1, 2, 4, 6]
    }

    dt_scoring = {
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'MSE': make_scorer(mean_squared_error, greater_is_better=False),
        'R-squared': make_scorer(r2_score)
    }

    dt_grid_search = GridSearchCV(
                        dt_model,
                        dt_parameters,
                        scoring=dt_scoring,
                        cv=3,
                        refit='R-squared'
                    )

    return dt_grid_search.fit(x_train, y_train)
#%%
def random_forest_regressor_cv(x_train, y_train):
    rf_model = RandomForestRegressor(random_state=42)

    rf_parameters = {
        'n_estimators': [5, 10, 50, 100, 150, 200, 150, 300, 350, 400],
        'max_depth': [None, 2, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 6],
        'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']
    }

    rf_scoring = {
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'MSE': make_scorer(mean_squared_error, greater_is_better=False),
        'R-squared': make_scorer(r2_score)
    }

    rf_grid_search = GridSearchCV(
                        rf_model,
                        rf_parameters,
                        scoring=rf_scoring,
                        cv=2,
                        refit='R-squared',
                        n_jobs=-1
                    )

    return rf_grid_search.fit(x_train, y_train)
#%%
def gradient_boosting_regressor_cv(x_train, y_train):
    gb_model = GradientBoostingRegressor(random_state=42)

    gb_parameters = {
        'learning_rate': [0.1, 0.01],
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [3, 5, 8, 10],
        'min_samples_split': [2, 5, 7, 9],
        'min_samples_leaf': [1, 2, 3, 4, 5]
    }

    gb_scoring = {
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'MSE': make_scorer(mean_squared_error, greater_is_better=False),
        'R-squared': make_scorer(r2_score)
    }

    gb_grid_search = GridSearchCV(
                        gb_model,
                        gb_parameters,
                        scoring=gb_scoring,
                        cv=2,
                        refit='R-squared',
                        n_jobs=-1
                    )

    return gb_grid_search.fit(x_train, y_train)
#%%
def svr_cv(x_train, y_train):
    svr_model = SVR()

    svr_parameters = {
        'C': [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0],
        'epsilon': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0],
        'kernel': ['linear']
    }

    svr_scoring = {
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'MSE': make_scorer(mean_squared_error, greater_is_better=False),
        'R-squared': make_scorer(r2_score)
    }

    svr_grid_search = GridSearchCV(
                        svr_model,
                        svr_parameters,
                        scoring=svr_scoring,
                        cv=3,
                        refit='R-squared',
                        n_jobs=-1
                    )

    return svr_grid_search.fit(x_train, y_train)
#%%
def logistic_regressor_cv(x_train, y_train):
    logreg_model = LogisticRegression(random_state=42)

    logreg_parameters = {
        'C': [0.1, 1.0, 5.0, 6.0, 8.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

    logreg_scoring = {
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'MSE': make_scorer(mean_squared_error, greater_is_better=False),
        'R-squared': make_scorer(r2_score)
    }

    logreg_grid_search = GridSearchCV(
                            logreg_model,
                            logreg_parameters,
                            scoring=logreg_scoring,
                            cv=2,
                            refit='R-squared'
                        )

    return logreg_grid_search.fit(x_train, y_train)
#%%
def error(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('Média do valor absoluto dos erros: ', mae)
    print('Média dos erros quadráticos', mse)
    print('Coeficiente de determinação (R²):', r2)
#%%
def best_params(model_gs):
    best_params = model_gs.best_params_
    print("Melhores parâmetros encontrados:", best_params)
#%%
def r2(y_test, y_pred):
    return r2_score(y_test, y_pred)
#%%