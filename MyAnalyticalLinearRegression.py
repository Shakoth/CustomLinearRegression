import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import NotFittedError

class CustomAnalyticalLinearRegression:

    def __init__(self, fit_intercept=True):
        self.W = None
        self.fit_intercept = fit_intercept

    def RSS(self, y_true, y_pred):
        residuals = y_true - y_pred
        return residuals.T @ residuals

    def residual_variance(self, y_true, y_pred, p):
        return self.RSS(y_true, y_pred)/(y_true.shape[0] - p)

    def w_var(self, y_true, y_pred, X):
        return np.diagonal(np.linalg.inv(X.T @ X) * self.residual_variance(y_true, y_pred, X.shape[1]))

    def w_standard_errors(self, y_true, y_pred, X):
        return np.sqrt(self.w_var(y_true, y_pred, X)).reshape(-1,1)
    
    def t_stats(self):
        if self.W is None:
            raise NotFittedError("This CustomLinearRegression instance is not fitted yet, run fit method.")
        return self.W / self.errors
    
    def t_test(self):
        if self.W is None:
            raise NotFittedError("This CustomLinearRegression instance is not fitted yet, run fit method.")
        return stats.t.sf(abs(self.t), self.n_samples) * 2

    def _add_intercept(self, X):
        if self.fit_intercept:
            ones_column = np.ones((X.shape[0], 1))
            return np.hstack((ones_column, X)) 
        else:
            return X

    def fit(self, X, y):                                                    
        y = y.reshape(-1, 1)
        self.n_samples = X.shape[0]
        X_new = self._add_intercept(X)

        self.W = np.linalg.inv(X_new.T @ X_new) @ X_new.T @ y
        y_pred = self.predict(X)
        self.errors = self.w_standard_errors(y, y_pred, X_new)
        self.t = self.t_stats()
        self.pvalues = self.t_test()
        self.results = pd.DataFrame(
            {'coef': self.W.reshape(-1), 
             'std_errors': self.errors.reshape(-1), 
             't-test': self.t.reshape(-1), 
             'p-values':self.pvalues.reshape(-1)})

    def predict(self, X):
        if self.W is None:
            raise NotFittedError("This CustomLinearRegression instance is not fitted yet, run fit method.")
        X_new = self._add_intercept(X)
        y_pred = X_new @ self.W
        return y_pred

    def __repr__(self):
        return "Linear Regression model"