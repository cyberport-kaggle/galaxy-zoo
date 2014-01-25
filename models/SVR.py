__author__ = 'mc'
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor

from models.Base import BaseModel
import numpy as np

class SVRRFModel(BaseEstimator):
    def __init__(self, alpha=14.0, n_estimators=100):
        self.svr_rgn = dict([(i, SGDRegressor()) for i in range(37)])
        self.rf_rgn = RandomForestRegressor(n_estimators=100)

    def fit(self, X, y):
        # loop through each column of y to train and predict
        svr_y = np.zeros_like(y)

        # Train SGD one column of y at a time
        for col in range(y.shape[1]):
            self.svr_rgn[col].fit(X, y[:, col])
            svr_y[:, col] = self.svr_rgn[col].predict(X)

        self.rf_rgn.fit(svr_y, y)

    def predict(self, X):
        svr_y = np.zeros((X.shape[0], 37))
        for col in range(37):
            svr_y[:, col] = self.svr_rgn[col].predict(X)

        return self.rf_rgn.predict(svr_y)