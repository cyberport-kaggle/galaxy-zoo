from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from models.Base import BaseModel


class RidgeClipped(Ridge):
    def predict(self, X):
        pred = super(RidgeClipped, self).predict(X)

        # clip predictions to 0 and 1.
        pred[pred > 1] = 1
        pred[pred < 0] = 0

        return pred


# Do ridge regression and then random forest
class RidgeRFEstimator(BaseEstimator):
    def __init__(self, alpha=14.0, n_estimators=100):
        # For some reason, passing in the alpha gives an exception when fitting the ridge
        """
        /home/hxu/src/galaxy-zoo/models/Ridge.py in fit(self, X, y)
             23
             24     def fit(self, X, y):
        ---> 25         self.ridge_rgn.fit(X, y)
             26         ridge_y = self.ridge_rgn.predict(X)
             27         self.rf_rgn.fit(ridge_y, y)

        /home/hxu/src/scikit-learn/sklearn/linear_model/ridge.pyc in fit(self, X, y, sample_weight)
            447         self : returns an instance of self.
        --> 449         return super(Ridge, self).fit(X, y, sample_weight=sample_weight)
            450
            451

        /home/hxu/src/scikit-learn/sklearn/linear_model/ridge.pyc in fit(self, X, y, sample_weight)
            336                                       max_iter=self.max_iter,
            337                                       tol=self.tol,
        --> 338                                       solver=self.solver)
            339         self._set_intercept(X_mean, y_mean, X_std)
            340         return self

        /home/hxu/src/scikit-learn/sklearn/linear_model/ridge.pyc in ridge_regression(X, y, alpha, sample_weight, solver, max_iter, tol)
            295         else:
            296             try:
        --> 297                 coef = _solve_dense_cholesky(X, y, alpha)
            298             except linalg.LinAlgError:
            299                 # use SVD solver if matrix is singular

        /home/hxu/src/scikit-learn/sklearn/linear_model/ridge.pyc in _solve_dense_cholesky(X, y, alpha)
             96
             97     if one_alpha:
        ---> 98         A.flat[::n_features + 1] += alpha[0]
             99         return linalg.solve(A, Xy, sym_pos=True,
            100                             overwrite_a=True).T

        TypeError: unsupported operand type(s) for +: 'float' and 'NoneType'

        """
        self.ridge_rgn = Ridge(alpha=14)
        self.rf_rgn = RandomForestRegressor(n_estimators=100)

    def fit(self, X, y):
        self.ridge_rgn.fit(X, y)
        ridge_y = self.ridge_rgn.predict(X)
        self.rf_rgn.fit(ridge_y, y)

    def predict(self, X):
        ridge_y = self.ridge_rgn.predict(X)
        return self.rf_rgn.predict(ridge_y)


class RidgeRFModel(BaseModel):
    train_predictors_file = 'data/data_ridge_rf_train_001.npy'
    test_predictors_file = 'data/data_ridge_rf_test_001.npy'
    n_features = 675
    estimator_defaults = {
        'n_estimators': 100,
        'alpha': [14]
    }
    estimator_class = RidgeRFEstimator

    @staticmethod
    def process_image(img):
        crop_size = 150
        scale = 0.1
        img.crop(crop_size)
        img.rescale(scale)
        img.flatten()
        return img.data
