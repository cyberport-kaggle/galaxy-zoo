import inspect
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from models.Base import BaseModel

from classes import logger


class RidgeClipped(Ridge):
    def predict(self, X):
        pred = super(RidgeClipped, self).predict(X)

        # clip predictions to 0 and 1.
        pred[pred > 1] = 1
        pred[pred < 0] = 0

        return pred


# Do ridge regression and then random forest
class RidgeRFEstimator(BaseEstimator):
    def __init__(self,
                 alpha=14.0,
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features="auto",
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=3,
    ):
        # Ridge params
        self.alpha = alpha

        # RF Params
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.bootstrap = bootstrap
        self.oob_score = oob_score

        # Common
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def _check_fitted(self):
        if not hasattr(self, "ridge_estimator_"):
            raise AttributeError("Model has not been trained yet.")

    def _populate_args(self, cls):
        args, varargs, kw, default = inspect.getargspec(cls.__init__)
        # Pop self
        args.pop(0)
        init_args = {}
        for a in args:
            val = getattr(self, a, None)
            if val is not None:
                init_args[a] = val

        return init_args

    def _get_ridge_model(self):
        init_args = self._populate_args(Ridge)
        return Ridge(**init_args)

    def _get_rf_model(self):
        init_args = self._populate_args(RandomForestRegressor)
        return RandomForestRegressor(**init_args)

    def fit(self, X, y):
        self.ridge_estimator_ = self._get_ridge_model()
        self.rf_estimator_ = self._get_rf_model()
        logger.info("Fitting Ridge model")
        self.ridge_estimator_.fit(X, y)
        ridge_y = self.ridge_estimator_.predict(X)
        logger.info("Fitting RF model")
        self.rf_estimator_.fit(ridge_y, y)

    def predict(self, X):
        self._check_fitted()
        ridge_y = self.ridge_estimator_.predict(X)
        return self.rf_estimator_.predict(ridge_y)


class RidgeRFModel(BaseModel):
    train_predictors_file = 'data/data_ridge_rf_train_001.npy'
    test_predictors_file = 'data/data_ridge_rf_test_001.npy'
    n_features = 675
    estimator_defaults = {
        'n_estimators': 100,
        'alpha': 14
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
