"""
Classes for Galaxy Zoo
"""


from __future__ import division
from functools import wraps
from scipy import misc
from skimage import color
import numpy as np
from matplotlib import pyplot
import time
from sklearn import grid_search, cross_validation
from sklearn.metrics import mean_squared_error, make_scorer
from constants import *
import os
import logging
from sklearn.linear_model import Ridge
from skimage.transform import rescale
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator

logger = logging.getLogger('galaxy')
logger.setLevel(logging.DEBUG)
log_formatter = logging.Formatter('%(asctime)s - %(module)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
# Log to file
logfile = logging.FileHandler('run.log')
logfile.setLevel(logging.DEBUG)
logfile.setFormatter(log_formatter)
# Log to console
logstream = logging.StreamHandler()
logstream.setLevel(logging.INFO)
logstream.setFormatter(log_formatter)

logger.addHandler(logfile)
logger.addHandler(logstream)


class TrainSolutions(object):
    """
    Utility class for storing training Ys
    """
    def __init__(self):
        solution = np.loadtxt(TRAIN_SOLUTIONS_FILE, delimiter=',', skiprows=1)
        self.data = solution[:, 1:]
        self.filenames = map(lambda x: str(int(x)) + '.jpg', list(solution[:, 0]))


def rmse(y_true, y_pred):
    """
    Calculates rmse for two numpy arrays
    """
    res = np.sqrt(mean_squared_error(y_true, y_pred))
    logger.info("RMSE: {}".format(res))
    return res


# Scorer that can be used with Scikit-learn CV
rmse_scorer = make_scorer(rmse, greater_is_better=False)


def get_test_ids():
    """
    Gets a (79971, 1) numpy array that can be attached for output
    """
    test_files = sorted(os.listdir(TEST_IMAGE_PATH))
    return np.array(map(lambda x: int(x[0:6]), test_files), ndmin=2).T


def cache_to_file(filename, fmt='%.18e'):
    """
    Decorator for wrapping methods so that the result of those methods are written to a file and cached
    If the file exists, then the method will instead read from the file.
    Any function that is wrapped by this shouldn't have side effects (e.g. set properties on the instance)
    """
    def cache_decorator(func):
        @wraps(func)
        def cached_func(*args, **kwargs):
            if os.path.exists(filename):
                logger.info("Result of {} already exists, loading from file {}".format(func.__name__, filename))
                res = np.loadtxt(filename, delimiter=',')
            else:
                res = func(*args, **kwargs)
                logger.info("Caching results of {} to {}".format(func.__name__, filename))
                np.savetxt(filename, res, delimiter=',', fmt=fmt)
            return res
        return cached_func
    return cache_decorator


class Submission(object):

    """
    Utility function that takes care of some common tasks relating to submissiosn files
    """
    submission_format = ['%i'] + ['%.10f' for x in range(37)]

    def __init__(self, data):
        """
        Initialize with the (N_TEST, 37) or (N_TEST, 38) ndarray
        If there are 38 columns, then we expect the first column to be the Galaxy Ids
        """
        if data.shape == (N_TEST, 38):
            # Strip out the rownames and save them separately
            self.row_names = data[:, 0:1]  # don't flatten to 1d array
            self.data = data[:, 1:]
        elif data.shape == (N_TEST, 37):
            # No colnames, so assume that the colnames are the test ids sorted
            self.data = data
            self.row_names = get_test_ids()
        else:
            raise RuntimeError("Submission data must be of the shape ({}, 37), was: {}".format(N_TEST, data.shape))

    @staticmethod
    def from_file(filename):
        """
        Load a submission from a csv file
        """
        solution = np.loadtxt(filename, delimiter=',', skiprows=1)
        return Submission(solution)

    def to_file(self, filename):
        """
        Output a submission to a file
        """
        predictions = np.concatenate((self.row_names, self.data), axis=1)
        outpath = os.path.join(SUBMISSION_PATH, filename)
        logger.info("Saving solutions to file {}".format(outpath))
        np.savetxt(outpath, predictions, delimiter=',', header=SUBMISSION_HEADER, fmt=self.submission_format, comments="")

    def check_count(self):
        """
        Check that the number of rows is correct
        """
        pass
    def check_probabilities(self):
        """
        Ensure that probabilities for subsequent questions in the tree add up to their parent's probability
        See http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/details/the-galaxy-zoo-decision-tree
        """
        pass


class RawImage(object):

    """
    Used to load raw image files
    """

    def __init__(self, filename):
        """
        Given a file name, load the ndarray
        """
        self._original_data = misc.imread(filename)
        self.data = self._original_data.copy()
        self.gid = filename[0:6]
        self.responses = np.zeros(37)

    def revert(self):
        """
        Reverts to original state
        """
        self.data = self._original_data.copy()
        return self

    def crop(self, size):
        """
        Crops the image from the center into a square with sides of size
        """
        center = int(self.data.shape[0] / 2)
        dim = int(size / 2)
        cropmin = center - dim
        cropmax = center + dim
        self.data = self.data[cropmin:cropmax, cropmin:cropmax]
        return self

    def grayscale(self):
        # Faster than the skimage.color.rgb2gray
        self.data = ((0.2125 * self.data[:, :, 0]) + (0.7154 * self.data[:, :, 1]) + (0.0721 * self.data[:, :, 2])) / 255
        return self

    def flatten(self):
        self.data = self.data.flatten()
        return self

    def grid_sample(self, step_size, steps):
        """
        Samples the pixels from the center in steps.  Total pixels return will be
        """
        # Get a list of the coordinates of the pixels we want to extract
        central_coord = self.central_pixel_coordinates
        top_left = (central_coord[0] - (steps * step_size), central_coord[1] - (steps * step_size))
        bottom_right = (central_coord[0] + (steps * step_size), central_coord[1] + (steps * step_size))
        min_x = top_left[0]
        max_x = bottom_right[0] + 1
        min_y = top_left[1]
        max_y = bottom_right[1] + 1
        return self.data[min_x:max_x:step_size, min_y:max_y:step_size].copy()

    def show(self, *args, **kwargs):
        pyplot.imshow(self.data, **kwargs)
        pyplot.show()

    def rescale(self, scale):
        self.data = rescale(self.data, scale)

    @property
    def central_pixel_coordinates(self):
        width = self.data.shape[1]
        height = self.data.shape[0]
        central_coord = (int(height / 2), int(width / 2))
        return central_coord

    @property
    def central_pixel(self):
        """
        Gets central pixel.  If data is 2D, then it's a number, otherwise it's an numpy array
        """
        central_coord = self.central_pixel_coordinates
        return self.data[central_coord[0], central_coord[1]]
    @property
    def average_intensity(self):
        return self.data.mean()


class BaseModel(object):
    # Filenames used to store the feature arrays used in fitting/predicting
    """
    Base model for training models.
    Rationale for having a class structure for models is so that we can:
      1) Do some standard utility things like timing
      2) Easily vary our models.  For example, if we have a model, and we want to have a variant that uses slightly different
         predictors, we can just subclass and override that part of the run function
    """
    # This is so that we don't have to iterate over all 70k images every time we fit.
    train_predictors_file = None
    test_predictors_file = None
    # Number of features that the model will generate
    n_features = None

    def __init__(self, *args, **kwargs):
        self.training_data = TrainSolutions()
        self.train_x = None
        self.train_y = self.training_data.data
        self.test_x = None
        self.estimator = None
        # Parameters for the grid search
        self.grid_search_parameters = kwargs.get('grid_search_parameters', None)
        self.grid_search_estimator = None
        # Sample to use for the grid search.  Should be between 0 and 1
        self.grid_search_sample = kwargs.get('grid_search_sample', None)
        self.grid_search_x = None
        self.grid_search_y = None

    def do_for_each_image(self, files, func, n_features, training):
        """
        Function that iterates over a list of files, applying func to the image indicated by that function.
        Returns an (n_samples, n_features) ndarray
        """
        dims = (N_TRAIN if training else N_TEST, n_features)
        predictors = np.zeros(dims)
        counter = 0
        for row, f in enumerate(files):
            filepath = TRAIN_IMAGE_PATH if training else TEST_IMAGE_PATH
            image = RawImage(os.path.join(filepath, f))
            predictors[row] = func(image)
            counter += 1
            if counter % 1000 == 0:
                logger.info("Processed {} images".format(counter))
        return predictors

    def build_features(self, files, training=True):
        """
        Utility method that loops over every image and applies self.process_image
        Returns a numpy array of dimensions (n_observations, n_features)
        """
        logger.info("Building predictors")
        predictors = self.do_for_each_image(files, self.process_image, self.n_features, training)
        return predictors

    def build_train_predictors(self):
        """
        Builds the training predictors.  Once the predictors are built, they are cached to a file.
        If the file already exists, the predictors are loaded from file.
        Couldn't use the @cache_to_file decorator because the decorator factory doesn't have access to self at compilation

        Returns:
            A numpy array of shape (n_train, n_features)
        """
        file_list = self.training_data.filenames
        if os.path.exists(self.train_predictors_file):
            logger.info("Training predictors already exists, loading from file {}".format(self.train_predictors_file))
            res = np.load(self.train_predictors_file)
        else:
            res = self.build_features(file_list, True)
            logger.info("Caching training predictors to {}".format(self.train_predictors_file))
            np.save(self.train_predictors_file, res)
        return res

    def build_test_predictors(self):
        """
        Builds the test predictors

        Returns:
            A numpy array of shape (n_test, n_features)
        """
        test_files = sorted(os.listdir(TEST_IMAGE_PATH))
        if os.path.exists(self.test_predictors_file):
            logger.info("Test predictors already exists, loading from file {}".format(self.test_predictors_file))
            res = np.load(self.test_predictors_file)
        else:
            res = self.build_features(test_files, False)
            logger.info("Caching test predictors to {}".format(self.test_predictors_file))
            np.save(self.test_predictors_file, res)
        return res

    def perform_grid_search_and_cv(self, *args, **kwargs):
        """
        Performs cross validation and grid search to identify optimal parameters and to score the estimator
        The grid search space is defined by self.grid_search_parameters.

        If grid_search_sample is defined, then a downsample of the full train_x is used to perform the grid search
        """
        if self.grid_search_parameters is not None:
            logging.info("Performing grid search")
            start_time = time.clock()
            self.grid_search_estimator = grid_search.GridSearchCV(self.get_estimator(),
                                                                  self.grid_search_parameters,
                                                                  scoring=rmse_scorer, verbose=3, **kwargs)
            if self.grid_search_sample is not None:
                logging.info("Using {} of the train set for grid search".format(self.grid_search_sample))
                # Downsample if a sampling rate is defined
                self.grid_search_estimator.refit = False
                self.grid_search_x, \
                self.grid_search_x_test, \
                self.grid_search_y, \
                self.grid_search_y_test = cross_validation.train_test_split(self.train_x,
                                                                            self.train_y,
                                                                            train_size=self.grid_search_sample)
            else:
                logging.info("Using full train set for the grid search")
                # Otherwise use the full set
                self.grid_search_x = self.train_x
                self.grid_search_y = self.train_y
            self.grid_search_estimator.fit(self.grid_search_x, self.grid_search_y)
            logger.info("Grid search completed in {}".format(time.clock() - start_time))

    def run(self):
        start_time = time.clock()

        res = self.execute()
        # A general workflow for a model would be as follows:
        # 1) Generate the features by iterating over each image
        # 3) Perform Grid Search and CV to get the best model
        # 4) Fit the best estimator on the full dataset
        # 5) Use the estimator to predict on the test set

        end_time = time.clock()
        logger.info("Model completed in {}".format(end_time - start_time))
        return res

    def execute(self):
        raise NotImplementedError("Don't use the base class")

    def get_estimator(self):
        """
        Returns a Scikit-learn estimator used in the final model.
        Subclasses should implement this method
        """
        raise NotImplementedError("Subclasses of BaseModel should implement get_estimator")
    @staticmethod
    def process_image(img):
        """
        A function that takes a RawImage object and returns a (1, n_features) numpy array
        Subclasses should implement this method
        """
        raise NotImplementedError("Subclasses of BaseModel should implement process_image")


class RidgeClipped(Ridge):
    def predict(self, X):
        pred = super(RidgeClipped, self).predict(X)

        # clip predictions to 0 and 1.
        pred[pred > 1] = 1
        pred[pred < 0] = 0

        return pred


# Do ridge regression and then random forest
class RidgeRF(BaseEstimator):
    def __init__(self, alpha=1.0, n_estimators=10):
        self.ridge_rgn = Ridge(alpha=14.0)
        self.rf_rgn = RandomForestRegressor(n_estimators=100)

    def fit(self, X, y):
        self.ridge_rgn.fit(X, y)
        ridge_y = self.ridge_rgn.predict(X)
        self.rf_rgn.fit(ridge_y, y)

    def predict(self, X):
        ridge_y = self.ridge_rgn.predict(X)
        return self.rf_rgn.predict(ridge_y)
