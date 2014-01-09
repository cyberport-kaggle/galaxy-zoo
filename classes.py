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
from constants import *
import os
from constants import N_TEST
import logging


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
    """
    Base model for training models.
    Rationale for having a class structure for models is so that we can:
      1) Do some standard utility things like timing
      2) Easily vary our models.  For example, if we have a model, and we want to have a variant that uses slightly different
         predictors, we can just subclass and override that part of the run function

    Subclasses must implement execute(), which is called by run().
    """
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

    def execute(self):
        raise NotImplementedError("Don't use the base class")

    def run(self):
        start_time = time.clock()

        res = self.execute()

        end_time = time.clock()
        logger.info("Model completed in {}".format(end_time - start_time))
        return res


def get_training_filenames(training_data=None):
    """
    Gets the list of training file names in order of the solutions file
    Returns a list

    Alternatively, if you've already loaded the training data, you can pass it in to prevent loading it twice
    """
    solution = training_data if training_data is not None else get_training_data()
    assert solution.shape == (N_TRAIN, 38), "Training data dimensions incorrect: was {}, expected {}".format(solution.shape, (N_TRAIN, 37))
    return map(lambda x: str(int(x)) + '.jpg', list(solution[:, 0]))


def get_training_data():
    solutions_file = 'data/solutions_training.csv'
    solution = np.loadtxt(solutions_file, delimiter=',', skiprows=1)
    return solution


def rmse(first, second):
    """
    Calculates rmse for two numpy arrays
    """
    res = np.sqrt(np.mean(np.square(first - second)))
    logger.info("In sample RMSE: {}".format(res))
    return res


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
