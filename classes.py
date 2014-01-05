"""
Classes for Galaxy Zoo
"""


from __future__ import division
from scipy import misc
from skimage import color
import numpy as np
from matplotlib import pyplot
from constants import *
import os


class Submission(object):
    """
    Wrapper for a submission dataset
    """
    def __init__(self):
        pass

    def from_file(self, filename):
        """
        Load a submission from a csv file
        """
        pass

    def to_file(self, filename):
        """
        Output a submission to a file
        """
        pass

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
        # ...but apparently wrong
        # self.data = (0.2125 * self.data[:, :, 0]) + (0.7154 * self.data[:, :, 1]) + (0.0721 * self.data[:, :, 2])
        self.data = color.rgb2gray(self.data)
        return self

    def flatten(self):
        self.data = self.data.flatten()
        return self

    def show(self, *args, **kwargs):
        pyplot.imshow(self.data, **kwargs)
        pyplot.show()

    @property
    def central_pixel(self):
        """
        Gets central pixel.  If data is 2D, then it's a number, otherwise it's an numpy array
        """
        width = self.data.shape[1]
        height = self.data.shape[0]
        central_coord = (int(height / 2), int(width / 2))
        return self.data[central_coord[0], central_coord[1]]

    @property
    def average_intensity(self):
        return self.data.mean()


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
    return np.sqrt(np.mean(np.square(first - second)))


def get_test_ids():
    """
    Gets a (79971, 1) numpy array that can be attached for output
    """
    test_files = sorted(os.listdir(TEST_IMAGE_PATH))
    return np.array(map(lambda x: int(x[0:6]), test_files), ndmin=2).T
