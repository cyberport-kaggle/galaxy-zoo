"""
Classes for Galaxy Zoo
"""


from scipy import misc
from skimage import color
import numpy as np
from matplotlib import pyplot


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
        center = self.data.shape[0] / 2
        dim = size / 2
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
    def average_intensity(self):
        return self.data.mean()

