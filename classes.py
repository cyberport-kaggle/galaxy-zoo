"""
Classes for Galaxy Zoo
"""


from scipy import misc
from skimage import color


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
        self.data = misc.imread(filename)
        self.gid = filename[0:6]

    def crop(self):
        pass

    def grayscale(self):
        self.data = color.rgb2gray(self.data).flatten()
