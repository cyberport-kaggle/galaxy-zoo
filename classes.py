"""
Classes for Galaxy Zoo
"""

from __future__ import division
from functools import wraps
import os
import logging
from IPython import embed

from scipy import misc
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error, make_scorer
from skimage.transform import rescale
from sklearn.cluster import MiniBatchKMeans
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.feature_extraction.image import extract_patches_2d
from numpy.lib.stride_tricks import as_strided

from constants import *

logging.basicConfig(level=logging.INFO)
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
    # Maps classes to columns, gids are removed
    class_map = {
        1: [0, 1, 2],
        2: [3, 4],
        3: [5, 6],
        4: [7, 8],
        5: [9, 10, 11, 12],
        6: [13, 14],
        7: [15, 16, 17],
        8: [18, 19, 20, 21, 22, 23, 24],
        9: [25, 26, 27],
        10: [28, 29, 30],
        11: [31, 32, 33, 34, 35, 36],
    }

    # Maps a class to the column that it should sum to
    # Keys are classes, while values are the corresponding column
    parent_class_map = {
        2: [1],
        3: [4],
        4: [5, 6],
        5: [8, 31, 32, 33, 34, 35, 36],
        # Class 6 should equal 1
        7: [0],
        8: [13],
        9: [3],
        10: [7],
        11: [28, 29, 30]
    }

    def __init__(self):
        solution = np.loadtxt(TRAIN_SOLUTIONS_FILE, delimiter=',', skiprows=1)
        self.data = solution[:, 1:]
        self.iids = solution[:, 0]
        self.filenames = map(lambda x: str(int(x)) + '.jpg', list(solution[:, 0]))

    @property
    def classes(self):
        """
        Iterator that returns columns of self.data grouped by class
        """
        for k, v in self.class_map.items():
            yield self.data[:, v]

    def get_columns_for_class(self, cls):
        """
        Returns the columns that correspond to class cls
        """
        return self.data[:, self.class_map[cls]]

    def get_sum_for_class(self, cls):
        """
        Returns a numpy columnar array of values to which the provided class should sum.
        E.g. doing get_sum_for_class(1) would return an array of 1s with dimensions (n_train, 1) since the columns
        of class 1 should always sum to 1
        """
        col_nums = self.parent_class_map.get(cls, None)
        if col_nums:
            cols = self.data[:, col_nums]
            return np.sum(cols, 1, keepdims=True)
        else:
            return np.ones((self.data.shape[0], 1))

    def get_rebased_columns_for_class(self, cls=None):
        """
        Returns the columns that correspond to class, but rebased so that the columns sum to 1
        """
        if cls:
            cols = self.get_columns_for_class(cls)
            colsums = np.sum(cols, 1, keepdims=True)
            # We can get some NaNs if some rows sum to 0
            # If it's 0, instead divide by 1
            colsums[colsums == 0] = 1
            return np.true_divide(cols, colsums)
        else:
            # Return for all classes
            res = np.zeros(self.data.shape)
            for i in range(1, 12):
                col_idxs = self.class_map[i]
                cols = self.get_rebased_columns_for_class(i)
                res[:, col_idxs] = cols
            return res


# Keep a single instance for the whole workspace
train_solutions = TrainSolutions()


def rmse(y_true, y_pred):
    """
    Calculates rmse for two numpy arrays
    """
    res = np.sqrt(mean_squared_error(y_true, y_pred))
    logger.info("RMSE: {}".format(res))
    return res


# Scorer that can be used with Scikit-learn CV
rmse_scorer = make_scorer(rmse, greater_is_better=False)


def colwise_rmse(y_true, y_pred):
    """
    Calculates the RMSE for each column of y
    """
    cols = range(0, y_true.shape[1])
    res = np.zeros(y_true.shape[1])
    logger.info("Column-wise RMSEs:")
    for c in cols:
        res[c] = np.sqrt(mean_squared_error(y_true[:, c], y_pred[:, c]))
        logger.info("Col {}: {:5f}".format(c, res[c]))
    return res


def classwise_rmse(y_true, y_pred):
    res = np.zeros(11)
    logger.info("Class-wise RMSEs:")
    for cls in range(1, 12):
        cols = train_solutions.class_map[cls]
        res[cls - 1] = np.sqrt(mean_squared_error(y_true[:, cols], y_pred[:, cols]))
        logger.info("Class {}: {:5f}".format(cls, res[cls - 1]))
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


def crop_to_memmap(crop_size=150, training=True):
    """
    Crop all training and testing images into two memmap array
    Train - the images will be in the same order as the train solutions files
    @return:
    """

    if training:
        path = TRAIN_IMAGE_PATH
        files = train_solutions.iids
        out = np.memmap('data/train_cropped_150.memmap', shape=(len(files), crop_size, crop_size, 3), mode='w+')
    else:
        path = TEST_IMAGE_PATH
        files = get_test_ids()
        out = np.memmap('data/test_cropped_150.memmap', shape=(len(files), crop_size, crop_size, 3), mode='w+')

    for i, f in enumerate(files):
        if i % 100 == 0:
            print i
        img = RawImage(path + '/' + str(int(f)) + '.jpg')
        img.crop(150)
        out[i] = img.data


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
        np.savetxt(outpath, predictions, delimiter=',', header=SUBMISSION_HEADER, fmt=self.submission_format,
                   comments="")

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
        self.data = (
                        (0.2125 * self.data[:, :, 0]) + (0.7154 * self.data[:, :, 1]) + (
                            0.0721 * self.data[:, :, 2])) / 255
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


class KMeansFeatures(object):
    """
    Implements Kmeans feature learning as per Adam Coates' MATLAB code

    Before using this, be sure ot run classes.crop_images_to_mmap to generate the training and test data

    fit() is the primary entry point for training the data
    transform() is the

    Parameters:
    ==========
    rf_size: int
        The size of the receptive fields (patches)

    num_centroids: int

    whitening: bool
        Perform ZCA whitening?

    num_patches: int
        The numbe rof patches to sample

    Properties:
    ===========
    patches: ndarray of shape (num_patchs, rf_size^2 * 3)
        ndarray of the patches.  If patches are 2x2 with three color channels, then stores a single patch as an array of
        dimension (1, 12) -- i.e. it is flattened

    trainX: memory mapped ndarray
        training data

    testX: memory mapped ndarray
        test data

    centroids: ???

    """

    def __init__(self, rf_size=6, num_centroids=1600, whitening=True, num_patches=400000):
        self.rf_size = rf_size
        self.num_centroids = num_centroids
        self.whitening = whitening
        self.num_patches = num_patches
        self.patches = np.zeros((num_patches, rf_size * rf_size * 3))

        # these appear to be hard coded and are basically constant -- probably don't need to store these on the object
        self.dim = np.array([150, 150, 3])
        # These actually do not need to be stored here -- the TrainSolutions class encapsulates these information
        # And it is instantiated with classes, so it is always available
        # id is an ndarray of training image ids
        self.id = train_solutions.iids
        # n is then the number of training images
        self.n = self.id.shape[0]

        # Fields for the results
        self.centroids = None

        # Training data
        self.trainX = np.memmap('data/train_cropped_150.memmap', mode='r', shape=(N_TRAIN, 150, 150, 3))
        self.testX = np.memmap('data/test_cropped_150.memmap', mode='r', shape=(N_TEST, 150, 150, 3))

    def extract_patches(self):
        # Extract patches from the training data
        for i in range(self.num_patches):
            if (i + 1) % 1000 == 0:
                print 'Extract patch {0} / {1}'.format(i + 1, self.num_patches)

            row = np.random.random_integers(self.dim[0] - self.rf_size)
            col = np.random.random_integers(self.dim[1] - self.rf_size)

            # img = RawImage(TRAIN_IMAGE_PATH + '/' + str(int(self.id[i % self.n])) + '.jpg')
            # img.crop(150)

            img = self.trainX[i % self.n]
            patch = img[row:row + self.rf_size, col:col + self.rf_size, :]
            self.patches[i, :] = patch.flatten()

    def normalize(self):
        temp1 = self.patches - self.patches.mean(1, keepdims=True)
        temp2 = np.sqrt(self.patches.var(1, keepdims=True) + 10)

        self.patches = temp1 / temp2

    def whiten(self):
        cov = np.cov(self.patches, rowvar=0)
        self.mean = self.patches.mean(0, keepdims=True)
        d, v = np.linalg.eig(cov)
        self.p = np.dot(v,
                        np.dot(np.diag(np.sqrt(1 / (d + 0.1))),
                               v.T))
        self.patches = np.dot(self.patches - self.mean, self.p)

    def cluster(self):
        kmeans = MiniBatchKMeans(n_clusters=self.num_centroids, verbose=True, batch_size=1000)
        kmeans.fit(self.patches)
        self.centroids = kmeans.cluster_centers_

    def fit(self):
        self.extract_patches()
        self.normalize()
        self.whiten()
        self.cluster()

        # clean up
        # self.patches = None

    def extract_features(self, x):
        # patches = np.vstack((self.im2col(x[:, :, 0]),
        #                      self.im2col(x[:, :, 1]),
        #                      self.im2col(x[:, :, 2]))).T

        patches = np.vstack((self.rolling_block(x[:, :, 0]),
                             self.rolling_block(x[:, :, 1]),
                             self.rolling_block(x[:, :, 2]))).T

        # normalize for contrast
        temp1 = patches - patches.mean(1, keepdims=True)
        temp2 = np.sqrt(patches.var(1, keepdims=True) + 10)
        patches = temp1 / temp2

        if self.whitening:
            patches = np.dot(patches - self.mean, self.p)

        xx = np.sum(patches ** 2, 1, keepdims=True)
        cc = np.sum(self.centroids ** 2, 1, keepdims=True).T
        xc = np.dot(patches, self.centroids.T)

        z = np.sqrt(cc + (xx - 2 * xc))
        mu = z.mean(1, keepdims=True)
        patches = np.maximum(mu - z, 0)

        prows = self.dim[0] - self.rf_size + 1
        pcols = self.dim[1] - self.rf_size + 1
        patches = patches.reshape((prows, pcols, self.num_centroids))

        halfr = int(np.rint(prows / 2))
        halfc = int(np.rint(pcols / 2))
        q1 = np.sum(patches[0:halfr, 0:halfc, :], (0, 1))
        q2 = np.sum(patches[halfr:, 0:halfc, :], (0, 1))
        q3 = np.sum(patches[0:halfr, halfc:, :], (0, 1))
        q4 = np.sum(patches[halfr:, halfc:, :], (0, 1))

        return np.hstack((q1.flatten(), q2.flatten(), q3.flatten(), q4.flatten()))

    def im2col(self, img):
        row, col = img.shape
        patches = np.zeros((self.rf_size * self.rf_size,
                            (row - self.rf_size + 1) * (col - self.rf_size + 1)))

        counter = 0
        for i in range(row - self.rf_size + 1):
            for j in range(col - self.rf_size + 1):
                temp = img[i:i + self.rf_size, j:j + self.rf_size]
                patches[:, counter] = temp.flatten()
                counter += 1

        return patches

    def rolling_block(self, A):
        block = (self.rf_size, self.rf_size)
        shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
        strides = (A.strides[0], A.strides[1]) + A.strides
        res = np.copy(as_strided(A, shape=shape, strides=strides))
        res = res.reshape(shape[0] * shape[1], shape[2] * shape[3]).T
        return res

    def transform(self, n):
        if n > self.trainX.shape[0]:
            n = self.trainX.shape[0]

        res = np.zeros((n, self.num_centroids * 4))

        for i, row in enumerate(self.trainX[0:n]):
            if (i + 1) % 10 == 0:
                logging.info('Extracting features from image {0} / {1}'.format(i + 1, n))

            res[i] = self.extract_features(row)

        return res

    def show_centroids(self):
        im = np.arange(100)
        im.shape = 10, 10

        fig = pyplot.figure(1, (4., 4.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(10, 10),  # creates 10x10 grid of axes
                         axes_pad=0.1,  # pad between axes in inch.
        )

        for grid_pos, i in enumerate(np.random.choice(self.centroids.shape[0], 100, replace=False)):
            grid[grid_pos].imshow(self.centroids[i].reshape(self.rf_size, self.rf_size, 3))  # The AxesGrid object work as a list of axes.

        pyplot.show()
