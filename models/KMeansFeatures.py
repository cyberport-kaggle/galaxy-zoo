from __future__ import division
import multiprocessing
import itertools
import math
from joblib import Parallel, delayed
from matplotlib import pyplot
from matplotlib.colors import Normalize
import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.cluster import MiniBatchKMeans, KMeans
from mpl_toolkits.axes_grid1 import ImageGrid
import classes
import logging
from constants import *

logger = logging.getLogger('galaxy')


def chunked_extract_patch(patch_nums, train_mmap, patch_size):
    """
    Extracts patches from images in chuncks

    Arguments:
    =========
    patch_nums: list of integers
        The image numbers from which to extract patches

    train_mmap: mem-mapped ndarray
        The image training set.  Dimensions of (n_training, image_height, image_width, channels)

    patch_size: int
        size of the patch to extract


    Returns:
    ========
    ndarray of shape (len(patch_nums), patch_size * patch_size * channels)

    """
    # Filter out any Nones that might have been passed in
    patch_nums = [x for x in patch_nums if x is not None]
    res = [None] * len(patch_nums)

    for i, p in enumerate(patch_nums):
        # train_mmap is of dimensions (n_training, image_rows, image_cols, channels)
        image_rows = train_mmap.shape[1]
        image_cols = train_mmap.shape[2]

        # Randomly get an offset
        row = np.random.random_integers(image_rows - patch_size)
        col = np.random.random_integers(image_cols - patch_size)

        # Pick the right image and extract the patch
        img = train_mmap[p % image_rows]
        patch = img[row:row + patch_size, col:col + patch_size, :]
        res[i] = patch.flatten()

    return np.vstack(res)


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

        # Fields for the results - populated after self.fit()
        self.centroids = None  # shape (num_centroids, rf_size * rf_size * channels)
        self.p = None  # shape (rf_size * rf_size * channels, rf_size * rf_size * channels)
        self.mean = None  # column means (1, rf_size * rf_size * channels)

        # Training data
        self.trainX = None

    def extract_patches(self):
        """
        Extracts patches from the training data
        Does this by looping through the images, taking one patch from each image, until we get the number
        of patches we want
        """
        # Find out the number of threads to split into
        cores = multiprocessing.cpu_count()
        patch_rng = range(self.num_patches)
        chunk_size = int(math.ceil(self.num_patches / cores))

        # Gives a list of tuples, with the list lenght being equal to the number of cores, and each tuple containing
        # The original range split into chunks.  The shortest chunk is padded with Nones
        # For why this works, see http://docs.python.org/2/library/functions.html#zip
        patch_chunks = list(itertools.izip_longest(*[iter(patch_rng)] * chunk_size))

        logger.info("Extracting patches in {} jobs, chunk sizes: {}".format(cores, [len(x) for x in patch_chunks]))
        res = Parallel(n_jobs=cores, verbose=3)(delayed(chunked_extract_patch)(x, self.trainX, 6) for x in patch_chunks)
        self.patches = np.vstack(res)

    def whiten(self):
        """
        ZCA whitening
        """
        cov = np.cov(self.patches, rowvar=0)
        self.mean = self.patches.mean(0, keepdims=True)
        d, v = np.linalg.eig(cov)
        self.p = np.dot(v,
                        np.dot(np.diag(np.sqrt(1 / (d + 0.1))),
                               v.T))
        self.patches = np.dot(self.patches - self.mean, self.p)

    def cluster(self):
        # kmeans = MiniBatchKMeans(n_clusters=self.num_centroids,
        #                          verbose=True,
        #                          batch_size=self.num_centroids * 20,
        #                          compute_labels=False,
        #                          max_iter=5,
        #                          max_no_improvement=None,
        # )
        kmeans = KMeans(n_clusters=self.num_centroids, verbose=10, n_jobs=-1)
        kmeans.fit(self.patches)
        self.kmeans = kmeans
        self.centroids = kmeans.cluster_centers_

    def fit(self, trainX):
        self.trainX = trainX
        logger.info("Extracting patches")
        self.extract_patches()
        logger.info("Normalizing")
        self.patches = normalize(self.patches)
        logger.info("Whitening")
        self.whiten()
        logger.info("Clustering")
        self.cluster()

        # clean up
        # self.patches = None

    def transform(self, x):
        cores = multiprocessing.cpu_count()
        n = x.shape[0]
        rng = range(n)
        chunk_size = int(math.ceil(n / cores))
        chunks = list(itertools.izip_longest(*[iter(rng)] * chunk_size))
        logger.info("Transforming in {} jobs, chunk sizes: {}".format(cores, [len(c) for c in chunks]))

        res = Parallel(n_jobs=cores, verbose=3)(
            delayed(chunked_extract_features)(i, x, self.rf_size, self.centroids, self.mean, self.p, self.whitening) for i in chunks
        )
        res = np.vstack(res)

        return res

    def show_centroids(self):
        im = np.arange(100)
        im.shape = 10, 10

        fig = pyplot.figure(1, (4., 4.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(20, 20),  # creates 20x20 grid of axes
                         axes_pad=0.1,  # pad between axes in inch.
        )

        for grid_pos, i in enumerate(np.random.choice(self.centroids.shape[0], 400, replace=False)):
            grid[grid_pos].imshow(self.centroids[i].reshape(3, self.rf_size, self.rf_size).swapaxes(0, 2))  # The AxesGrid object work as a list of axes.

        pyplot.show()


def show_centroids(centroids, reshape=(3, 6, 6)):
        fig = pyplot.figure(1, (4., 4.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(20, 20),  # creates 20x20 grid of axes
                         axes_pad=0.1,  # pad between axes in inch.
        )

        for grid_pos, i in enumerate(np.random.choice(centroids.shape[0], 400, replace=False)):
            norm = Normalize(vmin=-1.5, vmax=1.5, clip=True)
            grid[grid_pos].imshow(centroids[i].reshape(reshape).swapaxes(0, 2), norm=norm)  # The AxesGrid object work as a list of axes.
        pyplot.show()


def normalize(x):
    """
    Normalizes each patch by subtracting mean and dividing by variance
    """
    temp1 = x - x.mean(1, keepdims=True)
    temp2 = np.sqrt(x.var(1, keepdims=True) + 10)

    return temp1 / temp2


def chunked_extract_features(idx, x, rf_size, centroids, mean, p, whitening=True):
    """
    Receives a list of image indices to extract features from

    Arguments:
    ==========
    i: list of ints
        Indices of images to extract

    trainX: memmapped ndarray

    rf_size: int

    centroids: ndarray

    mean: ndarray

    p: ndarray
    """
    idx = [x for x in idx if x is not None]
    res = [None] * len(idx)
    for i, img_idx in enumerate(idx):
        if (i + 1) % 10 == 0:
            logger.info("Extracting features on image {} / {}".format(i + 1, len(idx)))

        x = x[img_idx]
        patches = np.vstack((rolling_block(x[:, :, 0], rf_size),
                             rolling_block(x[:, :, 1], rf_size),
                             rolling_block(x[:, :, 2], rf_size))).T

        # normalize for contrast
        patches = normalize(patches)

        if whitening:
            patches = np.dot(patches - mean, p)

        xx = np.sum(patches ** 2, 1, keepdims=True)
        cc = np.sum(centroids ** 2, 1, keepdims=True).T
        xc = np.dot(patches, centroids.T)

        z = np.sqrt(cc + (xx - 2 * xc))
        mu = z.mean(1, keepdims=True)
        patches = np.maximum(mu - z, 0)

        # 150 is hard coded in the crop size, which is actually pre-determined by classes.crop_image_to_mmap
        # So don't need to reference the attribute here
        prows = 150 - rf_size + 1
        pcols = 150 - rf_size + 1
        num_centroids = centroids.shape[0]
        patches = patches.reshape((prows, pcols, num_centroids))

        halfr = int(np.rint(prows / 2))
        halfc = int(np.rint(pcols / 2))
        q1 = np.sum(patches[0:halfr, 0:halfc, :], (0, 1))
        q2 = np.sum(patches[halfr:, 0:halfc, :], (0, 1))
        q3 = np.sum(patches[0:halfr, halfc:, :], (0, 1))
        q4 = np.sum(patches[halfr:, halfc:, :], (0, 1))

        res[i] = np.hstack((q1.flatten(), q2.flatten(), q3.flatten(), q4.flatten()))

    return np.vstack(res)


def rolling_block(A, block_size):
    """
    Gets a rolling window on A in a square with side length block_size
    Uses the stride trick
    """
    block = (block_size, block_size)
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    res = np.copy(as_strided(A, shape=shape, strides=strides))
    res = res.reshape(shape[0] * shape[1], shape[2] * shape[3]).T
    return res