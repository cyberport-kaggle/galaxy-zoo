from __future__ import division
import multiprocessing
import itertools
import math
from joblib import Parallel, delayed
from matplotlib import pyplot
import numpy as np
from numpy.lib.stride_tricks import as_strided
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
    # train_mmap is of dimensions (n_training, image_rows, image_cols, channels)
    n_images = train_mmap.shape[0]
    image_rows = train_mmap.shape[1]
    image_cols = train_mmap.shape[2]

    for i, p in enumerate(patch_nums):
        # Randomly get an offset
        row = np.random.randint(image_rows - patch_size + 1)
        col = np.random.randint(image_cols - patch_size + 1)

        # Pick the right image and extract the patch
        img = train_mmap[p % n_images]
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

        logger.info("Extracting patches in {} jobs, chunk sizes: {}".format(cores, len(patch_chunks[0])))
        res = Parallel(n_jobs=cores, verbose=3)(delayed(chunked_extract_patch)(x, self.trainX, self.rf_size) for x in patch_chunks)
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
        # In practice, the algorithm converges pretty quickly.  15-20 should be enough iterations
        self.centroids = spherical_kmeans(self.patches, self.num_centroids, 20)

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
        logger.info("Transforming in {} jobs, chunk sizes: {}".format(cores, len(chunks[0])))

        res = Parallel(n_jobs=cores, verbose=3)(
            delayed(chunked_extract_features)(i, x, self.rf_size, self.centroids, self.mean, self.p, self.whitening) for i in chunks
        )
        res = np.vstack(res)

        return res

    def save_to_file(self, file_base):
        """
        Saves the patch size, centroids, mean, and p to files for reloading.  Assumes whitening is always true
        """
        file_path = './data/' + file_base
        centroids_path = file_path + '_centroids.npy'
        logger.info('Saving centroids to {}'.format(centroids_path))
        np.save(centroids_path, self.centroids)

        means_path = file_path + '_means.npy'
        logger.info("Saving means to {}".format(means_path))
        np.save(means_path, self.mean)

        p_path = file_path + '_p.npy'
        logger.info("Saving p to {}".format(p_path))
        np.save(p_path, self.p)

    @classmethod
    def load_from_file(cls, file_base, rf_size=6):
        """
        loads in patch size, centroids, mean, and p
        """
        instance = cls(rf_size=rf_size, whitening=True)
        file_path = './data/' + file_base
        centroids_path = file_path + '_centroids.npy'
        logger.info('Loading centroids from {}'.format(centroids_path))
        instance.centroids = np.load(centroids_path)

        means_path = file_path + '_means.npy'
        logger.info("Loading means from {}".format(means_path))
        instance.mean = np.load(means_path)

        p_path = file_path + '_p.npy'
        logger.info("Loading p from {}".format(p_path))
        instance.p = np.load(p_path)
        return instance


def show_centroids(centroids, centroid_size, reshape=(3, 6, 6), swap_axis=None, normalize=True):
    """
    Shows centroids.  Expects centroids to be a (n_centroids, n_pixels) array.
    Reshape controls how each centroid is reshaped.  if reshape is none, then no reshaping is done

    General strategy is to calculate what size x by x grid we want, reshape each centroid, place in grid, then show the whole image
    """
    n_centroids = centroids.shape[0]
    # Infer the number of channels in the centroid
    channels = centroids.shape[1] / (centroid_size ** 2)
    cols = int(math.sqrt(n_centroids))
    rows = math.ceil(n_centroids / cols)
    # Add 1 pixel for padding
    image_size = (centroid_size + 1)
    images = np.ones((image_size * rows, image_size * cols, channels), dtype=np.uint8)
    for i in xrange(n_centroids):
        this_image = centroids[i]
        if reshape:
            this_image = this_image.reshape(reshape)
        if swap_axis:
            this_image = this_image.swapaxes(*swap_axis)
        offset_col = image_size * (i % cols)
        offset_col_end = offset_col + image_size - 1
        offset_row = image_size * (math.floor(i / cols))
        offset_row_end = offset_row + image_size - 1
        images[offset_row:offset_row_end, offset_col:offset_col_end] = this_image

    if normalize:
        # Normalize it all
        min = -1.5
        max = 1.5
        images = (images - min) / (max - min)
    pyplot.imshow(images)
    pyplot.show()


def normalize(x):
    """
    Normalizes each patch by subtracting mean and dividing by variance
    """
    temp1 = x - x.mean(1, keepdims=True)
    temp2 = np.sqrt(x.var(1, keepdims=True) + 10)

    return temp1 / temp2


def chunked_extract_features(idx, X, rf_size, centroids, mean, p, whitening=True):
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
    idx = [y for y in idx if y is not None]
    res = [None] * len(idx)
    for i, img_idx in enumerate(idx):
        if (i + 1) % 500 == 0:
            logger.info("Extracting features on image {} / {}".format(i + 1, len(idx)))

        this_x = X[img_idx]
        patches = np.vstack((rolling_block(this_x[:, :, 0], rf_size),
                             rolling_block(this_x[:, :, 1], rf_size),
                             rolling_block(this_x[:, :, 2], rf_size))).T

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
        prows = this_x.shape[0] - rf_size + 1
        pcols = this_x.shape[0] - rf_size + 1
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


def unique_rows(data):
    """
    Returns the number of unique rows in a 2D NumPy array.
    Using it to check the number of duplicated clusters in K-Means learning
    @param data:
    @return:
    """
    data_set = set([tuple(row) for row in data])
    return len(data_set)


def spherical_kmeans(X, k, n_iter, batch_size=1000):
    """
    Do a spherical k-means.  Line by line port of Coates' matlab code.

    Returns a (k, n_pixels) centroids matrix
    """

    # shape (k, 1)
    x2 = np.sum(X**2, 1, keepdims=True)

    # randomly initialize centroids
    centroids = np.random.randn(k, X.shape[1]) * 0.1

    for iteration in xrange(1, n_iter + 1):
        # shape (k, 1)
        c2 = 0.5 * np.sum(centroids ** 2, 1, keepdims=True)

        # shape (k, n_pixels)
        summation = np.zeros((k, X.shape[1]))
        counts = np.zeros((k, 1))
        loss = 0

        for i in xrange(0, X.shape[0], batch_size):
            last_index = min(i + batch_size, X.shape[0])
            m = last_index - i

            # shape (k, batch_size) - shape (k, 1)
            tmp = np.dot(centroids, X[i:last_index, :].T) - c2
            # shape (batch_size, )
            indices = np.argmax(tmp, 0)
            # shape (1, batch_size)
            val = np.max(tmp, 0, keepdims=True)

            loss += np.sum((0.5 * x2[i:last_index]) - val.T)

            # Don't use a sparse matrix here
            S = np.zeros((batch_size, k))
            S[range(batch_size), indices] = 1

            # shape (k, n_pixels)
            this_sum = np.dot(S.T, X[i:last_index, :])
            summation += this_sum

            this_counts = np.sum(S, 0, keepdims=True).T
            counts += this_counts

        # Sometimes raises RuntimeWarnings because some counts can be 0
        centroids = summation / counts

        bad_indices = np.where(counts == 0)[0]
        centroids[bad_indices, :] = 0

        assert not np.any(np.isnan(centroids))

        logger.info("K-means iteration {} of {}, loss {}".format(iteration, n_iter, loss))
    return centroids


def parallel_spherical_kmeans(X, k, n_iter, batch_size=1000):
    """
    spherical kmeans in parallel.  Unfortunately not that much faster.  Non-parallel takes about
    a minute per iteration on my machine, and this takes 40 seconds per iteration.

    Probably because of the overhead involved in threading things out.
    """

    # shape (k, 1)
    x2 = np.sum(X**2, 1, keepdims=True)

    # randomly initialize centroids
    centroids = np.random.randn(k, X.shape[1]) * 0.1

    for iteration in xrange(1, n_iter + 1):
        # shape (k, 1)
        c2 = 0.5 * np.sum(centroids ** 2, 1, keepdims=True)

        # shape (k, n_pixels)
        summation = np.zeros((k, X.shape[1]))
        counts = np.zeros((k, 1))
        loss = 0

        # Parallelizes by splitting X's rows into groups
        # Each subprocess gets a group, and traverses that group in batches
        cores = multiprocessing.cpu_count()
        chunk_size = int(math.floor(X.shape[0] / cores))
        ranges = [(0 + (i * chunk_size), 0 + ((i + 1) * chunk_size)) for i in range(cores)]
        # Any remainders get attached to the last chunk
        ranges[-1] = (ranges[-1][0], X.shape[0])

        logger.info("Chunked with offsets {} and chunk size {}".format(ranges, chunk_size))
        # Gets back an array of (summation, counts, loss) tuples
        res = Parallel(n_jobs=cores, verbose=3)(
            delayed(_process_batches)(X, start, end, batch_size, centroids, c2, x2, k) for start, end in ranges
        )

        for this_sum, this_count, this_loss in res:
            summation += this_sum
            counts += this_count
            loss += this_loss

        # Sometimes raises RuntimeWarnings because some counts can be 0
        centroids = summation / counts

        bad_indices = np.where(counts == 0)[0]
        centroids[bad_indices, :] = 0

        assert not np.any(np.isnan(centroids))

        logger.info("K-means iteration {} of {}, loss {}".format(iteration, n_iter, loss))
    return centroids


def _process_batches(X, start, end, batch_size, centroids, c2, x2, k):
    loss = 0
    summation = np.zeros((k, X.shape[1]))
    counts = np.zeros((k, 1))

    for i in xrange(start, end, batch_size):
        last_index = min(i + batch_size, end)
        m = last_index - i

        # shape (k, batch_size) - shape (k, 1)
        tmp = np.dot(centroids, X[i:last_index, :].T) - c2
        # shape (batch_size, )
        indices = np.argmax(tmp, 0)
        # shape (1, batch_size)
        val = np.max(tmp, 0, keepdims=True)

        loss += np.sum((0.5 * x2[i:last_index]) - val.T)

        # Don't use a sparse matrix here
        S = np.zeros((batch_size, k))
        S[range(batch_size), indices] = 1

        # shape (k, n_pixels)
        this_sum = np.dot(S.T, X[i:last_index, :])
        summation += this_sum

        this_counts = np.sum(S, 0, keepdims=True).T
        counts += this_counts

    return summation, counts, loss


"""
from cifar import *
import ipdb

data = [get_batch(i)['data'] for i in range(1, 6)]
data = np.vstack(data)
rdata = data.reshape((50000, 3, 32, 32))
rdata = rdata.swapaxes(1, 3)
rdata = rdata.swapaxes(1, 2)

mdl = models.KMeansFeatures.KMeansFeatures(rf_size=6, num_centroids=1600, num_patches=400000)
mdl.trainX = rdata
mdl.extract_patches()
mdl.patches = models.KMeansFeatures.normalize(mdl.patches)
mdl.whiten()

# ipdb.run('models.KMeansFeatures.spherical_kmeans(mdl.patches, 1600, 50)')

centroids = models.KMeansFeatures.spherical_kmeans(mdl.patches, 1600, 10)

K-means iteration 1 / 50, loss 7929682.156621
K-means iteration 2 / 50, loss 6360819.903194
K-means iteration 3 / 50, loss 5845735.334416
K-means iteration 4 / 50, loss 5662207.157659
K-means iteration 5 / 50, loss 5576102.374176
K-means iteration 6 / 50, loss 5527709.879265
K-means iteration 7 / 50, loss 5497342.736403
K-means iteration 8 / 50, loss 5476660.157031
K-means iteration 9 / 50, loss 5461946.867995
K-means iteration 10 / 50, loss 5450773.459769
K-means iteration 11 / 50, loss 5441964.667322
K-means iteration 12 / 50, loss 5434968.280210
K-means iteration 13 / 50, loss 5429328.339294
K-means iteration 14 / 50, loss 5424500.268900
K-means iteration 15 / 50, loss 5420282.964752
K-means iteration 16 / 50, loss 5416694.390692
K-means iteration 17 / 50, loss 5413672.757183
K-means iteration 18 / 50, loss 5411071.651387
K-means iteration 19 / 50, loss 5408777.344397
K-means iteration 20 / 50, loss 5406771.207412
K-means iteration 21 / 50, loss 5404987.410941
K-means iteration 22 / 50, loss 5403393.323705
K-means iteration 23 / 50, loss 5401991.038828
K-means iteration 24 / 50, loss 5400678.657783
K-means iteration 25 / 50, loss 5399466.917830
K-means iteration 26 / 50, loss 5398354.264769
K-means iteration 27 / 50, loss 5397318.404619
K-means iteration 28 / 50, loss 5396398.170130
K-means iteration 29 / 50, loss 5395568.792654
K-means iteration 30 / 50, loss 5394815.573593
K-means iteration 31 / 50, loss 5394129.751249
K-means iteration 32 / 50, loss 5393487.839848
K-means iteration 33 / 50, loss 5392907.159336
K-means iteration 34 / 50, loss 5392365.621484
K-means iteration 35 / 50, loss 5391848.820510
K-means iteration 36 / 50, loss 5391386.130618
K-means iteration 37 / 50, loss 5390965.156332
K-means iteration 38 / 50, loss 5390565.283935
K-means iteration 39 / 50, loss 5390192.369274
K-means iteration 40 / 50, loss 5389848.162731
K-means iteration 41 / 50, loss 5389510.443807
K-means iteration 42 / 50, loss 5389206.547821
K-means iteration 43 / 50, loss 5388940.929436
K-means iteration 44 / 50, loss 5388683.464927
K-means iteration 45 / 50, loss 5388430.057730
K-means iteration 46 / 50, loss 5388187.639730
K-means iteration 47 / 50, loss 5387962.393151
K-means iteration 48 / 50, loss 5387738.482807
K-means iteration 49 / 50, loss 5387525.722404
K-means iteration 50 / 50, loss 5387330.119329

"""
