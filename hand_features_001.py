"""
Model using more hand generated features instead of raw pixels
"""
from __future__ import division
import os
import itertools
import joblib
from joblib import Parallel
import models
from skimage.morphology import disk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib import delayed
from classes import ImageIteratorMixin, logger, RawImage, train_solutions, chunks
import classes
from constants import TRAIN_IMAGE_PATH, TEST_IMAGE_PATH
import numpy as np


class ImageFeatureExtractor(ImageIteratorMixin, BaseEstimator, TransformerMixin):
    """
    Transformer that loops over every image, and generates features according
    to the functions that are passed in when the transformer is instantiated.

    Arguments:
    ==========
    features: array of functions
        Array of functions that should be applied to each image.  Each function
        should expect to receive a RawImage object.  Each function should return
        an ndarray of (1, n_features)

    Returns:
    ========
    ndarray of n_images, n_features, where n_images depends on whether
    it is the train or test set, and n_features depends on each feature generator
    function that is provided when instantiated.

    """
    def __init__(self, features, training=True, force_rerun=False, verbose=3, memmap=True, result_path=None):
        self.features = list(features)
        self.training = training
        self.n_jobs = 1  # Can't pickle the closures
        self.force_rerun = force_rerun
        self.verbose = verbose
        self.memmap = memmap
        self.result_path = result_path or self._get_result_path()

    def _get_result_path(self):
        if self.training:
            return 'data/img_train_hand_features_001.npy'
        else:
            return 'data/img_test_hand_features_001.npy'

    def fit(self, X=None, y=None):
        return self

    def _transform(training, features, file_list):
        filepath = TRAIN_IMAGE_PATH if training else TEST_IMAGE_PATH
        res = []

        for i, f in enumerate(file_list):
            if i % 5000 == 0:
                logger.info("Processing image {} of {}".format(i, len(file_list)))
            img = RawImage(os.path.join(filepath, f))
            features = [func(img) for func in features]
            res.append(np.hstack(features))
        return res


def average_intensity_circle(r):
    """
    TODO: Implement as object with __call__ method implemented, so that it can be pickled
    Generator for functions that calculate average intensity of pixels within radius r
    """
    mask = disk(r, np.bool)
    def func(image):
        offset = int((image.data.shape[0] - mask.shape[0]) / 2)
        offset_end = offset + mask.shape[0]
        res = image.data[offset:offset_end, offset:offset_end]
        if len(res.shape) == 2:
            # grayscale
            return np.array([res[mask].mean()])
        elif len(res.shape) == 3:
            # color
            return res[mask].mean(0)
    return func


def average_intensity_ring(r, w):
    """
    Calculate average intensity within a ring of radius r and width w
    """
    inner_mask = disk(r, np.bool)
    outer_mask = disk(r + w, np.bool)
    mask = np.copy(outer_mask)
    mask[w:w+inner_mask.shape[0], w:w+inner_mask.shape[0]] = -inner_mask
    def func(image):
        offset = int((image.data.shape[0] - mask.shape[0]) / 2)
        offset_end = offset + mask.shape[0]
        res = image.data[offset:offset_end, offset:offset_end]
        if len(res.shape) == 2:
            # grayscale
            return np.array([res[mask].mean()])
        elif len(res.shape) == 3:
            # color
            return res[mask].mean(0)

    return func


def hand_features_001():
    extractor = ImageFeatureExtractor(itertools.chain(
        [average_intensity_circle(r) for r in range(5, 26, 5)],
        [average_intensity_circle(50), average_intensity_circle(100)],
        [average_intensity_ring(r, 3) for r in range(5, 26, 5)],
        [average_intensity_ring(50, 3), average_intensity_ring(100, 3)],
    ), force_rerun=True, n_jobs=5, verbose=3)

    train_x = extractor.transform()
    train_y = classes.train_solutions.data

    mdl = models.Base.ModelWrapper(models.Ridge.RidgeRFEstimator, {
        'alpha': 14,
        'n_estimators': 200,
        'verbose': 3,
        'oob_score': True
    }, n_jobs=-1)

    # .129 RMSE, really not much better than the pixel sampling
    mdl.cross_validation(train_x, train_y, sample=0.5, n_folds=2)
