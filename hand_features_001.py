"""
Model using more hand generated features instead of raw pixels
"""
import os
from sklearn.base import BaseEstimator, TransformerMixin
from classes import ImageIteratorMixin, logger, RawImage
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
    def __init__(self, features, training=True, n_jobs=1, force_rerun=False, verbose=3, memmap=True, result_path=None):
        self.features = features
        self.training = training
        self.n_jobs = n_jobs
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

    def _transform(self, file_list):
        filepath = TRAIN_IMAGE_PATH if self.training else TEST_IMAGE_PATH
        res = []

        for i, f in enumerate(file_list):
            if i % 5000 == 0:
                logger.info("Processing image {} of {}".format(i, len(file_list)))
                img = RawImage(os.path.join(filepath, f))
                features = [func(img) for func in self.features]
                res.append(np.hstack(features))
        return res


def average_intensity_circle(r):
    """
    Generator for functions that calculate average intensity of pixels within radius r
    """
    pass


def average_intensity_ring(r, w):
    """
    Calculate average intensity within a ring of radius r and width w
    """
    pass
