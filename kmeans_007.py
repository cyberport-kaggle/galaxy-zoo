"""
Trying some other estimators besides Ridge
"""

import gc
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from classes import logger
import classes
import models
from models.Base import CropScaleImageTransformer, ModelWrapper
from models.KMeansFeatures import KMeansFeatureGenerator


def get_images(crop=150, s=15):
    train_x_crop_scale = CropScaleImageTransformer(training=True,
                                                   result_path='data/data_train_crop_{}_scale_{}.npy'.format(crop, s),
                                                   crop_size=crop,
                                                   scaled_size=s,
                                                   n_jobs=-1,
                                                   memmap=True)
    images = train_x_crop_scale.transform()
    return images


def train_kmeans_generator(images, n_centroids=3000, n_patches=400000, rf_size=5):

    kmeans_generator = KMeansFeatureGenerator(n_centroids=n_centroids,
                                              rf_size=rf_size,
                                              result_path='data/mdl_kmeans_006_centroids_{}'.format(n_centroids),
                                              n_iterations=20,
                                              n_jobs=-1,)


    patch_extractor = models.KMeansFeatures.PatchSampler(n_patches=n_patches,
                                                         patch_size=rf_size,
                                                         n_jobs=-1)
    patches = patch_extractor.transform(images)
    kmeans_generator.fit(patches)
    return kmeans_generator


def gradient_boosting_grid_search():
    crop = 150
    s = 15
    n_centroids = 3000

    images = get_images(crop=crop, s=s)
    kmeans_generator = train_kmeans_generator(images, n_centroids=n_centroids)

    train_x = kmeans_generator.transform(images, save_to_file='data/data_kmeans_features_006_centroids_{}.npy'.format(n_centroids), memmap=True)
    train_y = classes.train_solutions.data

    # Unload some objects
    del images
    gc.collect()
    # We'll focus on the columns that have high errors based on the analysis in kmeans_006.py

    params = {
        'loss': ['ls', 'lad', 'huber', 'quantile'],
        'learning_rate': [0.01, 0.1, 1, 5, 10],
        'n_estimators': [100, 250, 500, 1000],
        'max_depth': [2, 3, 5, 10],
        'subsample': [0.2, 0.5, 1]
    }
    wrapper = ModelWrapper(GradientBoostingRegressor, {'verbose':1}, n_jobs=1)
    # not sure why it just dies here, on CV too
    #   File "/usr/lib/python2.7/multiprocessing/pool.py", line 319, in _handle_tasks
    # put(task)
    # SystemError: NULL result without error in PyObject_Call
    # seems like the parallelization is broken

    # Without parallelization, will run, but is super slow, probably because of the high dimensionality
    # of the train_x, which is n_centroids * 4 dimensions (12000), because of the pooling
    # It says something like 300 minutes to train 100 iterations
    wrapper.grid_search(train_x, train_y[:, 0], params, sample=0.3, refit=False)
