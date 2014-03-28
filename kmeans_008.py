"""
Tuning for max pooling and other factors
"""
from __future__ import division
import gc
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import classes
import models
from models.Base import CropScaleImageTransformer, ModelWrapper
from models.KMeansFeatures import KMeansFeatureGenerator, normalize, whiten
import numpy as np


def get_images(crop=150, s=15):
    train_x_crop_scale = CropScaleImageTransformer(training=True,
                                                   result_path='data/data_train_crop_{}_scale_{}.npy'.format(crop, s),
                                                   crop_size=crop,
                                                   scaled_size=s,
                                                   n_jobs=-1,
                                                   memmap=True)
    images = train_x_crop_scale.transform()
    return images


def train_kmeans_generator(images, n_centroids=3000, n_patches=400000, rf_size=5, pool_method='max'):

    # Can reuse the centroids
    kmeans_generator = KMeansFeatureGenerator(n_centroids=n_centroids,
                                              rf_size=rf_size,
                                              result_path='data/mdl_kmeans_006_centroids_{}'.format(n_centroids),
                                              n_iterations=20,
                                              n_jobs=-1,
                                              pool_method=pool_method)


    patch_extractor = models.KMeansFeatures.PatchSampler(n_patches=n_patches,
                                                         patch_size=rf_size,
                                                         n_jobs=-1)
    patches = patch_extractor.transform(images)
    kmeans_generator.fit(patches)
    return kmeans_generator


def max_pooling():
    # Seems to be a lot worse than the sum pooling
    # 2014-03-28 10:26:28 - Base - INFO - Cross validation completed in 1433.7291348.  Scores:
    # 2014-03-28 10:26:28 - Base - INFO - [-0.11968588 -0.12018345]
    crop = 150
    s = 15
    n_centroids = 3000

    images = get_images(crop=crop, s=s)
    kmeans_generator = train_kmeans_generator(images, n_centroids=n_centroids, pool_method='max')

    # Need something larger than the 15G RAM, since RAM usage seems to spike when recombining from parallel
    train_x = kmeans_generator.transform(images, save_to_file='data/data_kmeans_features_008_maxpool.npy', memmap=True)
    train_y = classes.train_solutions.data

    # Unload some objects
    del images
    gc.collect()

    wrapper = ModelWrapper(models.Ridge.RidgeRFEstimator, {'alpha': 500, 'n_estimators': 500}, n_jobs=-1)
    wrapper.cross_validation(train_x, train_y, sample=0.5, parallel_estimator=True)


def mean_pooling():
    # Wow mean pooling is really bad
    # 2014-03-28 11:28:42 - Base - INFO - Cross validation completed in 1523.09399891.  Scores:
    # 2014-03-28 11:28:42 - Base - INFO - [-0.13083991 -0.12989765]
    crop = 150
    s = 15
    n_centroids = 3000

    images = get_images(crop=crop, s=s)
    kmeans_generator = train_kmeans_generator(images, n_centroids=n_centroids, pool_method='mean')

    train_x = kmeans_generator.transform(images, save_to_file='data/data_kmeans_features_008_meanpool.npy', memmap=True)
    train_y = classes.train_solutions.data

    # Unload some objects
    del images
    gc.collect()

    wrapper = ModelWrapper(models.Ridge.RidgeRFEstimator, {'alpha': 500, 'n_estimators': 500}, n_jobs=-1)
    wrapper.cross_validation(train_x, train_y, sample=0.5, parallel_estimator=True)


def rf_size_10():
    # Pretty bad as well
    # 2014-03-28 13:04:07 - Base - INFO - Cross validation completed in 1475.74401999.  Scores:
    # 2014-03-28 13:04:07 - Base - INFO - [-0.12217214 -0.12209735]

    n_centroids = 3000
    s = 15
    crop = 150
    n_patches = 400000
    rf_size = 5

    train_x_crop_scale = CropScaleImageTransformer(training=True,
                                                   result_path='data/data_train_crop_{}_scale_{}.npy'.format(crop, s),
                                                   crop_size=crop,
                                                   scaled_size=s,
                                                   n_jobs=-1,
                                                   memmap=True)
    test_x_crop_scale = CropScaleImageTransformer(training=False,
                                                  result_path='data/data_test_crop_{}_scale_{}.npy'.format(crop, s),
                                                  crop_size=crop,
                                                  scaled_size=s,
                                                  n_jobs=-1,
                                                  memmap=True)

    kmeans_generator = KMeansFeatureGenerator(n_centroids=n_centroids,
                                              rf_size=rf_size,
                                              result_path='data/mdl_kmeans_008_rf10'.format(n_centroids),
                                              n_iterations=20,
                                              n_jobs=-1,)

    patch_extractor = models.KMeansFeatures.PatchSampler(n_patches=n_patches,
                                                         patch_size=rf_size,
                                                         n_jobs=-1)
    images = train_x_crop_scale.transform()

    patches = patch_extractor.transform(images)

    kmeans_generator.fit(patches)

    del patches
    gc.collect()

    train_x = kmeans_generator.transform(images, save_to_file='data/data_kmeans_features_008_rf10.npy'.format(n_centroids), memmap=True)
    train_y = classes.train_solutions.data

    del images
    gc.collect()

    wrapper = ModelWrapper(models.Ridge.RidgeRFEstimator, {'alpha': 500, 'n_estimators': 500}, n_jobs=-1)
    wrapper.cross_validation(train_x, train_y, sample=0.5, parallel_estimator=True)


def extratress():
    # 2014-03-28 13:24:22 - Base - INFO - Cross validation completed in 1139.1731801.  Scores:
    # 2014-03-28 13:24:22 - Base - INFO - [-0.11048638 -0.11060714]

    crop = 150
    s = 15
    n_centroids = 3000

    images = get_images(crop=crop, s=s)
    kmeans_generator = train_kmeans_generator(images, n_centroids=n_centroids, pool_method='sum')

    # Need something larger than the 15G RAM, since RAM usage seems to spike when recombining from parallel
    train_x = kmeans_generator.transform(images, save_to_file='data/data_kmeans_features_006_centroids_{}.npy'.format(n_centroids), memmap=True)
    train_y = classes.train_solutions.data

    # Unload some objects
    del images
    gc.collect()

    wrapper = ModelWrapper(models.Ridge.RidgeExtraTreesEstimator, {'alpha': 500, 'n_estimators': 500}, n_jobs=-1)
    wrapper.cross_validation(train_x, train_y, sample=0.5, parallel_estimator=True)


def extra_trees_submission():
    # Somehow the submission on the leaderboard scores 0.22
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

    wrapper = ModelWrapper(models.Ridge.RidgeExtraTreesEstimator, {'alpha': 500, 'n_estimators': 500}, n_jobs=-1)
    # wrapper.cross_validation(train_x, train_y, sample=0.5, parallel_estimator=True)
    wrapper.fit(train_x, train_y)

    test_x_crop_scale = CropScaleImageTransformer(training=False,
                                                  result_path='data/data_test_crop_{}_scale_{}.npy'.format(crop, s),
                                                  crop_size=crop,
                                                  scaled_size=s,
                                                  n_jobs=-1,
                                                  memmap=True)

    test_images = test_x_crop_scale.transform()
    test_x = kmeans_generator.transform(test_images, save_to_file='data/data_test_kmeans_features_006_centroids_{}.npy'.format(n_centroids), memmap=True)
    res = wrapper.predict(test_x)
    sub = classes.Submission(res)
    sub.to_file('sub_kmeans_008.csv')


def ensemble():
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

    wrapper1 = ModelWrapper(models.Ridge.RidgeExtraTreesEstimator, {'alpha': 500, 'n_estimators': 500}, n_jobs=-1)
    wrapper1.fit(train_x, train_y)

    wrapper2 = ModelWrapper(models.Ridge.RidgeRFEstimator, {'alpha': 500, 'n_estimators': 500}, n_jobs=-1)
    wrapper2.fit(train_x, train_y)

    pred1 = wrapper1.predict(train_x)
    pred2 = wrapper2.predict(train_x)
    wrapper3 = ModelWrapper(Ridge)
    wrapper3.cross_validation(np.vstack((pred1, pred2)), train_y)



def second_layer_kmeans():
    crop = 150
    s = 15
    n_centroids = 3000

    images = get_images(crop=crop, s=s)
    kmeans_generator = train_kmeans_generator(images, n_centroids=n_centroids)

    # n_images, 12000
    x_l1 = kmeans_generator.transform(images, save_to_file='data/data_kmeans_features_006_centroids_{}.npy'.format(n_centroids), memmap=False)
    # train_y = classes.train_solutions.data

    # Normalize each column
    scaler = StandardScaler()
    x_l1 = scaler.fit_transform(x_l1)

    # Whiten
    # This is really really slow
    cov = np.cov(x_l1, rowvar=0)
    mean = x_l1.mean(0, keepdims=True)
    d, v = np.linalg.eig(cov)
    p = np.dot(v,
               np.dot(np.diag(np.sqrt(1 / (d + 0.1))),
                      v.T))
    res = np.dot(x_l1 - mean, p)


# This still takes a long time, can parallelize, but still will take a long time.  Also, not sure that I'm
# doing this correctly, since the paper says that the formulas need to be expanded and the values can be
# accumulated just once
def get_energies(X):
    energies = np.zeros((X.shape[1], X.shape[1]))
    for k in xrange(X.shape[1]):
        print k
        for j in xrange(k+1, X.shape[1]):
            xk = X[:, k]
            xj = X[:, j]
            corr = np.corrcoef(xk, xj)[0, 1]
            beta = (1 - corr) ** (-1/2)
            gamma = (1 + corr) ** (-1/2)
            xjhat = 0.5 * ((gamma + beta) * xj) + ((gamma - beta) * xk)
            xkhat = 0.5 * ((gamma - beta) * xj) + ((gamma + beta) * xk)
            denom = np.sqrt(np.sum((xjhat ** 4) - 1) * np.sum((xkhat ** 4) - 1))
            energy = (np.sum(np.multiply(xkhat ** 2, xjhat **2)) - 1) / denom
            energies[k, j] = energy
            energies[j, k] = energy
