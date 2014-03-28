"""
Tuning for max pooling and other factors
"""
import gc
import classes
import models
from models.Base import CropScaleImageTransformer, ModelWrapper
from models.KMeansFeatures import KMeansFeatureGenerator, normalize, whiten


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
    kmeans_generator = train_kmeans_generator(images, n_centroids=n_centroids)

    # Need something larger than the 15G RAM, since RAM usage seems to spike when recombining from parallel
    train_x = kmeans_generator.transform(images, save_to_file='data/data_kmeans_features_008_maxpool.npy', memmap=True)
    train_y = classes.train_solutions.data

    # Unload some objects
    del images
    gc.collect()

    wrapper = ModelWrapper(models.Ridge.RidgeRFEstimator, {'alpha': 500, 'n_estimators': 500}, n_jobs=-1)
    wrapper.cross_validation(train_x, train_y, sample=0.5, parallel_estimator=True)


def mean_pooling():
    crop = 150
    s = 15
    n_centroids = 3000

    images = get_images(crop=crop, s=s)
    kmeans_generator = train_kmeans_generator(images, n_centroids=n_centroids, pool_method='mean')

    # Need something larger than the 15G RAM, since RAM usage seems to spike when recombining from parallel
    train_x = kmeans_generator.transform(images, save_to_file='data/data_kmeans_features_008_meanpool.npy', memmap=True)
    train_y = classes.train_solutions.data

    # Unload some objects
    del images
    gc.collect()

    wrapper = ModelWrapper(models.Ridge.RidgeRFEstimator, {'alpha': 500, 'n_estimators': 500}, n_jobs=-1)
    wrapper.cross_validation(train_x, train_y, sample=0.5, parallel_estimator=True)


def second_layer_kmeans():
    crop = 150
    s = 15
    n_centroids = 3000

    images = get_images(crop=crop, s=s)
    kmeans_generator = train_kmeans_generator(images, n_centroids=n_centroids)

    # n_images, 12000
    x_l1 = kmeans_generator.transform(images, save_to_file='data/data_kmeans_features_006_centroids_{}.npy'.format(n_centroids), memmap=True)
    # train_y = classes.train_solutions.data
