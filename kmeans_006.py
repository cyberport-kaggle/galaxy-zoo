import gc
from classes import logger
import classes
import models
from models.Base import CropScaleImageTransformer, ModelWrapper
from models.KMeansFeatures import KMeansFeatureGenerator


def kmeans_006():
    """
    Testing number of centroids

    [(1000, array([-0.10926318, -0.10853047])),
     (2000, array([-0.10727502, -0.10710292])),
     (2500, array([-0.107019  , -0.10696262])),
     (3000, array([-0.10713973, -0.1066932 ]))]

    """
    n_centroids_vals = [1000, 2000, 2500, 3000]
    scores = []

    for n_centroids in n_centroids_vals:
        s = 15
        crop = 150
        n_patches = 400000
        rf_size = 5
        logger.info("Training with n_centroids {}".format(n_centroids))

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
                                                  result_path='data/mdl_kmeans_006_centroids_{}'.format(n_centroids),
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

        train_x = kmeans_generator.transform(images, save_to_file='data/data_kmeans_features_006_centroids_{}.npy'.format(n_centroids), memmap=True)
        train_y = classes.train_solutions.data
        # Unload some objects
        del images
        gc.collect()

        wrapper = ModelWrapper(models.Ridge.RidgeRFEstimator, {'alpha': 500, 'n_estimators': 250}, n_jobs=-1)
        wrapper.cross_validation(train_x, train_y, n_folds=2, parallel_estimator=True)

        score = (n_centroids, wrapper.cv_scores)
        logger.info("Scores: {}".format(score))
        scores.append(score)

        del wrapper
        gc.collect()


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


def kmeans_006_submission():
    # Final submission
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

    wrapper = ModelWrapper(models.Ridge.RidgeRFEstimator, {'alpha': 500, 'n_estimators': 500}, n_jobs=-1)
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
    sub.to_file('sub_kmeans_006.csv')
