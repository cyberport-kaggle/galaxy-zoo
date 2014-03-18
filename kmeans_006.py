import gc
from sklearn.cross_validation import train_test_split
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


def kmeans_006_colwise_rmse():
    crop = 150
    s = 15
    n_centroids = 3000

    images = get_images(crop=crop, s=s)
    kmeans_generator = train_kmeans_generator(images, n_centroids=n_centroids)

    train_x = kmeans_generator.transform(images, save_to_file='data/data_kmeans_features_006_centroids_{}.npy'.format(n_centroids), memmap=True)
    train_y = classes.train_solutions.data

    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, train_size=0.2, test_size=0.2)

    # Unload some objects
    del images
    gc.collect()

    wrapper = ModelWrapper(models.Ridge.RidgeRFEstimator, {'alpha': 500, 'n_estimators': 500, 'verbose': 1}, n_jobs=-1)
    wrapper.fit(train_x, train_y)
    # About 11 minutes to train the ridge regression on an m2.4 xlarge with 50% of the train set
    # Took about a minute to train ridge on 0.1 of the train set, but the overall rmse was .114 compared to .106 on 50%, and .104 actual
    # 5 minutes to train ridge on 0.2 of the train set, with rmse of .111
    kmeans_preds = wrapper.predict(test_x)

    logger.info('Kmeans')
    colwise = classes.colwise_rmse(kmeans_preds, test_y)
    overall = classes.rmse(kmeans_preds, test_y)

    """
    Columnwise rmse on 50% of the train set
    2014-03-18 05:44:39 - classes - INFO - Column-wise RMSEs:
    2014-03-18 05:44:39 - classes - INFO - Col 0: 0.158280
    2014-03-18 05:44:39 - classes - INFO - Col 1: 0.164700
    2014-03-18 05:44:39 - classes - INFO - Col 2: 0.034519
    2014-03-18 05:44:39 - classes - INFO - Col 3: 0.093202
    2014-03-18 05:44:39 - classes - INFO - Col 4: 0.169289
    2014-03-18 05:44:39 - classes - INFO - Col 5: 0.127860
    2014-03-18 05:44:39 - classes - INFO - Col 6: 0.170405
    2014-03-18 05:44:39 - classes - INFO - Col 7: 0.182594
    2014-03-18 05:44:39 - classes - INFO - Col 8: 0.157834
    2014-03-18 05:44:39 - classes - INFO - Col 9: 0.067742
    2014-03-18 05:44:39 - classes - INFO - Col 10: 0.124075
    2014-03-18 05:44:39 - classes - INFO - Col 11: 0.129824
    2014-03-18 05:44:39 - classes - INFO - Col 12: 0.056982
    2014-03-18 05:44:39 - classes - INFO - Col 13: 0.161206
    2014-03-18 05:44:39 - classes - INFO - Col 14: 0.161206
    2014-03-18 05:44:39 - classes - INFO - Col 15: 0.112903
    2014-03-18 05:44:39 - classes - INFO - Col 16: 0.129670
    2014-03-18 05:44:39 - classes - INFO - Col 17: 0.062162
    2014-03-18 05:44:39 - classes - INFO - Col 18: 0.079066
    2014-03-18 05:44:39 - classes - INFO - Col 19: 0.027080
    2014-03-18 05:44:39 - classes - INFO - Col 20: 0.048757
    2014-03-18 05:44:39 - classes - INFO - Col 21: 0.063673
    2014-03-18 05:44:39 - classes - INFO - Col 22: 0.071035
    2014-03-18 05:44:39 - classes - INFO - Col 23: 0.079968
    2014-03-18 05:44:39 - classes - INFO - Col 24: 0.023673
    2014-03-18 05:44:39 - classes - INFO - Col 25: 0.081469
    2014-03-18 05:44:39 - classes - INFO - Col 26: 0.029800
    2014-03-18 05:44:39 - classes - INFO - Col 27: 0.053743
    2014-03-18 05:44:39 - classes - INFO - Col 28: 0.107447
    2014-03-18 05:44:39 - classes - INFO - Col 29: 0.100361
    2014-03-18 05:44:39 - classes - INFO - Col 30: 0.082481
    2014-03-18 05:44:39 - classes - INFO - Col 31: 0.050674
    2014-03-18 05:44:39 - classes - INFO - Col 32: 0.147045
    2014-03-18 05:44:39 - classes - INFO - Col 33: 0.059929
    2014-03-18 05:44:39 - classes - INFO - Col 34: 0.034570
    2014-03-18 05:44:39 - classes - INFO - Col 35: 0.032258
    2014-03-18 05:44:39 - classes - INFO - Col 36: 0.090012
    2014-03-18 05:44:39 - classes - INFO - RMSE: 0.106822640675
    """


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
