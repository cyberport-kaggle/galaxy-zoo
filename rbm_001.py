from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MinMaxScaler
from classes import logger
import classes
import models
from models.Base import CropScaleImageTransformer, ModelWrapper


def rbm_001():
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

    patch_extractor = models.KMeansFeatures.PatchSampler(n_patches=n_patches,
                                                         patch_size=rf_size,
                                                         n_jobs=-1)
    images = train_x_crop_scale.transform()
    images = images.reshape((images.shape[0], 15 * 15 * 3))

    # rbm needs inputs to be between 0 and 1
    scaler = MinMaxScaler()
    images = scaler.fit_transform(images)

    # Training takes a long time, says 80 seconds per iteration, but seems like longer
    # And this is only with 256 components
    rbm = BernoulliRBM(verbose=1)
    rbm.fit(images)

    train_x = rbm.transform(images)
    train_y = classes.train_solutions.data

    # 0.138 CV on 50% of the dataset
    wrapper = ModelWrapper(models.Ridge.RidgeRFEstimator, {'alpha': 500, 'n_estimators': 500}, n_jobs=-1)
    wrapper.cross_validation(train_x, train_y, sample=0.5, parallel_estimator=True)