from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MinMaxScaler
from classes import logger
import models
from models.Base import CropScaleImageTransformer


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

    patches = patch_extractor.transform(images)

    # rbm needs inputs to be between 0 and 1
    scaler = MinMaxScaler()
    patches = scaler.fit_transform(patches)

    rbm = BernoulliRBM(verbose=1)
    rbm.fit(patches)
