import gc
import classes
import models
from models.Base import ModelWrapper, CropScaleImageTransformer
from models.KMeansFeatures import KMeansFeatureGenerator


def get_images(crop=150, s=15):
    """
    Iterates over each image file, cropping it to 150 pixels, then scaling it to 15 pixels.

    Returns an ndarray (possibly memmapped) of size (n_images, 15, 15, 3)
    """
    train_x_crop_scale = CropScaleImageTransformer(training=True,
                                                   result_path='data/data_train_crop_{}_scale_{}.npy'.format(crop, s),
                                                   crop_size=crop,
                                                   scaled_size=s,
                                                   n_jobs=-1,
                                                   memmap=True)
    images = train_x_crop_scale.transform()
    return images


def train_kmeans_generator(images, n_centroids=3000, n_patches=400000, rf_size=5):
    """
    Takes the image ndarray and extracts patches, then trains the kmeans feature generator
    with those patches.

    Patches are taken by iterating over images sequentially, and randomly selecting a patch within each image.
    For example, if you have 1000 images, the 1st and the 1001st patch will both be from the first image.

    The feature generator applies normalization and ZCA whitening before using spherical k-means to find the
    centroids.
    """
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


crop = 150
s = 15
n_centroids = 3000

# Get the cropped and scaled images
images = get_images(crop=crop, s=s)

# Get the trained feature generator
kmeans_generator = train_kmeans_generator(images, n_centroids=n_centroids)

# Transform the training images into the features.
# Since we have 3,000 centroids with pooling over quadrants, we'll get 12,000 (3000 * 4) features
train_x = kmeans_generator.transform(images, save_to_file='data/data_kmeans_features_006_centroids_{}.npy'.format(n_centroids), memmap=True)
train_y = classes.train_solutions.data

# Unload some objects for memory
del images
gc.collect()

# ModelWrapper is a convenience class that we created to automate some of the typical tasks
# like logging, grid search and cross validation.
# For fit, it is basically equivalent to calling fit on the estimator

# The estimator takes the X and y and trains a ridge regression (sklearn.linear_model.Ridge),
# predicts using the ridge regressor, then uses the results of the prediction to train a random forest.
wrapper = ModelWrapper(models.Ridge.RidgeRFEstimator, {'alpha': 500, 'n_estimators': 500}, n_jobs=-1)
wrapper.fit(train_x, train_y)

test_x_crop_scale = CropScaleImageTransformer(training=False,
                                              result_path='data/data_test_crop_{}_scale_{}.npy'.format(crop, s),
                                              crop_size=crop,
                                              scaled_size=s,
                                              n_jobs=-1,
                                              memmap=True)

# Crop and scale the test images
test_images = test_x_crop_scale.transform()

# Generate the test features
test_x = kmeans_generator.transform(test_images, save_to_file='data/data_test_kmeans_features_006_centroids_{}.npy'.format(n_centroids), memmap=True)

# Predict on the test features
res = wrapper.predict(test_x)

# Generate a submission file
sub = classes.Submission(res)
sub.to_file('sub_kmeans_006.csv')