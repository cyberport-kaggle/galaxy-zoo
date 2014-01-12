"""
Run scripts for individual models in Galaxy Zoo
"""
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import classes
import numpy as np
import logging
from constants import *
from sklearn.cluster import KMeans
from IPython import embed
from sklearn.decomposition import RandomizedPCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

logger = logging.getLogger('galaxy')


def train_set_average_benchmark(outfile="sub_average_benchmark_000.csv"):
    """
    What should be the actual baseline.  Takes the training set solutions, averages them, and uses that as the
    submission for every row in the test set
    """
    start_time = time.clock()
    training_data = classes.get_training_data()[:, 1:]

    solutions = np.mean(training_data, axis=0)

    # Calculate an RMSE
    train_solution = np.tile(solutions, (N_TRAIN, 1))
    rmse = classes.rmse(train_solution, training_data)

    solution = classes.Submission(np.tile(solutions, (N_TEST, 1)))
    solution.to_file(outfile)

    end_time = time.clock()
    logger.info("Model completed in {}".format(end_time - start_time))


class CentralPixelBenchmark(classes.BaseModel):
    @staticmethod
    def process_image(img):
        return img.central_pixel.copy()

    def build_features(self, files, training=True):
        return self.do_for_each_image(files, self.process_image, 3, training)

    @classes.cache_to_file('data/data_central_pixel_001.csv', '%i')
    def build_train_predictors(self):
        # Build the training data - load all of the images and extract the RGB values of the central pixel
        # Resulting training dataset should be (70948, 3)
        logger.info("Building predictors")
        file_list = classes.get_training_filenames(self.train_y)
        return self.build_features(file_list, True)

    def fit_estimator(self):
        # Fit a k-means clustering estimator
        # We use 37 centers initially because there are 37 classes
        # Seems like the sample submission used 6 clusters
        start_time = time.clock()
        logger.info("Fitting kmeans estimator")
        self.estimator = KMeans(init='k-means++', n_clusters=37)
        self.estimator.fit(self.predictors)
        logger.info("Finished fitting model in {}".format(time.clock() - start_time))

    def get_cluster_averages(self):
        # Get the average response for each cluster in the training set
        # This is a 37 x 37 array, one row for each cluster, and one column for each class
        logger.info("Calculating cluster averages")
        average_responses = np.zeros((37, 37))
        for cluster in range(37):
            idx = self.estimator.labels_ == cluster
            responses = self.train_y[idx, 1:]
            average_responses[cluster] = responses.mean(axis=0)
        logger.info("Finished calculating cluster averages")
        return average_responses

    @classes.cache_to_file('data/data_central_pixel_test_001.csv', '%i')
    def build_test_predictors(self):
        test_files = sorted(os.listdir(TEST_IMAGE_PATH))
        test_predictors = self.build_features(test_files, False)
        return test_predictors

    def predict_test(self, average_responses):
        logger.info("Calculating predictions for test set")

        # Now calculate the test set responses
        test_predictors = self.build_test_predictors()
        test_clusters = self.estimator.predict(test_predictors)
        test_averages = average_responses[test_clusters]
        return test_averages

    def execute(self):
        self.train_y = classes.get_training_data()
        self.predictors = self.build_train_predictors()
        self.fit_estimator()
        average_responses = self.get_cluster_averages()

        # Assign cluster averages for the training set, to get an in sample RMSE
        training_averages = average_responses[self.estimator.labels_]
        rmse = classes.rmse(training_averages, self.train_y[:, 1:])

        return self.predict_test(average_responses)


def central_pixel_benchmark(outfile="sub_central_pixel_001.csv"):
    """
    Tries to duplicate the central pixel benchmark, which is defined as:
    Simple benchmark that clusters training galaxies according to the color in the center of the image
    and then assigns the associated probability values to like-colored images in the test set.
    """

    test_averages = CentralPixelBenchmark().run()
    predictions = classes.Submission(test_averages)
    # Write to file
    predictions.to_file(outfile)


class RandomForestModel(classes.BaseModel):
    train_predictors_file = 'data/data_random_forest_train_001.csv'
    test_predictors_file = 'data/data_random_forest_test_001.csv'

    @staticmethod
    def process_image(img):
        return img.grid_sample(20, 2).flatten().astype('float64') / 255

    def build_features(self, files, training=True):
        #  Sample a 5x5 grid of pixels.  Totals 75 features: 5x5x3
        logger.info("Building predictors")

        predictors = self.do_for_each_image(files, self.process_image, 75, training)
        return predictors

    @classes.cache_to_file(train_predictors_file)
    def build_train_predictors(self):
        file_list = classes.get_training_filenames(self.train_y)
        return self.build_features(file_list, True)

    @classes.cache_to_file(test_predictors_file)
    def build_test_predictors(self):
        test_files = sorted(os.listdir(TEST_IMAGE_PATH))
        return self.build_features(test_files, False)

    def get_estimator(self):
        classifier = RandomForestClassifier(random_state=0, verbose=1)
        return classifier

    def fit_estimator(self):
        start_time = time.clock()
        logger.info("Fitting random forest estimator")
        self.estimator = self.get_estimator()
        self.estimator.fit(self.train_x, self.train_y)
        logger.info("Finished fitting model in {}".format(time.clock() - start_time))

    def predict_test(self):
        self.test_y = self.estimator.predict(self.test_x)
        return self.test_y

    def execute(self):
        self.train_y = classes.get_training_data()
        self.train_x = self.build_train_predictors()
        self.fit_estimator()

        # Get an in sample RMSE
        training_predict = self.estimator.predict(self.train_x)
        rmse = classes.rmse(training_predict, self.train_y)

        self.test_x = self.build_test_predictors()
        return self.predict_test()


def random_forest_001(outfile="sub_random_forest_001.csv"):
    """
    First attempt at implementing a neural network.
    Uses a sample of central pixels in RGB space to feed in as inputs to the neural network
    Model is not tuned or CV'd, which are to be implemented in later models.
    """
    model = RandomForestModel()
    predictions = model.run()
    output = classes.Submission(predictions)
    output.to_file(outfile)


def ridge_regression():
    # read train Y
    train_y = classes.get_training_data()

    # randomly sample 10% Y and select the gid's
    n = 7000
    crop_size = 150

    train_y = train_y[np.random.randint(train_y.shape[0], size=n), :]
    train_x = np.zeros((n, crop_size ** 2))

    # load the training images and crop at the same time
    for row, gid in enumerate(train_y[:, 0]):
        img = classes.RawImage('data/images_training_rev1/' + str(int(gid)) + '.jpg')
        img.grayscale()
        img.crop(crop_size)
        img.flatten()
        train_x[row] = img.data
        print row

    pca = RandomizedPCA(1000, whiten=True)
    rgn = classes.RidgeClipped()

    pca_ridge = Pipeline([('pca', pca),
                          ('ridge', rgn)])

    # best ridge alpha = 10 ** 3.42
    parameters = {'ridge__alpha': 10 ** np.linspace(2.8, 4, 4)}

    grid_search = GridSearchCV(pca_ridge, parameters, cv=2, n_jobs=1, scoring='mean_squared_error')
    grid_search.fit(train_x, train_y[:, 1:])

    return grid_search
    # construct pipeline. PCA or LLE then RR
    # feed into cross validation