"""
Run scripts for individual models in Galaxy Zoo
"""
import os
import time

import classes
import numpy as np
import logging
from constants import *
from sklearn.cluster import KMeans


logger = logging.getLogger(__name__)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
# Log to file
logfile = logging.FileHandler('run.log')
logfile.setLevel(logging.DEBUG)
logfile.setFormatter(log_formatter)
# Log to console
logstream = logging.StreamHandler()
logstream.setLevel(logging.INFO)
logstream.setFormatter(log_formatter)

logger.addHandler(logfile)
logger.addHandler(logstream)


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


def get_central_pixel_predictors(file_list, training):
    logger.info("Building predictors")
    dims = (N_TRAIN if training else N_TEST, 3)
    predictors = np.zeros(dims)
    counter = 0
    for row, f in enumerate(file_list):
        filepath = TRAIN_IMAGE_PATH if training else TEST_IMAGE_PATH
        image = classes.RawImage(os.path.join(filepath, f))
        predictors[row] = image.central_pixel.copy()
        counter += 1
        if counter % 1000 == 0:
            logger.info("Processed {} images".format(counter))
    return predictors


class CentralPixelBenchmark(classes.BaseModel):
    def build_train_predictors(self):
        # Build the training data - load all of the images and extract the RGB values of the central pixel
        # Resulting training dataset should be (70948, 3)
        self.train_y = classes.get_training_data()
        file_list = classes.get_training_filenames(self.train_y)
        self.predictors = get_central_pixel_predictors(file_list, True)

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

    def predict_test(self, average_responses):
        logger.info("Calculating predictions for test set")
        # Now calculate the test set responses
        test_files = sorted(os.listdir(TEST_IMAGE_PATH))
        test_predictors = get_central_pixel_predictors(test_files, False)
        test_clusters = self.estimator.predict(test_predictors)
        test_averages = average_responses[test_clusters]
        return test_averages

    def execute(self):
        self.build_train_predictors()

        # Save the dataset for future use
        data_filename = 'data/data_central_pixel_001.csv'
        logger.info("Finished loading predictors, saving to file {}".format(data_filename))
        np.savetxt(data_filename, self.predictors, delimiter=',', fmt="%i")

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


class NeuralNetworkModel(classes.BaseModel):
    def build_features(self, files, training=True):
        # Can parameterize this by having arguments:
        #   - feature_generator_function
        #   - n_features
        # So once an image is loaded, it gets passed to the feature_generator_function,
        # And the result of this function is added to the predictors array with n_features columns
        logger.info("Building predictors")
        dims = (N_TRAIN if training else N_TEST, 75)
        predictors = np.zeros(dims)
        counter = 0
        for row, f in enumerate(files):
            filepath = TRAIN_IMAGE_PATH if training else TEST_IMAGE_PATH
            image = classes.RawImage(os.path.join(filepath, f))
            predictors[row] = image.grid_sample(20, 2).flatten().astype('float64') / 255
            counter += 1
            if counter % 1000 == 0:
                logger.info("Processed {} images".format(counter))
        return predictors

    def build_train_predictors(self):
        self.train_y = classes.get_training_data()
        file_list = classes.get_training_filenames(self.train_y)
        self.train_x = get_central_pixel_predictors(file_list, True)

    def execute(self):
        self.build_train_predictors()
