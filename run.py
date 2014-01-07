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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


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
    logger.info("In sample RMSE: {}".format(rmse))

    test_ids = classes.get_test_ids()
    solution = np.tile(solutions, (N_TEST, 1))
    predictions = np.concatenate((test_ids, solution), axis=1)
    outpath = os.path.join(SUBMISSION_PATH, outfile)
    submission_format = ['%i'] + ['%.10f' for x in range(37)]
    np.savetxt(outpath, predictions, delimiter=',', header=SUBMISSION_HEADER, fmt=submission_format, comments="")

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


def central_pixel_benchmark(outfile="sub_central_pixel_001.csv"):
    """
    Tries to duplicate the central pixel benchmark, which is defined as:
    Simple benchmark that clusters training galaxies according to the color in the center of the image
    and then assigns the associated probability values to like-colored images in the test set.
    """

    start_time = time.clock()

    # Build the training data - load all of the images and extract the RGB values of the central pixel
    # Resulting training dataset should be (70948, 3)
    training_data = classes.get_training_data()
    file_list = classes.get_training_filenames(training_data)
    predictors = get_central_pixel_predictors(file_list, True)

    # Save the dataset for future use
    data_filename = 'data/data_central_pixel_001.csv'
    logger.info("Finished loading predictors, saving to file {}".format(data_filename))
    np.savetxt(data_filename, predictors, delimiter=',', fmt="%i")

    # Fit a k-means clustering estimator
    # We use 37 centers initially because there are 37 classes
    # Seems like the sample submission used 6 clusters
    logger.info("Fitting kmeans estimator")
    estimator = KMeans(init='k-means++', n_clusters=37)
    estimator.fit(predictors)
    logger.info("Finished fitting model in {}".format(time.clock() - start_time))

    # Get the average response for each cluster in the training set
    # This is a 37 x 37 array, one row for each cluster, and one column for each class
    logger.info("Calculating cluster averages")
    average_responses = np.zeros((37, 37))
    for cluster in range(37):
        idx = estimator.labels_ == cluster
        responses = training_data[idx, 1:]
        average_responses[cluster] = responses.mean(axis=0)
    logger.info("Finished calculating cluster averages")

    # Assign cluster averages for the training set, to get an in sample RMSE
    training_averages = average_responses[estimator.labels_]
    rmse = classes.rmse(training_averages, training_data[:, 1:])
    logger.info("In sample RMSE: {}".format(rmse))

    logger.info("Calculating predictions for test set")
    # Now calculate the test set responses
    test_files = sorted(os.listdir(TEST_IMAGE_PATH))
    test_predictors = get_central_pixel_predictors(test_files, False)
    test_clusters = estimator.predict(test_predictors)
    test_averages = average_responses[test_clusters]
    predictions = np.concatenate((classes.get_test_ids(), test_averages), axis=1)

    outpath = os.path.join(SUBMISSION_PATH, outfile)
    logger.info("Writing submission to file {}".format(outpath))
    submission_format = ['%i'] + ['%.10f' for x in range(37)]
    np.savetxt(outpath, predictions, delimiter=',', header=SUBMISSION_HEADER, fmt=submission_format, comments="")

    end_time = time.clock()
    logger.info("Model completed in {}".format(end_time - start_time))
