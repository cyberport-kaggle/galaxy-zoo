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


def central_pixel_benchmark(outfile="sub_central_pixel_001.csv"):
    """
    Tries to duplicate the central pixel benchmark, which is defined as:
    Simple benchmark that clusters training galaxies according to the color in the center of the image
    and then assigns the associated probability values to like-colored images in the test set.
    """

    start_time = time.clock()

    # Build the training data - load all of the images and extract the RGB values of the central pixel
    # Resulting training dataset should be (70948, 3)
    logger.info("Building predictors")
    predictors = np.zeros((N_TRAIN, 3))
    training_data = classes.get_training_data()
    file_list = classes.get_training_filenames(training_data)
    counter = 0
    for row, f in enumerate(file_list):
        image = classes.RawImage(os.path.join(TRAIN_IMAGE_PATH, f))
        predictors[row] = image.central_pixel.copy()
        counter += 1
        if counter % 1000 == 0:
            logger.info("Processed {} images".format(counter))

    # Save the dataset for future use
    data_filename = 'data/data_central_pixel_001.csv'
    logger.info("Finished loading predictors, saving to file {}".format(data_filename))
    np.savetxt(data_filename, predictors, delimiter=',', fmt="%i")

    # Fit a k-means clustering estimator
    # We use 37 centers initially because there are 37 classes
    logger.info("Fitting kmeans estimator")
    estimator = KMeans(init='k-means++', n_clusters=37)
    estimator.fit(predictors)
    logger.info("Finished fitting model in {}".format(time.clock() - start_time))

    # Get the average response for each cluster in the training set
    # This is a 37 x 37 array, one row for each cluster, and one column for each class
    average_responses = np.zeros((37, 37))

    end_time = time.clock()
    logger.info("Model completed in {}".format(end_time - start_time))
