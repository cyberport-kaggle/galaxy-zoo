from sklearn.cluster import KMeans
import classes
from classes import logger
import time
from constants import *
import numpy as np
import os
from models.Base import BaseModel


class CentralPixelBenchmark(BaseModel):
    # TODO: can significantly refactor, since this was created before the BaseModel class was improved.  But maybe not worth it.
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
        return self.build_features(self.training_data.filenames, True)

    def fit_estimator(self):
        # Fit a k-means clustering estimator
        # We use 37 centers initially because there are 37 classes
        # Seems like the sample submission used 6 clusters
        start_time = time.time()
        logger.info("Fitting kmeans estimator")
        self.estimator = KMeans(init='k-means++', n_clusters=37)
        self.estimator.fit(self.predictors)
        logger.info("Finished fitting model in {}".format(time.time() - start_time))

    def get_cluster_averages(self):
        # Get the average response for each cluster in the training set
        # This is a 37 x 37 array, one row for each cluster, and one column for each class
        logger.info("Calculating cluster averages")
        average_responses = np.zeros((37, 37))
        for cluster in range(37):
            idx = self.estimator.labels_ == cluster
            responses = self.train_y[idx, :]
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
        self.predictors = self.build_train_predictors()
        self.fit_estimator()
        average_responses = self.get_cluster_averages()

        # Assign cluster averages for the training set, to get an in sample RMSE
        training_averages = average_responses[self.estimator.labels_]
        rmse = classes.rmse(training_averages, self.train_y)

        return self.predict_test(average_responses)

