"""
Run scripts for individual models in Galaxy Zoo
"""
import time

import classes
import numpy as np
import logging
from constants import *
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import models


logger = logging.getLogger('galaxy')


def train_set_average_benchmark(outfile="sub_average_benchmark_000.csv"):
    """
    What should be the actual baseline.  Takes the training set solutions, averages them, and uses that as the
    submission for every row in the test set
    """
    start_time = time.time()
    training_data = classes.TrainSolutions().data

    solutions = np.mean(training_data, axis=0)

    # Calculate an RMSE
    train_solution = np.tile(solutions, (N_TRAIN, 1))
    rmse = classes.rmse(train_solution, training_data)

    solution = classes.Submission(np.tile(solutions, (N_TEST, 1)))
    solution.to_file(outfile)

    end_time = time.time()
    logger.info("Model completed in {}".format(end_time - start_time))


def central_pixel_benchmark(outfile="sub_central_pixel_001.csv"):
    """
    Tries to duplicate the central pixel benchmark, which is defined as:
    Simple benchmark that clusters training galaxies according to the color in the center of the image
    and then assigns the associated probability values to like-colored images in the test set.
    """

    test_averages = models.Benchmarks.CentralPixelBenchmark().run()
    predictions = classes.Submission(test_averages)
    # Write to file
    predictions.to_file(outfile)


def random_forest_001(outfile="sub_random_forest_001.csv", n_jobs=1):
    """
    First attempt at implementing a neural network.
    Uses a sample of central pixels in RGB space to feed in as inputs to the neural network

    # 3-fold CV using half the training set reports RMSE of .126 or so
    """
    model = models.RandomForest.RandomForestModel(n_jobs=n_jobs)
    predictions = model.run()
    output = classes.Submission(predictions)
    output.to_file(outfile)


def random_forest_cascade_test():
    """
    Experiment to compare whether training the random forest with all Ys or training the Ys in a cascade is better
    """
    mdl_cascade = models.RandomForest.RandomForestModel(cascade=True)
    mdl_base = models.RandomForest.RandomForestModel()


def ridge_rf_001(outfile='sub_ridge_rf_001.csv'):
    mdl = models.Ridge.RidgeRFModel(cv_sample=0.5, cv_folds=2)
    mdl.run('cv')
    mdl.run('train')
    mdl.run('predict')
    sub = classes.Submission(mdl.test_y)
    sub.to_file(outfile)
