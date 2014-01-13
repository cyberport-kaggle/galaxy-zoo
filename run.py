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


def ridge_regression():
    # read train Y
    train_y = classes.train_solutions.data
    train_filenames = classes.train_solutions.filenames

    # randomly sample 10% Y and select the gid's
    n = 7000
    crop_size = 150
    scale = 0.1
    train_y = train_y[np.random.randint(train_y.shape[0], size=n), :]
    train_x = np.zeros((n, (crop_size * scale) ** 2 * 3))

    # load the training images and crop at the same time
    for row, gid in enumerate(train_filenames):
        img = classes.RawImage('data/images_training_rev1/' + str(int(gid)) + '.jpg')
        img.crop(crop_size)
        img.rescale(scale)
        img.flatten()
        train_x[row] = img.data
        if (row % 10) == 0: print row

    pca = RandomizedPCA(1000, whiten=True)
    rgn = classes.RidgeClipped()

    pca_ridge = Pipeline([('ridge', rgn)])

    # best ridge alpha = 10 ** 3.42
    parameters = {'ridge__alpha': 10 ** np.linspace(-1, 2, 8)}

    grid_search = GridSearchCV(pca_ridge, parameters, cv=2, n_jobs=1, scoring='mean_squared_error', refit=False)
    grid_search.fit(train_x, train_y)

    return grid_search


def ridge_rf():
# read train Y
    train_y = classes.train_solutions.data

    # randomly sample 10% Y and select the gid's
    n = 7000
    crop_size = 150
    scale = 0.1
    train_y = train_y[np.random.randint(train_y.shape[0], size=n), :]
    train_x = np.zeros((n, (crop_size * scale) ** 2 * 3))

    # load the training images and crop at the same time
    for row, gid in enumerate(classes.train_solutions.filenames):
        img = classes.RawImage('data/images_training_rev1/' + str(int(gid)) + '.jpg')
        img.crop(crop_size)
        img.rescale(scale)
        img.flatten()
        train_x[row] = img.data
        if (row % 10) == 0: print row

    ridge_rf = classes.RidgeRF()

    parameters = {'alpha': [14], 'n_estimators': [10]}

    grid_search = GridSearchCV(ridge_rf, parameters, cv=2, n_jobs=1, scoring='mean_squared_error', refit=False)
    grid_search.fit(train_x, train_y)

    return grid_search
