"""
Scripts/scratchpad for data exploration
"""
import os
import numpy as np
import classes
import logging
from scipy import misc
from skimage import color

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def create_predictors():
    predictors = np.zeros((70948, ))
    logger.info("Reading solutions file")
    solutions_file = 'data/solutions_training.csv'
    solution = np.loadtxt(open(solutions_file, 'rb'), delimiter=',', skiprows=1)
    counter = 0
    logger.info("Reading in images")
    for row, gid in enumerate(solution):
        filename = str(int(gid[0]))
        image = classes.RawImage('data/images_training/{}.jpg'.format(filename))
        image.grayscale()
        predictors[row] = image.data[212, 212]
        counter += 1
        if counter % 1000 == 0:
            logger.debug("Processed {} images".format(counter))

    return predictors

