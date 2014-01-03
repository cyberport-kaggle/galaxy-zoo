"""
Scripts/scratchpad for data exploration
"""
import os
import numpy as np

solutions_file = 'data/solutions_training.csv'
solution = np.loadtxt(open(solutions_file, 'rb'), delimiter=',', skiprows=1)

training_files = os.listdir('data/images_training')