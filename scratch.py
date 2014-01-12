import classes
import run
import numpy as np

a = run.RandomForestModel()
a.train_x = a.build_train_predictors()


import classes
import run
import numpy as np
params = {
    'n_estimators': [10, 100, 250],
    'max_features': ['sqrt', 'log2', 'auto'],
    'min_samples_split': [2, 10, 50, 100],
    'min_samples_leaf': [1, 5, 10, 25, 50]
}
a = run.RandomForestModel(grid_search_parameters=params,
                          grid_search_sample=0.1)
a.train_x = a.build_train_predictors()
a.perform_grid_search_and_cv()
