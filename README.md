# galaxy-zoo


## Current Models

### Central Pixel Clustering Benchmark

### Ridge Regression

### Random Forest Regression


## Current Feature sets

### Sampled Pixels around center

## Ideas

### Training methods

 - Training by class - Instead of training all 37 response columns at once, train each class separately.
 - Normalize class probability sums to 1 - Because of the structure of the Galaxy Zoo tree, each row's class sums
 are different (except for classes 1, and 6, which sum to 1 for all rows).  Instead of using the raw numbers, scale
 the classes such that they always sum to 1
 - Feed parent class predictions to children classes - Follow the hierarchy of the Galaxy Zoo tree structure.  For example,
 Feed the prediction for Class 1.1 into the model for Class 7.  Not sure if this should be paired with the normalization above.

### Data preprocessing / Feature generation

 - Unsupervised feature learning - http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial
 - K-means clustering for feature learning - Research from the Stanford group suggests that k-means can be just as effective
 as more complicated unsupervised feature learning techniques in generating features
 - Whitening - Something called ZCA that is apparently critical to improving the accuracy of k-means for feature generation:
 http://ufldl.stanford.edu/wiki/index.php/Whitening and http://ufldl.stanford.edu/wiki/index.php/Exercise:PCA_and_Whitening

