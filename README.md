# Kaggle Galaxy Zoo - Shallow Learners submission

## Final Submission

See `final_submission.py` for the code to run the final submission.  The cross validation score
for the model was about .106, and the leaderboard score was .104.

### Workspace setup

 - Ensure that a folder `data` exists
 - Ensure that the training images are extracted to `images_training_rev1` under `data`
 - Ensure that the test images are extracted to `images_test_rev1` under `data

### Overview of model

Our final model is a single layer K-means feature learner for generating features, which are then used in a
ridge regression / random forest predictor.  Our feature learning followed the procedure outlined in
[this paper](http://www.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf) by
[Adam Coates](http://www.stanford.edu/~acoates/) and Andrew Ng of Stanford.

The full model runs in under two hours on a hi1.4xlarge AWS instance (16 cores, 60.5 GB RAM).

In pseudocode, the overall pipeline looks like this:

 - Crop images to 150x150, then scale to 15x15 (keep RGB channels)
 - From the training images, extract 400,000 5x5 image patches
 - Normalize and whiten the patches
 - Fit k-means on the image patches with 3,000 centroids
 - For each training image, and for each 5x5 patch in the image (the "window"), generate the features:

   - Normalize and whiten the window
   - Use a soft threshold to encode the window with the centroids
   - Pool the encoded features over the quadrants of the image

 - Use the 12,000 features (3,000 centroids x 4 quadrants) to train a ridge regressor on all 37 response columns
 - Use the predictions from the ridge regressor to train a random forest on all 37 response columns
 - On the test data, repeat the feature generation process
 - Predict using the ridge regressor and random forest

We experimented briefly with alternative predictors at the end of the pipeline with little success.  We
mostly used ridge regression because of it's built-in ability to handled multi-class outputs.  We then
used the random forest to account for correlations among response columns.

### Tuning parameters

We experimented with the following parameters in order to arrive at our final settings:

  - Crop size
  - Scale size
  - Number of patches to extract
  - Patch size
  - Using test images in patch extraction
  - Number of centroids
  - Step size when extracting windows from images
  - Pooling methodology

We initially had really bad performance with the raw images because the patches were not large enough
to pick up any meaningful features.  Cropping and scaling was critical to significantly improving the
performance of the feature generator.  However, given the small scaling, we worried that we were losing too
much fidelity and tried increasing the scale size, but quickly ran into processing constraints.

Increasing the number of centroids was one of the more important tuning factors.

### Next steps

It is possible to stack the kmeans feature generators as you would with a deep learning network, but we
ran out of time before we could implement it.
