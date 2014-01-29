"""
Testing KMeansFeatures on the CIFAR dataset to make sure that our clustering works
"""

import numpy as np
import models


def unpickle(fname):
    import cPickle
    fo = open(fname, 'rb')
    data = cPickle.load(fo)
    fo.close()
    return data


def get_batch(n):
    fpath = "./data/cifar-10/data_batch_{}".format(n)
    return unpickle(fpath)

data = [get_batch(i)['data'] for i in range(1, 6)]
data = np.vstack(data)
rdata = data.reshape((50000, 3, 32, 32))
rdata = rdata.swapaxes(1, 3)
rdata = rdata.swapaxes(1, 2)

mdl = models.KMeansFeatures.KMeansFeatures(rf_size=6, num_centroids=1600, num_patches=400000)
mdl.fit(rdata)
