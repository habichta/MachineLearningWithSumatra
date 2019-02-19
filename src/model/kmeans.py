print(__doc__)

from time import time

import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans


class SimpleKMeans():

    def __init__(self, data, config, sumatra_config=None):
        self.config = config
        self.sumatra_config = sumatra_config
        self.data = data
        np.random.seed(42)

    def bench_k_means(self, estimator, name, data):
        t0 = time()

        estimator.fit(data.X)
        print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
              % (name, (time() - t0), estimator.inertia_,
                 metrics.homogeneity_score(data.y, estimator.labels_),
                 metrics.completeness_score(data.y, estimator.labels_),
                 metrics.v_measure_score(data.y, estimator.labels_),
                 metrics.adjusted_rand_score(data.y, estimator.labels_),
                 metrics.adjusted_mutual_info_score(data.y, estimator.labels_)))

    def fit(self, X, y):

        print("n_digits: %d, \t n_samples %d, \t n_features %d"
              % (self.data.n_digits, self.data.n_samples, self.data.n_features))

        print(82 * '_')
        print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI')

        self.bench_k_means(KMeans(init=self.config.init, n_clusters=self.data.n_digits, n_init=10),
                           name=self.config.init, data=self.data)

        print(82 * '_')
