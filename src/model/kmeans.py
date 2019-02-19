print(__doc__)

from time import time

import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


class SimpleKMeans():

    def __init__(self, data, config, sumatra_config=None):
        self.config = config
        self.sumatra_config = sumatra_config
        self.data = data
        np.random.seed(42)

    def bench_k_means(self, estimator, name, data):
        t0 = time()

        estimator.fit(data)
        print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
              % (name, (time() - t0), estimator.inertia_,
                 metrics.homogeneity_score(self.labels, estimator.labels_),
                 metrics.completeness_score(self.labels, estimator.labels_),
                 metrics.v_measure_score(self.labels, estimator.labels_),
                 metrics.adjusted_rand_score(self.labels, estimator.labels_),
                 metrics.adjusted_mutual_info_score(self.labels, estimator.labels_)))

    def fit(self, X, y):
        X = self.data.X
        n_samples, n_features = self.data.shape
        n_digits = len(np.unique(self.data.y))

        print("n_digits: %d, \t n_samples %d, \t n_features %d"
              % (n_digits, n_samples, n_features))

        print(82 * '_')
        print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

        self.bench_k_means(KMeans(init=self.config.init, n_clusters=n_digits, n_init=10),
                           name=self.config.init, data=X)

        print(82 * '_')
