print(__doc__)

from time import time

import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


class SimpleKMeans():

    def __init__(self):
        self.digits = load_digits()
        self.labels = self.digits.target
        self.sample_size = 300
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
                 metrics.adjusted_mutual_info_score(self.labels, estimator.labels_),
                 metrics.silhouette_score(data, estimator.labels_,
                                          metric='euclidean',
                                          sample_size=self.sample_size)))

    def fit(self,X,y):
        data = scale(self.digits.data)
        n_samples, n_features = data.shape
        n_digits = len(np.unique(self.digits.target))

        print("n_digits: %d, \t n_samples %d, \t n_features %d"
              % (n_digits, n_samples, n_features))

        print(82 * '_')
        print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

        self.bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
                           name="k-means++", data=data)

        self.bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
                           name="random", data=data)

        # in this case the seeding of the centers is deterministic, hence we run the
        # kmeans algorithm only once with n_init=1
        pca = PCA(n_components=n_digits).fit(data)
        self.bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
                           name="PCA-based",
                           data=data)
        print(82 * '_')
