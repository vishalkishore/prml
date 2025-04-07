import numpy as np

class KMeans:
    def __init__(self, n_clusters=8, init='random', max_iter=300, tol=1e-4, random_state=None, verbose=False):
        
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = np.random.RandomState(random_state)
        self.verbose = verbose

    def fit(self, X):
        n_samples = X.shape[0]

        if self.init == 'random':
            indices = self.random_state.choice(n_samples, self.n_clusters, replace=False)
            self.cluster_centers_ = X[indices]
        else:
            raise NotImplementedError("Only 'random' init is implemented")

        for i in range(self.max_iter):
            if self.verbose:
                print(f"Iteration {i + 1}")

            labels = self._assign_labels(X)
            new_centers = self._compute_centroids(X, labels)

            shift = np.linalg.norm(self.cluster_centers_ - new_centers)
            if shift < self.tol:
                break

            self.cluster_centers_ = new_centers

        self.labels_ = labels
        return self

    def _assign_labels(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X, labels):
        centroids = []
        for i in range(self.n_clusters):
            points = X[labels == i]
            if len(points) > 0:
                centroids.append(points.mean(axis=0))
            else:
                centroids.append(self.cluster_centers_[i]) 
        return np.array(centroids)

    def predict(self, X):
        return self._assign_labels(X)
