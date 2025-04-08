import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixture:
    
    def __init__(self, n_components=1, max_iter=100, tol=1e-3, random_state=None):
        
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = np.random.RandomState(random_state)

    def _initialize(self, X):
        
        n_samples, n_features = X.shape

        self.weights_ = np.full(self.n_components, 1 / self.n_components)

        self.means_ = X[self.random_state.choice(n_samples, self.n_components, replace=False)]

        shared_cov = np.cov(X.T) + 1e-6 * np.eye(n_features)
        self.covariances_ = np.array([shared_cov for _ in range(self.n_components)])

    def _e_step(self, X):
        n_samples = X.shape[0]
        self.resp = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            self.resp[:, k] = self.weights_[k] * multivariate_normal.pdf(
                X, mean=self.means_[k], cov=self.covariances_[k]
            )

        self.resp /= self.resp.sum(axis=1, keepdims=True)  

    def _m_step(self, X):
        n_samples, n_features = X.shape
        N_k = self.resp.sum(axis=0)  

        self.weights_ = N_k / n_samples
        self.means_ = np.dot(self.resp.T, X) / N_k[:, np.newaxis]

        self.covariances_ = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            weighted_sum = np.dot(self.resp[:, k] * diff.T, diff)
            self.covariances_[k] = weighted_sum / N_k[k] + 1e-6 * np.eye(n_features)

    def _log_likelihood(self, X):
        likelihood = np.zeros((X.shape[0], self.n_components))

        for k in range(self.n_components):
            likelihood[:, k] = self.weights_[k] * multivariate_normal.pdf(
                X, mean=self.means_[k], cov=self.covariances_[k]
            )

        total_likelihood = np.sum(likelihood, axis=1)
        return np.sum(np.log(total_likelihood + 1e-10))  # Add epsilon to prevent log(0)

    def fit(self, X):
        
        self._initialize(X)
        log_likelihood_old = None

        for i in range(self.max_iter):
            self._e_step(X)
            self._m_step(X)
            log_likelihood_new = self._log_likelihood(X)

            if log_likelihood_old is not None:
                if np.abs(log_likelihood_new - log_likelihood_old) < self.tol:
                    break
            log_likelihood_old = log_likelihood_new

        return self

    def predict_proba(self, X):
       
        probs = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            probs[:, k] = self.weights_[k] * multivariate_normal.pdf(
                X, self.means_[k], self.covariances_[k]
            )
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
