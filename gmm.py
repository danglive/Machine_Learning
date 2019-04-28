from clustering import Clustering
import numpy as np
from scipy.stats import multivariate_normal

class GMM(Clustering):
    def fit(self, X, cluster_num=3):
        self.membership = np.random.dirichlet(np.ones(cluster_num), size=(X.shape[0]))
        last_loss = 1e9
        
        prior = np.random.dirichlet(np.ones(cluster_num), size=1)[0]
        means = np.random.normal(size=(cluster_num, X.shape[1]))
        cvars = np.random.rand(cluster_num, X.shape[1], X.shape[1])
        
        while True:
            # Update model parameter
            n = np.sum(self.membership, axis=0)

            prior = n/X.shape[0]

            for i in xrange(means.shape[0]):
                means[i] = 1 / n[i] * np.sum(X * self.membership[:,i][:, np.newaxis], axis=0)

            for i in xrange(cvars.shape[0]):
                A = X - means[i]
                cvars[i] = 1 / n[i] * (A*self.membership[:,i][:, np.newaxis]).transpose().dot(A)

            # Update posterier distribution
            for k in xrange(prior.shape[0]):
                self.membership[:,k] = multivariate_normal.pdf(X, mean=means[k], cov=cvars[k]) * prior[k]

            loss = np.sum(np.log(np.sum(self.membership, axis=1)))
            self.membership = self.membership/np.sum(self.membership, axis=1)[:, np.newaxis]

            if np.abs(loss - last_loss) < 1e-6:
                break
            else:
                last_loss = loss
                
    def get_membership(self):
        return np.argmax(self.membership, axis=1)