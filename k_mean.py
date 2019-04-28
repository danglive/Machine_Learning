from clustering import Clustering
import numpy as np

class KMEAN(Clustering):
    def fit(self, X, cluster_num=2):
        self.membership = np.random.randint(cluster_num, size=(X.shape[0]))
        clusters = np.zeros((cluster_num, X.shape[1]))
        
        last_menbership = self.membership
        
        while True:
            # update cluster centers
            for i in xrange(cluster_num):
                clusters[i, :] = np.mean(X[self.membership == i], axis=0)

            # update membership
            for i in xrange(X.shape[0]):
                self.membership[i] = np.argmin(np.sum(np.square(clusters - X[i,:]), axis=1))
        
            if all(last_menbership == self.membership):
                break
            else:
                last_menbership = self.membership