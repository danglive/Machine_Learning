from classifier import Classifier
import numpy as np

class NaiveBayes(Classifier):
    def fit(self, X_train, y_train):
        # Estimate pi - the prior distribution of labels
        self.labels, counts = np.unique(y_train, return_counts=True)
        self.pi = counts.astype('float')/y_train.shape[0]
        
        # Estimate the Poisson likelihood indexed by label and word
        self.poisson_map = np.zeros((len(self.labels), X_train.shape[1]))
        
        for label in self.labels:
            self.poisson_map[label] = np.sum(X_train[y_train==label], axis=0).astype('float')/np.sum(y_train==label)
            
        self.sum_poisson_map = np.sum(self.poisson_map, axis=1)
        
    def predict(self, X_test):
        y_pred = np.zeros(X_test.shape[0], dtype='int')
        
        for i in xrange(X_test.shape[0]):
            x = X_test[i]
            probs = []
            
            for label in self.labels:
                indices = x.indices[self.poisson_map[label, x.indices] != 0]
                x_j =        x.data[self.poisson_map[label, x.indices] != 0]
                
                prob   = np.log(self.pi[label]) - self.sum_poisson_map[label] + np.sum(x_j * np.log(self.poisson_map[label, indices]))
                probs += [prob]
                
            y_pred[i] = np.argmax(np.asarray(probs))
            
        return y_pred