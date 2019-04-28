from classifier import Classifier
import numpy as np

class GaussianBayes(Classifier):
    def fit(self, X_train, y_train):
        # Estimate pi - the prior distribution of labels
        self.labels, counts = np.unique(y_train, return_counts=True)
        self.pi = counts.astype('float')/y_train.shape[0]

        self.means = []
        self.cvars = []

        # Estimate means and covarances - the likelihood of input
        for label in self.labels:
            A = X_train[y_train == label]
            mean = np.mean(A, axis=0)

            A = A - mean
            cvar = A.transpose().dot(A)/A.shape[0]

            self.means += [mean]
            self.cvars += [cvar]
        
    def predict(self, X_test):
        y_pred = np.zeros(X_test.shape[0], dtype='int')
        
        for i in xrange(len(X_test)):
            x = X_test[i]
            probs = []

            for label in self.labels:
                prob  = self.pi[label] / np.sqrt(np.linalg.det(self.cvars[label]))
                prob *= np.exp(-1/2*(x-self.means[label]).transpose().dot(np.linalg.inv(self.cvars[label])).dot(x-self.means[label]))
                probs += [prob]
                
            y_pred[i] = np.argmax(np.asarray(probs))
            
        return y_pred