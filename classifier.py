import numpy as np

class Classifier(object):
    def fit(self, X_train, y_train):
        pass
        
    def predict(self, X_test):
        pass
        
    def eval(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.sum(y_pred == y_test)/float(y_pred.shape[0])