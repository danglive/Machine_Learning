import numpy as np

class Regressor(object):
    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test):
    	pass
        
    def eval(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return 0.5 * np.mean(np.square(y_pred - y_test))