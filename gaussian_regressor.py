from regressor import Regressor
import numpy as np

class GaussianRegressor(Regressor):
    def fit(self, X_train, y_train, lam=1):
        self.K_n_inverse = self._compute_rbf(X_train, X_train)
        self.K_n_inverse = np.linalg.inv(self._compute_rbf(X_train, X_train))
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test):
        K_0 = self._compute_rbf(X_test, self.X_train)
        return K_0.dot(self.K_n_inverse).dot(self.y_train)
    
    def _compute_rbf(self, x, y, sigma2=1):
        x = np.expand_dims(np.repeat(x, y.shape[0], axis=1), axis=-1)
        y = np.expand_dims(np.repeat(y, x.shape[0], axis=1).transpose(), axis=-1)
        
        difference = np.sum(np.square(x - y), axis=-1)
        return np.exp(-1/2/sigma2*difference)
    