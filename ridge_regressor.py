from regressor import Regressor
import numpy as np

class RidgeRegressor(Regressor):
    def fit(self, X_train, y_train, lam=1):
        p_in = np.linalg.pinv(lam*np.identity(X_train.shape[1]) + np.transpose(X_train).dot(X_train))
        self.w_rr = p_in.dot(np.transpose(X_train)).dot(y_train)

    def predict(self, X_test):
        return X_test.dot(self.w_rr)