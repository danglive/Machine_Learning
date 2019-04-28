from classifier import Classifier
import numpy as np
from cvxpy import *

class SVM(Classifier):
    def fit(self, X_train, y_train, soft_margin=1):
        n, d = X_train.shape
        
        self.w_0 = Variable(1)
        self.w   = Variable(d)
        self.zet = Variable(n)
        
        A = X_train * y_train[:, np.newaxis]
        
        objective = Minimize(sum_squares(self.w)/2 + soft_margin*sum_entries(self.zet))
        constraints = [self.zet >= 0, A*self.w + y_train*self.w_0 >= 1 - self.zet]
        problem = Problem(objective, constraints)
        
        result = problem.solve()
        
    def predict(self, X_test):
        y_pred = ((X_test.dot(self.w.value) + self.w_0.value) > 0)*2 - 1
        y_pred = np.array(y_pred)[:,0]
        
        return y_pred