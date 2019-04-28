from classifier import Classifier
import numpy as np

class Node(object):
    def __init__(self, split_variable, split_value, left_child, right_child):
        self.split_variable = split_variable
        self.split_value = split_value
        self.left_child = left_child
        self.right_child = right_child

    def get_name(self):
        return 'node'

class Leaf(object):
    def __init__(self, label):
        self.label = label

    def get_name(self):
        return 'leaf'

class DecisionTree(Classifier):
    def fit(self, X_train, y_train, soft_margin=1):
        self.tree = self._build_tree(X_train, y_train)
        
    def predict(self, X_test):
        y_pred = np.zeros(X_test.shape[0], dtype='int')
        
        for i in xrange(X_test.shape[0]):
            x = X_test[i]
            y_pred[i] = self._predict_element(x)
        
        return y_pred
    
    def _predict_element(self, x):
        node = self.tree
        
        while node.get_name() != 'leaf':
            if x[node.split_variable] < node.split_value:
                node = node.left_child
            else:
                node = node.right_child
            
        return node.label
    
    def _build_tree(self, X, y):
        cur_gini = self._compute_gini(y)
        
        if cur_gini < 0.1 or len(y) < 5:
            values, counts = np.unique(y,return_counts=True)
            return Leaf(values[np.argmax(counts)])
        else:
            split_variable = -1
            split_value =  0
            min_gini = cur_gini

            for i in xrange(X.shape[1]):
                splits = np.unique(X[:,i])
                
                if len(splits) > 1:
                    for j in xrange(len(splits) - 1):
                        split = (splits[j] + splits[j+1])/2

                        temp_y_1 = y[X[:,i] < split]
                        temp_y_2 = y[X[:,i] > split]

                        gini = float(len(temp_y_1))/len(y)*self._compute_gini(temp_y_1) + float(len(temp_y_2))/len(y)*self._compute_gini(temp_y_2)

                        if gini < min_gini:
                            min_gini = gini
                            split_variable = i
                            split_value = split

                            y_1 = temp_y_1
                            y_2 = temp_y_2
                            X_1 = X[X[:,i] < split]
                            X_2 = X[X[:,i] > split]  
                            
            if min_gini >= cur_gini:
                values, counts = np.unique(y, return_counts=True)
                return Leaf(values[np.argmax(counts)])
            else:
                left_child  = self._build_tree(X_1, y_1)
                right_child = self._build_tree(X_2, y_2)

                return Node(split_variable, split_value, left_child, right_child)
        
    def _compute_gini(self, labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        freqs = counts.astype('float')/len(labels)
        
        return -np.sum(freqs * np.log(freqs))