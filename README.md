
These are minimal implementations of most basic machine learning algorithms taught in Columbia Machine Learning on edx.org. All abstractions are strictly avoided to highlight the direct translation from mathematical formulations to algorithm implementations. This is only for educational purpose and by no means for production.

Implemented algorithms are:

* Regression
    * Least squares
    * Ridge regression
    * Gaussian process
* Classification
    * Gaussian Bayes
    * Naive Bayes (Poisson)
    * Decision tree
    * Support vector machine
* Clustering
    * k-mean
    * Gaussian mixture model
    
# Code structure

1. classifier.py, regressor.py, clustering.py => 3 base classes for classification, regression and clustering
2. ridge_regressor.py, gaussian_regressor.py => implements Least Square, Ridge Regression and Gaussian Process Regression
3. gaussian_bayes.py, naive_bayes.py, decision_tree.py, svm.py => implements the respective classification methods
4. k_mean.py, gmm.py => implements the respective clustering method

# Usage

This is a minimal working example for decision tree.

cd basic-ml-algorithms

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree

dataset = datasets.load_iris()
X = dataset.data[:,:3]
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

classifier = DecisionTree()
classifier.fit(X_train, y_train)

print 'Accuracy:', classifier.eval(X_test, y_test)
```

The notebook examples.ipynb contains demonstrations of all algorithms.
