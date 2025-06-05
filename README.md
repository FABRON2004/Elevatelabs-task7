# Elevatelabs-task7
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

data = datasets.load_breast_cancer()
X = data.data[:, :2]
y = data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf_linear = SVC(kernel='linear', C=1)
clf_linear.fit(X_train, y_train)

clf_rbf = SVC(kernel='rbf', C=1, gamma=0.5)
clf_rbf.fit(X_train, y_train)

def plot_decision_boundary(clf, X, y, title):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.show()

plot_decision_boundary(clf_linear, X, y, "SVM with Linear Kernel")
plot_decision_boundary(clf_rbf, X, y, "SVM with RBF Kernel")

scores_linear = cross_val_score(clf_linear, X, y, cv=5)
scores_rbf = cross_val_score(clf_rbf, X, y, cv=5)

print("Linear Kernel CV Accuracy:", scores_linear.mean())
print("RBF Kernel CV Accuracy:", scores_rbf.mean())
