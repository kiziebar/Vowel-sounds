import numpy as np
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, resolution=0.1):
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z=classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z=Z.reshape(xx1.shape)
    plt.contour(xx1, xx2, Z)


def plot_decision_regions1d(X, y, classifier):
    x1_min, x1_max = X.min() - 1, X.max() + 1

    #xx1 = np.linspace(x1_min, x1_max, len(X))
    xx1 = np.arange(x1_min, x1_max, 0.01)
    xx1=xx1.reshape(-1,1)
    Z=classifier.predict(xx1)
    plt.plot(xx1,Z)
    