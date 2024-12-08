import numpy as np
from absl import app


def createDataSet():
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    labels = ['Love', 'Love', 'Action', 'Action']
    return group, labels


if __name__ == '__main__':
    group, labels = createDataSet()
    print(group)
    print(labels)
