import numpy as np


def create_data_set():
    """Create data set for traning.
    
    Args:
      none
      
    Returns:
      group: data set
      labels: data labels
    """
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    labels = ['Love', 'Love', 'Action', 'Action']
    return group, labels


if __name__ == '__main__':
    group, labels = create_data_set()
    print(group)
    print(labels)
