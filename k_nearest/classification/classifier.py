import numpy as np

from raw_data.data_creator import create_data_set


def classify(input: list, reference_data_set: np.array, labels: list,
             k: int) -> str:
    """Create data set for traning.
    
    Args:
      input: unclassified data point.
      reference_data_set: referred trained data set.
      labels: label list.
      k: kNN Algorithm args, the kth nearest points. 
      
    Returns:
      classified_result: the label result.
    """
    rows = reference_data_set.shape[0]
    diff_set = np.tile(input, (rows, 1)) - reference_data_set
    square_diff_set = diff_set**2
    square_distance = square_diff_set.sum(axis=1)
    distance = square_distance**0.5
    sorted_distance = distance.argsort()
    class_count = {}
    for i in k:
        selected_label = labels[sorted_distance[i]]
        class_count[selected_label] = class_count.get(selected_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), k)


if __name__ == '__main__':
    group, labels = create_data_set()
    print(group)
    print(labels)
