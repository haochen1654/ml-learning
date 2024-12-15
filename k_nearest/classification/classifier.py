import numpy as np
import operator

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
    for i in range(k):
        selected_label = labels[sorted_distance[i]]
        class_count[selected_label] = class_count.get(selected_label, 0) + 1
    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1),
                                reverse=True)
    return sorted_class_count[0][0]


if __name__ == '__main__':
    group, labels = create_data_set()
    test = [101, 20]

    # kNN classification
    test_class = classify(test, group, labels, 3)

    print(test_class)
