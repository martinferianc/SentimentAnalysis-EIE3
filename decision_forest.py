from decision_tree import DecisionTree
from exceptions import NoModeError
import pickle
import os
import numpy as np
import copy


class DecisionForest:
    """
    A wrapper class to store the trees
    """

    def __init__(self, folder_path=None):
        # Loads all the trees inside a folder
        self.trees = []
        if folder_path is not None:
            trees_files = os.listdir(folder_path)
            for tree_file in trees_files:
                if "tree" in tree_file:
                    t = DecisionTree()
                    t.load(tree_file)
                    self.trees.append(copy.deepcopy(t))

    # Get scores from all the emotion trees
    def test_trees(self, feature_set, mode="var"):
        test_results = []
        for i in range(len(self.trees)):
            # Appends the label, variance and level of the node, also the index representing the emotion
            test_results.append((self.trees[i].test(feature_set), i + 1))

        test_results_ones = []
        # Filters out zero labels
        for result in test_results:
            # Label
            if result[0][0] == 1:
                test_results_ones.append(result)
        if len(test_results_ones) == 0:
            test_results.sort(key=lambda tup: tup[0][1])
            test_results.reverse()
            return test_results[0][1]

        # If no tree can identify it just return a random label
        if len(test_results_ones) == 0:
            return np.random.randint(1, len(test_results))

        if mode == "var":
            test_results_ones.sort(key=lambda tup: tup[0][1])
        elif mode == "level":
            test_results_ones.sort(key=lambda tup: tup[0][2])
        else:
            raise NoModeError(
                "The mode has not been specified, select between 'var' and 'level'")

        # Return the index of the correct tree
        return test_results_ones[0][1]

    # Tests all the data in bult
    def test(self, data, mode="var"):
        labels = []
        for row in data:
            labels.append(self.test_trees(row, mode))
        return np.array(labels)

    # Save the trees
    def save(self, file_path):
        with open(file_path, 'wb') as handle:
            pickle.dump(self.trees, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return True

    # Loads the trees
    def load(self, file_path):
        with open(file_path, 'rb') as handle:
            self.trees = pickle.load(handle)
        return True
