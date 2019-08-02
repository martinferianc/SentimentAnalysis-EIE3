from exceptions import EmptyDataError, NoBestFeatureError
from math import log
import numpy as np
from node import Node
import pickle
from graphviz import Digraph
import copy


class DecisionTree:
    """
    Decision Tree algorithm for machine learning and
    binary classification.
    """

    def __init__(self, max_level=5):
        # Root node
        self.root = None
        # Maximum level of the tree
        self.max_level = max_level
        self.height = -1

    # Calculates entropy based on a list of labels (0,1), returns final entropy
    def calculate_entropy(self, row):
        if len(row) == 0:
            raise EmptyDataError("No data left in row")
        distinct_row = list(set(row))
        total_count = {}
        for dr in distinct_row:
            total_count[dr] = 0
            for x in row:
                if(x == dr):
                    total_count[dr] += 1
        E = 0
        for key, value in total_count.items():
            E -= value / len(row) * log(value / len(row), 2)
        return E

    # Takes the data in its current shape and calculates the
    # best attribute to do the split on
    def choose_best_feature(self, data):
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise EmptyDataError("No data left in data")
        row = [feature_set[-1] for feature_set in data]
        E = self.calculate_entropy(row)

        # Initialize the best possible gain and index
        best_gain = 0.0
        best_feature_index = -1

        # Calculate the entropy for each column in the data
        for attribute in range(data.shape[1] - 1):
            unique_attribute_values = set(
                [feature_set[attribute] for feature_set in data])
            new_E = 0.0
            for value in unique_attribute_values:
                new_data = self.split_data(data, attribute, value)
                p = len(new_data) / float(len(data))
                row = [feature_set[-1] for feature_set in new_data]
                new_E += p * self.calculate_entropy(row)
            if (E - new_E >= best_gain):
                best_gain = E - new_E
                best_feature_index = attribute

        return (best_feature_index, best_gain)

    # Main function to do the training of the tree
    def train(self, data):
        if data.size == 0:
            raise EmptyDataError("No data left in data")
        data_deepcopy = copy.deepcopy(data)
        self.n_classes = len(set([feature_set[-1]
                                  for feature_set in data_deepcopy]))
        indices = [x for x in range(0, data_deepcopy.shape[1] - 1)]
        self.root = self.create_tree(data_deepcopy, 0, indices)

    def print_tree(self, spacing=4):
        def print_tree_rec(tree, value):
            print("{0}{5} T({4}) = a: {1}, v: {2}, l: {3}".format(
                " " * tree.level * spacing, tree.attribute, tree.label, tree.leaf, tree.level, value))
            for branch, val in tree.branches:
                print_tree_rec(branch, val)
        print_tree_rec(self.root, 0)

    def print_dot(self, file_path="tree.png"):
        def print_dot_rec(dot, tree, value, identifier):
            if tree.leaf:
                dot.node("{}".format(identifier), "label: {}\nvar: {:.3f}\nlevel: {}".format(
                    tree.label, tree.label_var, tree.level), style="filled", fillcolor="green")
            elif tree.level == 0:
                dot.node("{}".format(
                    identifier), "{}".format(tree.attribute), style="filled", fillcolor="red")
            else:
                dot.node("{}".format(identifier), "{}".format(tree.attribute))
            orig_id = identifier
            for branch, val in tree.branches:
                identifier += 1
                dot.edge("{}".format(orig_id), "{}".format(
                    identifier), label=str(val))
                identifier = print_dot_rec(dot, branch, val, identifier)
            return identifier

        dot = Digraph(comment="Decision Tree")
        print_dot_rec(dot, self.root, 0, 0)
        dot.format = 'png'
        dot.render(file_path)

    # This function removes the data not corresponding to a value in a given column (axis)
    def split_data(self, data, axis, value):
        if data.size == 0:
            raise EmptyDataError("No data left in data")
        new_data = []
        for feature_set in data:
            if feature_set[axis] == value:
                new_feature_set = np.array(
                    np.delete(feature_set, obj=axis), copy=True)
                new_data.append(new_feature_set)
        new_data = np.array(new_data, copy=True)
        return new_data

    # Main function to create a tree
    def create_tree(self, data, level, indices):
        if level > self.height:
            self.height = level

        # Base cases
        # if level is more than the max level and have no data return random label
        if data.size == 0:
            # Stop creating the Tree
            label = np.random.randint(0, self.n_classes)
            node = Node(-2, level, leaf=True, label=label,
                        label_exp=label, label_var=1)
            return node

        # If we have reached the maximum level vote on the label counts to assign the label
        # Remember also the expected value for reference
        labels = np.array([feature_set[-1] for feature_set in data])
        label_exp = np.mean(labels)
        label_var = np.var(labels)
        counts = np.bincount(labels)
        # Label with the highest count
        label = np.argmax(counts)

        # If the variance in the data is zero then, all the labels are the same and we do not need to split further
        if label_var == 0.0:
            node = Node(-1, level, leaf=True, label=label,
                        label_exp=label_exp, label_var=label_var)
            return node

        if level >= self.max_level:
            node = Node(-2, level, leaf=True, label=label,
                        label_exp=label_exp, label_var=label_var)
            return node

        # Recursive case
        best_index, gain = self.choose_best_feature(data)
        if best_index == -1:
            node = Node(-1, level, leaf=True, label=label,
                        label_exp=label_exp, label_var=label_var)
            return node

        # Retrieve the actual index for this node
        actual_index = indices[best_index]
        node = Node(actual_index, level, leaf=False,
                    label=label, label_exp=label_exp,
                    label_var=label_var)
        distinct_elements = list(
            set([feature_set[best_index] for feature_set in data]))
        for element in distinct_elements:
            new_data = self.split_data(data, best_index, element)
            new_indices = copy.deepcopy(indices)
            new_indices.remove(actual_index)
            next_node = self.create_tree(new_data, level + 1, new_indices)
            node.append_branch((next_node, element))
        return node

    def prune(self):
        def prune_tree(tree, accuracy):
            pass

    # Pickles the tree
    def save_tree(self, file_path):
        with open(file_path, 'wb') as handle:
            pickle.dump(self.root, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return True

    # Loads the tree
    def load_tree(self, file_path):
        with open(file_path, 'rb') as handle:
            self.root = pickle.load(handle)
        return True

    def test_bulk(self, data, level=10000):
        result = []
        for row in data:
            result.append(self.test(row, level)[0])
        return result

    # Return a label, variance and a level of the resultant node just for one feature set
    def test(self, feature_set, level=10000):
        if feature_set.size == 0:
            raise EmptyDataError("No data passed to test")
        root = self.root
        while (not root.leaf) and (root.level < level):
            for branch, val in root.branches:
                if feature_set[root.attribute] == val:
                    root = branch
                    break
            else:
                return (root.label, root.label_var, root.level)
        return (root.label, root.label_var, root.level)
