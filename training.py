import os
import sys
import copy

import numpy as np

from decision_tree import DecisionTree
from decision_forest import DecisionForest
from postprocessing import Postprocessing
from exceptions import UnevenLabelData, NonExistantTree
from test import loadTrees
from math import sqrt


def confidence_interval(error, z, n):
    return (error - z*sqrt(error*(1-error)/n), error+z*sqrt(error*(1-error)/n))


# Changes the data to represent only that given emotion
def select_emotion(emotion, data):
    datac = copy.deepcopy(data)
    for row in datac:
        if row[-1] == emotion:
            row[-1] = 1
        else:
            row[-1] = 0
    return datac


# Calculates the accuracy on the data
def get_accuracy(results, actual):
    error = 0
    if len(results) != len(actual):
        raise UnevenLabelData("Expected and Predicted data lengths do not match")
    for i in range(len(results)):
        if results[i] != actual[i]:
            error += 1
    return float(1-(error / len(results)))


# Calculates what is the best height for the tree to avoid overfitting on validation set
def get_best_tree_height(tree, validation_data):
    max_accuracy = 0
    level = 0
    if tree.height < 1:
        raise NonExistantTree("Tree is not large enough")
    for height in range(3, tree.height+1):
        data = [x[:-1] for x in validation_data]
        labels = tree.test_bulk(data, height)
        expected_labels = [x[-1] for x in validation_data]
        accuracy = get_accuracy(labels, expected_labels)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            level = height
        print("Accuracy: {:.2f}%, max_accuracy: {:.2f}%, best height: {}".format(
            accuracy*100, max_accuracy*100, level))
    return level


# Trains the decision forest from the decision trees
def train_decision_forest(train_data, validation_data, n_classes=6, data_set="clean"):
    decision_forest = DecisionForest()
    for emotion in range(1, n_classes+1):
        print("\n########## Training Emotion {} ##########\n".format(emotion))
        train_data_emotion = select_emotion(emotion, train_data)
        validation_data_emotion = select_emotion(emotion, validation_data)

        # Create a tree with maximum possible height
        tree = DecisionTree(10000)
        tree.train(train_data_emotion)
        print("Trained maximum height tree")

        print("Testing different heights")
        best_height = get_best_tree_height(tree, validation_data_emotion)
        print("Best height found: {}".format(best_height))

        print("Chosing best height: {}".format(best_height))
        tree = DecisionTree(best_height)
        tree.train(train_data_emotion)
        tree.print_dot("visual/tree_{}_{}.gv".format(data_set, emotion))

        print("Chosing best height: {} and appending this tree to the forest".format(best_height))
        # Append the tree to the forest and return the forest when finished
        decision_forest.trees.append(tree)
    return decision_forest


def load_forest(directory, split):
    if os.path.isfile("forests/decision_forest_{}_{}.forest".format(directory, split)):
        decision_forest = DecisionForest()
        decision_forest.load("forests/decision_forest_{}_{}.forest".format(directory, split))
        print("Loaded decision forest from: {}".format("forest/decision_forest_{}_{}.tree".format(
            directory, split)))
        return decision_forest


# The main function to train and test the decision forests
# Mode fore the testing can be set by changing mode to "var" or "level"
def train_trees(directories=["clean", "noisy"], splits=10, final_forests=True,
                load=True, n_classes=6, forest_mode="var", save=True):
    data_dir = os.path.join("..", "..", "data", "cw1")
    print("Loading base data from directory: {}\n".format(data_dir))

    validation_accuracies = {}
    test_accuracies = {}
    confusion_matrices = []

    # If forests were not
    if not os.path.isdir("forests"):
        os.mkdir("forests")

    for d in directories:
        for i in range(10):
            if not os.path.exists("forests/decision_forest_{0}_{1}.forest".format(d,i)):
                load = False

    if not os.path.isdir("visual"):
        os.mkdir("visual")

    for directory in directories:
        average_test_accuracy = 0
        average_validation_accuracy = 0
        analysis = Postprocessing(n_classes)
        for split in range(splits):
            print("\n########## Split {} ##########\n".format(split))

            train_data = np.load(os.path.join(data_dir, "processed", directory, "{}training.npy".format(
                split)))
            validation_data = np.load(os.path.join(
                data_dir, "processed", directory, "{}validation.npy".format(split)))
            test_data = np.load(os.path.join(data_dir, "processed", directory, "{}test.npy".format(
                split)))

            decision_forest = None
            if load:
                decision_forest = load_forest(directory, split)
            else:
                decision_forest = train_decision_forest(train_data, validation_data)

            val_data = [x[:-1] for x in validation_data]
            val_data_labels = [x[-1] for x in validation_data]
            result_labels = decision_forest.test(val_data, forest_mode)
            validation_accuracy = get_accuracy(result_labels, val_data_labels)
            average_validation_accuracy += validation_accuracy

            t_data = [x[:-1] for x in test_data]
            t_data_labels = [x[-1] for x in test_data]
            result_labels = decision_forest.test(t_data, forest_mode)
            test_accuracy = get_accuracy(result_labels, t_data_labels)
            average_test_accuracy += test_accuracy

            analysis.confusion_gen(t_data_labels, result_labels)
            print("\n\n########## Testing the Forest ##########\n")
            print("\nAccuracy Validation = {:.2f}%".format(validation_accuracy*100))
            print("\nAccuracy Testing = {:.2f}%".format(test_accuracy * 100))
            if save:
                decision_forest.save("forests/decision_forest_{}_{}.forest".format(directory, split))

            del decision_forest
        average_test_accuracy /= 10
        average_validation_accuracy /= 10

        test_accuracies[directory] = average_test_accuracy
        validation_accuracies[directory] = average_validation_accuracy
        confusion_matrices.append(analysis)

    if final_forests:
        for directory in directories:
            decision_forest = DecisionForest()
            train_data = np.load(os.path.join(data_dir, "processed", directory, "{}training.npy".format(0)))
            validation_data = np.load(os.path.join(
                data_dir, "processed", directory, "{}validation.npy".format(0)))
            test_data = np.load(
                os.path.join(data_dir, "processed", directory, "{}test.npy".format(0)))
            train_data = np.concatenate((
                train_data, validation_data), axis=0)
            print("\n\n########## Training final tree for directory: {} ##########\n".format(directory))
            if os.path.isdir("forests") and load:
                decision_forest = loadTrees(directory)
            else:
                decision_forest = train_decision_forest(train_data, test_data, 6, directory)

            val_data = [x[:-1] for x in validation_data]
            val_data_labels = [x[-1] for x in validation_data]
            result_labels = decision_forest.test(val_data, forest_mode)
            accuracy = get_accuracy(result_labels, val_data_labels)
            print("Accuracy on the validation data: {:.2f}%".format(accuracy * 100))
            decision_forest.save("forests/decision_forest_{}.forest".format(directory))

    print("\n\n########## Finished training & Testing ##########\n")
    for accuracy in test_accuracies:
        print("Total Test Accuracy average for {} data: {:.2f}%".format(
            accuracy, test_accuracies[accuracy] * 100))
    print("##################################################")
    for accuracy in validation_accuracies:
        print("Total Validation Accuracy average for {} data: {:.2f}%".format(
            accuracy, validation_accuracies[accuracy] * 100))

    for x in range(len(confusion_matrices)):
        print(confusion_matrices[x].print_stats())
        confusion_matrices[x].plot_confusion_matrix("confusion_matrix_{}_{}".format(
            directories[x], forest_mode))
    print("##################################################")
    for x in test_accuracies:
        conf1, conf2 = confidence_interval(1-test_accuracies[x], 1.96, 100)
        print("Average Accuracy: {:.2f}%\nConfidence Interval: ({:.3f}, {:.3f})".format(
            test_accuracies[x]*100, conf1, conf2))
        print("Interval: {:.2f}% < x < {:.2f}%".format((1-conf1)*100, (1-conf2)*100))
    print("##################################################")


if __name__ == "__main__":
    sys.exit(train_trees(load=False, final_forests=False))
