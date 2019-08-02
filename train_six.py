import numpy as np
import os

from decision_tree import DecisionTree
from training import get_accuracy, get_best_tree_height


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data_dir = "../../data/cw1/"
    test_accuracies = {}
    validation_accuracies = {}
    if not os.path.isdir("trees"):
        os.mkdir("trees")
    name = "tree_{}_{}.tree"
    for d in ["clean", "noisy"]:
        average_validation_accuracy = 0
        average_test_accuracy = 0
        for s in range(10):
            train_data = np.load(os.path.join(data_dir, "processed", d, "{}training.npy".format(
                s)))
            validation_data = np.load(os.path.join(
                data_dir, "processed", d, "{}validation.npy".format(s)))
            test_data = np.load(os.path.join(data_dir, "processed", d, "{}test.npy".format(
                s)))

            dt = DecisionTree(100)
            if os.path.isfile("trees/"+name.format(d, s)):
                dt.load_tree("trees/"+name.format(d, s))
            else:
                dt = DecisionTree(100)
                dt.train(train_data)

                x = get_best_tree_height(dt, validation_data)

                dt = DecisionTree(x)
                dt.train(train_data)

            val_data = [x[:-1] for x in validation_data]
            val_data_labels = [x[-1] for x in validation_data]
            result_labels = dt.test_bulk(val_data)
            validation_accuracy = get_accuracy(result_labels, val_data_labels)
            average_validation_accuracy += validation_accuracy

            t_data = [x[:-1] for x in test_data]
            t_data_labels = [x[-1] for x in test_data]
            result_labels = dt.test_bulk(t_data)
            test_accuracy = get_accuracy(result_labels, t_data_labels)
            average_test_accuracy += test_accuracy

            dt.save_tree("trees/"+name.format(d, s))

            dt.print_dot("visual/"+name.format(d, s))
        average_test_accuracy /= 10
        average_validation_accuracy /= 10

        test_accuracies[d] = average_test_accuracy
        validation_accuracies[d] = average_validation_accuracy

    print("##################################################")
    print("Finished training")
    print(test_accuracies)


if __name__ == '__main__':
    main()
