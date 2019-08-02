import numpy as np
import unittest
from math import log
from decision_tree import DecisionTree


class TestDecisionTree(unittest.TestCase):

    """
    Test the DecisionTree methods for functionality
    """

    def setUp(self):
        """Set up the initial variables"""
        self.data = [[1, 1, 0, 1],
                     [0, 1, 1, 0],
                     [1, 1, 1, 1],
                     [1, 0, 1, 0],
                     [1, 1, 1, 1],
                     [1, 0, 0, 1]]
        self.data = np.array(self.data)

    def test_calculate_entropy(self):
        """Test if the entropy is calculated correctly"""
        expected_entropy = - 2 / 3 * log(2 / 3, 2) - 1 / 3 * log(1 / 3, 2)
        dt = DecisionTree()
        entropy = dt.calculate_entropy(self.data.transpose()[-1])

        self.assertEqual(expected_entropy, entropy)

    def test_choose_best_feature(self):
        """Test if the tree chooses the correct attribute"""
        dt = DecisionTree()
        total_entropy = dt.calculate_entropy(self.data.transpose()[-1])
        ent0 = total_entropy - 5 / 6 * \
            (-4 / 5 * log(4 / 5, 2) - 1 / 5 * log(1 / 5, 2))
        # ent1 = total_entropy - 2 / 3 * (-3 / 4 * log(3 / 4, 2) - 1 / 4 * log(
        #     1 / 4, 2)) - 1 / 3 * (-1 / 2 * log(1 / 2, 2) - 1 / 2 * log(1 / 2, 2))
        # ent2 = total_entropy - 2 / 3 * \
        #     (-1 / 2 * log(1 / 2, 2) - 1 / 2 * log(1 / 2, 2))
        i, E = dt.choose_best_feature(self.data)

        self.assertEqual(i, 0)
        self.assertEqual(E, ent0)

    def test_decision_tree(self):
        """Test if the decision tree works"""
        dt = DecisionTree()
        dt.max_level = 100

        random_attr = np.random.randint(2, size=(100, 40))
        random_label = np.random.randint(2, size=(100, 1))

        random_test = np.append(random_attr, random_label, axis=1)

        dt.train(random_test)

        outcome = dt.test_bulk(random_attr)
        expected_outcome = random_label.transpose()[0]
        for x in range(len(outcome)):
            self.assertEqual(outcome[x], expected_outcome[x])


if __name__ == "__main__":
    unittest.main()
