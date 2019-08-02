# Sentiment analysis for grayscale small pictures of faces by decision forests

This project was a part of coursework for module CO395 Introduction to Machine Learning at Imperial College London

## Specification

The task was to accept the 2D grayscale image as a 1D vector of features which have been used both for training and testing of decision forests which could classify emotions.

When training the decision trees on the data, we train six different trees that become binary
classifiers on one specific emotion. These are then combined into a decision forest which is then
used to classify the test data. The decision forests are then stored [here](forests/) after
they have been generated and can therefore easily be loaded again.

## Testing

To test the trees, the trained trees should first be loaded from the disk or trained on the data.

The trained decision forests should be stored in the folder called `forests`. If that directory is
not present, the trees will have to be retrained by following the guide on how to do that in the training section of the README.

The `loadTrees` function in [test.py](test.py) can be used to load a tree from the `forests`
directory. This function first takes in the data type, which can be either 'noisy' or 'clean'. As a
second argument it takes in the split from the cross validation that should be loaded, which can be
a number from 0-9. If the split is of type None, or not present, then the full decision forest is returned.

Once the forest is loaded, it can be tested by using the `testTrees` function in [test.py](test.py).
This function takes in the forest that was loaded earlier and the test data that the forest should be
tested on. It then returns the predicted labels to the input data.

There is a `load_data` function which when passed the file path to a .mat data file will return a tuple of the data and corresponding labels.

An example of how to use these functions can be seen below:
``` python
from preprocessing import load_data

X,Y = load_data("path/to/data_testing.mat")

from test import testTrees, loadTrees

forest = loadTrees("clean")
predicted_labels = testTrees(forest, X)
```

`testTrees` can also take in an optional parameter, which determines with which hyperparameter the
decision forest chooses the right label. Two different parameters can be used for the label identification
one being 'var' and the other being 'level'. By default, the decision forest will use the variance,
by using the 'var' flag, to choose the correct label. These can be set as follows.

`predicted_labels = testTrees(forest, X, "var")`

or

`predicted_labels = testTrees(forest, X, "level")`

## Training

To train the trees and store them, the main.py python file can be executed, after all the requirements have
been installed using pip3. From the root of the repository, the following commands will generate the
trees.

``` shell
pip install -r requirements.txt
python main.py
```

This will perform cross-validation on all of the provided data, training all the trees required for this analysis, as well as training trees on all of the data. The .forest files `decision_forest_clean.forest` and `decision_forest_noisy.forest` are the final decision forests to be used on identifying real data.

## Credits

Yann Herklotz, George Punter, Divyansh Manocha and Martin Ferianc at Imperial College London, 2018.
