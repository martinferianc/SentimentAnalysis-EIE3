from sklearn import tree
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

data_dir = os.path.join("..", "..", "data", "cw1")
train_data = np.load(os.path.join(data_dir, "processed", "noisy", "{}training.npy".format(0)))
test_data = np.load(os.path.join(data_dir, "processed", "noisy", "{}test.npy".format(0)))

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data[:, :-1], train_data[:, -1].transpose())

trained_res = clf.predict(test_data[:, :-1])
real_pred = test_data[:, -1].transpose()

s = 0
for i in range(len(real_pred)):
    if real_pred[i] == trained_res[i]:
        s += 1

print("Accuracy:", s / len(real_pred))

print(pd.crosstab(real_pred, trained_res, rownames=['True'], colnames=['Predicted']).apply(lambda r: 100.0 * r/r.sum()))

print(classification_report(real_pred, trained_res))
