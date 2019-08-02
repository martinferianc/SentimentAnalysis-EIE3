import scipy.io
import numpy as np
import copy
import os

BASE_DIR = "data/cw1/"

# Load the data and convert it into numpy matrices


def load_data(file_path):
    data = scipy.io.loadmat(file_path)
    X = data["x"]
    Y = data["y"]
    return (X, Y)


def load_data_set(data_quality, fold=0, d_set="training", base_dir=BASE_DIR + "processed"):
    file_path = base_dir + "/{0}/{1}{2}.npy".format(data_quality, fold, d_set)
    print("Loading file: {0}".format(file_path))
    return np.load(file_path)


def n_fold(data_in, target_dir, n=10, validation=0.1, test=0.1):
    # Load the target data into two vars
    X, Y = data_in
    # Append Y to the end of X
    data = np.append(X, Y, axis=1)
    data = np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)
    tseg = int(len(X) * test)
    vseg = int((len(X) - tseg) * validation)
    for i in range(0, n):
        # Split the data into test and non-test data
        tsplit_start = (i * tseg) % len(data)
        tsplit_end = (i * tseg + tseg) % len(data)
        test_data = data[tsplit_start:tsplit_end, :]
        # Split remaining data into training and validation
        r_data = np.vstack((data[0:tsplit_start, :], data[tsplit_end:, :]))
        vsplit_start = (i * vseg) % len(r_data)
        vsplit_end = (i*vseg + vseg)  % len(r_data)
        validation_data = r_data[vsplit_start:vsplit_end, :]
        training_data = np.vstack((r_data[0:vsplit_start, :], r_data[vsplit_end:, :]))

        #r_data[vseg:, :]
        print("Test data is {0}.".format(test_data.shape))
        print("Validation data is {0}.".format(validation_data.shape))
        print("Training data is {0}.".format(training_data.shape))
        np.save(target_dir + "{0}test.npy".format(i), test_data)
        np.save(target_dir + "{0}training.npy".format(i), training_data)
        np.save(target_dir + "{0}validation.npy".format(i), validation_data)

def build_data_sets(types = ["noisy", "clean"]):
    for t in types:
        if not os.path.exists(os.path.join(BASE_DIR, "processed", t)):
            os.makedirs(os.path.join(BASE_DIR, "processed", t))
        file_path = (BASE_DIR + "raw/{}data_students.mat").format(t)
        print("Loading from file_path {}".format(file_path))
        data = load_data(file_path)
        if t == "noisy":
            # Replace missing values with None's

            pass
        n_fold(copy.deepcopy(data), (BASE_DIR + "processed/{0}/").format(t))


if __name__ == "__main__":
    build_data_sets()
