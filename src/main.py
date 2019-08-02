from preprocessing import build_data_sets
from training import train_trees

import os
import sys


def main():
    # change directory to the current file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # build the data sets to train the trees on
    # build_data_sets()

    # train the trees and run analysis
    # Options: directories=["clean", "noisy"]
    #          splits=10 # number of splits in the data directory
    #          final_forests=True # create final trees
    #          load=True # If trees are already created, load them instead of training new trees
    #          n_classes=6
    #          forest_mode="var" # Which mode to use for testing the trees
    #          save=True

    for mode in ["var", "level"]:
        train_trees(load=True, save=False, forest_mode=mode)

    return 0


if __name__ == '__main__':
    sys.exit(main())
