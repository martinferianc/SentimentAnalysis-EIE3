from exceptions import NonExistantDataType, OutofBoundsError
from decision_forest import DecisionForest


def testTrees(T, X, mode="var"):
    return T.test(X, mode)


def loadTrees(dataType="clean", split=None):
    """Returns a DecisionForest according to the datatype and split that
    was input. If the split is None, the full decision forest is returned.
    The dataType can be 'clean' and 'noisy'."""
    if dataType not in ["clean", "noisy"]:
        raise NonExistantDataType("'"+dataType+"' is not a valid data type")
    df = DecisionForest()
    if split is None:
        df.load("forests/decision_forest_{}.forest".format(dataType))
    else:
        if split < 0 or split > 9:
            raise OutofBoundsError("split '"+split+"' is not available.")
        df.load("forests/decision_forest_{}_{}.forest".format(dataType, split))

    return df
