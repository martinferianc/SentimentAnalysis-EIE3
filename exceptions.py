class EmptyDataError(Exception):
    """
    Gets thrown when the Data is of size 0, and therefore nothing
    else can be done.
    """
    pass


class NoBestFeatureError(Exception):
    """
    Gets thrown when no best feature could be found.
    """
    pass


class NoModeError(Exception):
    """
    The mode for testing has not been specified
    """
    pass


class NoForestError(Exception):
    """
    The mode for testing has not been specified
    """
    pass


class RateCalcError(Exception):
    """
    Thrown when dividing by zero during rate calculations
    """
    pass


class OutofBoundsError(Exception):
    """
    Thrown when generating the confusion matrix
    """
    pass


class NonExistantDataType(Exception):
    """
    Thrown when the data type is not correct
    """
    pass


class UnevenLabelData(Exception):
    """
    Thrown when data is uneven
    """
    pass


class NonExistantTree(Exception):
    """
    Thrown when the tree hasn't built properly
    """
    pass
