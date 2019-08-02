class Node:

    """
    A node in the tree that stores the attribute used to
    split on
    """

    def __init__(self, attribute, level, leaf=False, label=-1, label_exp=-1, label_var=-1):
        self.attribute = attribute
        self.branches = []
        self.level = level
        self.leaf = leaf
        self.label = label
        self.label_exp = label_exp
        self.label_var = label_var

    # Gets a target branch
    def get_branches(self, index=None):
        if index is None:
            return self.branches

        elif index >= len(self.branches) or index < 0:
            raise Exception("Invalid Index")

        return self.branches[index]

    # Determines if it is a leaf
    def is_leaf(self):
        return self.leaf

    # Append to the branch
    def append_branch(self, node):
        self.branches.append(node)
