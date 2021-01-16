from sklearn import tree
from sklearn import ensemble

models = {
    "decision_tree_gini": tree.DecisionTreeClassifier(
        criterion="gini",
        max_depth=10
    ),
    "decision_tree_entropy": tree.DecisionTreeClassifier(
        criterion="entropy",
        max_depth=10
    ),
    "rf": ensemble.RandomForestClassifier()
}