from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


class Algorithms:
    def __init__(self, X, y, test_size=0.2, random_state=0):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False)

    def random_forest_classifier(self):

        # Creating a classifier
        clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                     max_depth=150, max_features='auto', max_leaf_nodes=None,
                                     min_impurity_decrease=0.0,
                                     min_samples_leaf=1, min_samples_split=2,
                                     min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,
                                     oob_score=False, random_state=None, verbose=0,
                                     warm_start=False)

        # Training the model
        clf.fit(self.X_train, self.y_train)

        # Predicting
        y_pred = clf.predict(self.X_test)

        # calculating accuracy and returning the score
        return metrics.accuracy_score(self.y_test, y_pred)
