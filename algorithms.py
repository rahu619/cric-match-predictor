from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics
from sklearn import svm
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


class Algorithms:
    def __init__(self, X, y, test_size=0.25, random_state=42):
        self.X = X
        self.y = y
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False)

        # self.__optimal_hyper_parameters_for_RFC()

    def random_forest_classifier(self):
        """Uses the RFC model and calculates the accuracy"""
        # Creating a classifier
        # clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
        #                              max_depth=150, max_features='sqrt', max_leaf_nodes=None,
        #                              min_impurity_decrease=0.0,
        #                              min_samples_leaf=1, min_samples_split=2,
        #                              min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,
        #                              oob_score=False, random_state=None, verbose=0,
        #                              warm_start=False)

        clf = RandomForestClassifier(
            n_estimators=1400, max_depth=16, random_state=self.random_state)

        # Training the model
        clf.fit(self.X_train, self.y_train)

        # Predicting
        predict_test = clf.predict(self.X_test)

        # calculating accuracy and returning the score
        print(classification_report(self.y_test, predict_test))
        return metrics.accuracy_score(self.y_test, predict_test)

    def random_forest_regressor(self):
        """Uses the RFR model and calculates the accuracy"""

        # Using 300 trees; max-features will be sqrt of the no:of parameters in training dataset
        clf = RandomForestRegressor(
            n_estimators=300, max_features='sqrt', max_depth=5, random_state=self.random_state)

        clf.fit(self.X_train, self.y_train)

        y_pred = clf.predict(self.X_test)

        mean_abs_percentage_error = metrics.mean_absolute_percentage_error(
            self.y_test, y_pred)
        print('mean_abs_percentage_error', mean_abs_percentage_error)
        return round(100*(1 - mean_abs_percentage_error), 2)

    def support_vector_machine(self):
        """Uses the SVM model and calculates the accuracy"""
        scaler = StandardScaler()
        # train_data = scaler.fit_transform(self.X_train)
        clf = svm.SVC(C=10, cache_size=200, class_weight='balanced', coef0=0.0,
                      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                      max_iter=-1, probability=False, random_state=None, shrinking=True,
                      tol=0.001, verbose=False)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        return metrics.accuracy_score(self.y_test, y_pred)

    def KNeighbours(self):
        neighbors = 41
        knn = KNeighborsClassifier(
            n_neighbors=neighbors,
            weights='uniform',
            p=1
        )

        knn = knn.fit(self.X_train, self.y_train)

        y_predict_train = knn.predict(self.X_train)
        y_predict_test = knn.predict(self.X_test)

        accuracy_train = sum(
            y_predict_train == self.y_train) / len(self.y_train)
        accuracy_test = sum(y_predict_test == self.y_test) / len(self.y_test)

        print(
            "Neighbours:",
            "%3d" % neighbors,
            " Trained Accuracy:",
            "%6.2f" % (100*accuracy_train),
            " Test Accuracy:",
            "%6.2f" % (100*accuracy_test)
        )
        return (100*accuracy_train)

    def neural(self):
        mlp = MLPClassifier(hidden_layer_sizes=(8, 8, 8),
                            activation='relu', solver='adam', max_iter=500)

        mlp.fit(self.X_train, self.y_train)

        predict_train = mlp.predict(self.X_train)
        predict_test = mlp.predict(self.X_test)
        # print(confusion_matrix(self.y_train, predict_train))
        # print(classification_report(self.y_train, predict_train))
        print(confusion_matrix(self.y_test, predict_test))
        print(classification_report(self.y_test, predict_test))

    def __optimal_hyper_parameters_for_RFC(self):
        rfc = RandomForestClassifier()
        parameters = {
            "n_estimators": [5, 50, 250, 1000, 1200, 1400],
            "max_depth": [16, 32, 64, 128, 256, None]

        }
        self.__find_optimal_hyper_parameter(rfc, parameters)

    def __find_optimal_hyper_parameter(self, model, parameters):
        """Using GridSearchCV to cross-validate and find the right hyper-parameter values for the model"""
        cv = GridSearchCV(model, parameters, cv=5)
        cv.fit(self.X, self.y.values.ravel())
        results = cv
        print(f'Best parameters are: {results.best_params_}')
        print("\n")
        mean_score = results.cv_results_['mean_test_score']
        std_score = results.cv_results_['std_test_score']
        params = results.cv_results_['params']
        for mean, std, params in zip(mean_score, std_score, params):
            print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')
