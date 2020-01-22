import numpy as np
from sklearn.linear_model import LogisticRegression

n_features = 20



def reduce_features(specimen, features):
    selected_features = np.flatnonzero(specimen)
    reduced_features = features[:, selected_features]
    return reduced_features


def pop_fitness(X_train, X_test, y_train, y_test, population):
    accuracies = np.zeros(population.shape[0])
    idx = 0
    for specimen in population:
        X_train_reduced = reduce_features(specimen, X_train)
        X_test_reduced = reduce_features(specimen, X_test)

        logreg = LogisticRegression(solver='lbfgs')
        logreg.fit(X_train_reduced, y_train)
        accuracies[idx] = logreg.score(X_test_reduced, y_test)
        idx = idx + 1
    return accuracies
