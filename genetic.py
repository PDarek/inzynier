import numpy as np
from sklearn.linear_model import LogisticRegression

n_features = 20

pop_size = 8 # Population size.
num_parents_mating = 4 # Number of parents inside the mating pool.
mutations = 3 # Number of elements to mutate.

pop_shape = (pop_size, n_features)
new_population = np.random.randint(2, size=pop_shape)
print(new_population.shape)
print(new_population)


def reduce_features(solution, features):
    selected_elements_indices = np.where(solution == 1)[0]
    reduced_features = features[:, selected_elements_indices]
    return reduced_features


def pop_fitness(X_train, X_test, y_train, y_test, logreg, population):
    accuracies = np.zeros(population.shape[0])
    idx = 0

    for sample in population:
        reduced_features = reduce_features(sample,  X_train)
        X_train_data = reduced_features[X_train, :]
        X_test_data = reduced_features[X_test, :]

        logreg.fit(X_train_data, y_train)
        accuracies[idx] = logreg.score(X_test_data, y_test)
        idx = idx + 1
    return accuracies
