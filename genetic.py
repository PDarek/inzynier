import numpy as np


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


def pop_fitness(X_train, X_test, y_train, y_test, algorithm, population):
    accuracies = numpy.zeros(population.shape[0])
    idx = 0

