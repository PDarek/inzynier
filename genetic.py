import numpy as np
from sklearn.linear_model import LogisticRegression


def reduce_features(creature, features):
    selected_features = np.flatnonzero(creature)
    reduced_features = features[:, selected_features]
    return reduced_features


def pop_fitness(X_train, X_test, y_train, y_test, population):
    accuracies = np.zeros(population.shape[0])
    idx = 0
    for creature in population:
        X_train_reduced = reduce_features(creature, X_train)
        X_test_reduced = reduce_features(creature, X_test)

        logreg = LogisticRegression(solver='lbfgs', max_iter=10000)
        logreg.fit(X_train_reduced, y_train)
        accuracies[idx] = logreg.score(X_test_reduced, y_test)
        idx = idx + 1
    return accuracies

def select_mating_pool(population, fitness, number_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((number_parents, population.shape[1]))
    for parent_num in range(number_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = population[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring, n_mutations=2):
    mutation_idx = np.random.randint(low=0, high=offspring.shape[1], size=n_mutations)
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring.shape[0]):
        # The random value to be added to the gene.
        offspring[idx, mutation_idx] = 1 - offspring[idx, mutation_idx]
    return offspring