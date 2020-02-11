import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import time
import genetic

start_time = time.time()
# make classification parameters
n_samples = 10000
n_features = 63
n_informative = 50
n_redundant = 10
n_repeated = 3

# genetic algorithm parameters
population_size = 80  # Population size.
n_parents = population_size // 2  # Number of parents inside the mating pool.
n_mutations = 3  # Number of elements to mutate.
n_generations = 60  # Number of generations.

X, y = make_classification(n_samples, n_features, n_informative, n_redundant, n_repeated)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
population_shape = (population_size, n_features)
# Starting population

new_population = np.random.randint(2, size=population_shape)

best_outputs = []  # Table for best outputs score in every generation

raw_logreg = LogisticRegression(penalty='none', solver='newton-cg', max_iter=1000, random_state=42)
raw_logreg.fit(X_train, y_train)
y_pred = raw_logreg.predict(X_test)

raw_logreg_score = raw_logreg.score(X_test, y_test)
raw_logit_roc_auc = roc_auc_score(y_test, raw_logreg.predict(X_test))
raw_fpr, raw_tpr, raw_thresholds = roc_curve(y_test, raw_logreg.predict_proba(X_test)[:, 1])

print('Fitness of raw logistic regression : ', raw_logreg_score)

for generation in range(n_generations):
    print("Generation : ", generation+1)
    # Measuring the fitness of each chromosome in the population.
    calculation_time = time.time()
    fitness = genetic.pop_fitness(X_train, X_test, y_train, y_test, new_population)
    print('Generation calculation time : ', time.time()-calculation_time)
    best_outputs.append(np.max(fitness))

    print('Number of creatures with best fitness : ', (fitness == np.max(fitness)).sum())

    # The best result in the current generation.
    print("Best result : ", best_outputs[-1])

    # Selecting the best parents for mating.
    parents = genetic.select_mating_pool(new_population, fitness, n_parents)

    # Generating next generation.
    offspring_crossover = genetic.crossover(parents, offspring_size=(population_shape[0]-parents.shape[0], n_features))

    # Adding some variations to the offspring using mutation.
    offspring_mutation = genetic.mutation(offspring_crossover, n_mutations)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

# Getting the best solution after finishing all generations.
# At first, the fitness is calculated for each solution in the final generation.
fitness = genetic.pop_fitness(X_train, X_test, y_train, y_test, new_population)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.where(fitness == np.max(fitness))[0]
best_match_idx = best_match_idx[0]

best_solution = new_population[best_match_idx, :]
best_solution_indices = np.flatnonzero(best_solution)
best_solution_num_elements = best_solution_indices.shape[0]
best_solution_fitness = fitness[best_match_idx]

print("best_match_idx : ", best_match_idx)
print("best_solution : ", best_solution)
print("Selected indices : ", best_solution_indices)
print("Number of selected elements : ", best_solution_num_elements)
print("Best solution fitness : ", best_solution_fitness)

plt.figure()
plt.plot(best_outputs, label='Genetic algorithm')
plt.axhline(y=raw_logreg_score, xmin=0, xmax=n_generations, color='r', linestyle='--', label='Raw logit')
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend(loc="lower right")
plt.show()

"""

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression: {:.2f}'.format(logreg.score(X_test, y_test)))
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('RO characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC')
plt.show()

"""

print("Program took %s seconds " % (time.time() - start_time))
