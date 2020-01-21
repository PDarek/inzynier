import numpy as np


n_features = 20

pop_size = 8 # Population size.
num_parents_mating = 4 # Number of parents inside the mating pool.
mutations = 3 # Number of elements to mutate.

pop_shape = (pop_size, n_features)
new_population = np.random.randint(2, size=pop_shape)
print(new_population.shape)
print(new_population)
