import numpy as np
import mnist_digits_nn as md
import matplotlib.pyplot as plt

POP_SIZE = 20
N_GEN = 100
IM_SIZE = (28, 28)


def fitness(model, population):
    predictions = [i[-1] for i in model.predict(population)]
    return predictions

def select_parents(model, population):
    f = fitness(model, population)
    ind = np.argpartition(f, -2)[-2:]
    return [population[ind[0]], population[ind[1]]]

def crossover(parents):
    children = []
    child = np.zeros(shape=IM_SIZE)
    for i in range(0, POP_SIZE - 2):
        child = parents[0] + parents[1]
        for j in range(0, IM_SIZE[0]):
            for k in range(0, IM_SIZE[1]):
                if parents[0][j][k] == parents[1][j][k]:
                    child[j][k] = parents[0][j][k]
                else:
                    if np.random.randint(2, size=1)[0] == 0:
                        child[j][k] = parents[0][j][k]
                    else:
                        child[j][k] = parents[1][j][k]
        children.append(child)
    return children

def mutate(population):
    for p in population:
        for i in range(0, IM_SIZE[0]):
            for j in range(0, IM_SIZE[1]):
                if np.random.rand(1)[0] < 0.010:
                    if p[i][j] == 0:
                        p[i][j] = 1
                    else:
                        p[i][j] = 0
    return population


population = np.random.randint(2, size=(20, 28, 28))  # Create initial random population
model = md.get_model()
for gen in range(0, N_GEN):
    parents = select_parents(model, population)  # Parents selection
    children = mutate(crossover(parents))  # Crossover & mutation
    population = np.array(parents + children)
    print("GEN ", gen)
    print("MAX fitness: ", max(fitness(model, population)))


plt.imshow(population[0])
plt.show()
print(population[0])
print(model.predict(np.array([population[0]])))
    