#maksymalizujemy funkcję f(x) = x^2 dla x -> [0,31] (zakodowanego binarnie)

import random

from Tools.scripts.stable_abi import binutils_get_exported_symbols


def fitness(x):
    return x**2

def select(population):
    return max(population, key=fitness)

def crossover(p1, p2):
    point = random.randint(1, 4)
    mask = (1<<point) - 1
    return (p1 & ~mask) | (p2 & mask)

def mutate(x, mutation_rate=0.1):
    if random.random() < mutation_rate:
        bit = 1 << random.randint(0, 4)
        x ^= bit
    return x

#inicjalizacja populacji
population = [random.randint(0,31) for _ in range(10)]

#główna pętla
for generation in range(20):
    new_population = []
    for _ in range(len(population)):
        p1,p2 = random.sample(population, 2)
        child = crossover(p1, p2)
        child = mutate(child)
        new_population.append(child)
    population = new_population
    best = select(population)
    print(f"Generacja {generation}: best individual = {select(population)}, fitness = {best}")
