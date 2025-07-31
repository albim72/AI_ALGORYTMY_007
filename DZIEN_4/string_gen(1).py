import random
import string

from maksymalizacja import mutate

TARGET = "HELLO WORLD"
POP_SIZE = 100
MUTATION_RATE = 0.05
GENES = string.ascii_uppercase + " "

def fitness(individual):
    return sum([1 for i,c in enumerate(individual) if c == TARGET[i]])

def mutate(ind):
    return "".join(
        c if random.random() > MUTATION_RATE else random.choice(GENES)
        for c in ind
    )

def crossover(p1, p2):
    return "".join(random.choice([c1,c2]) for c1,c2 in zip(p1,p2))

population = ["".join(random.choices(GENES,k=len(TARGET))) for _ in range(POP_SIZE)]

#pętla główna
for gen in range(1000):
    population.sort(key=lambda x:fitness(x), reverse=True)
    best = population[0]
    print(f"Generation {gen}: best individual = {best}, fitness = {fitness(best)}")
    if best == TARGET:
        break

    parents = population[:10]
    new_population = []
    for _ in range(POP_SIZE):
        p1,p2 = random.sample(parents, 2)
        child = mutate(crossover(p1, p2))
        new_population.append(child)
    population = new_population
