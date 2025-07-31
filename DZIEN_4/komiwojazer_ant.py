import numpy as np
import matplotlib.pyplot as plt
import random

# Parametry
NUM_CITIES = 20
NUM_ANTS = 30
NUM_ITER = 100
ALPHA = 1.0       # Wpływ feromonu
BETA = 5.0        # Wpływ odległości
EVAPORATION = 0.5
Q = 100           # Stała ilość feromonu

# Losowe współrzędne miast
cities = np.random.rand(NUM_CITIES, 2)

# Odległości między miastami
def distance(a, b):
    return np.linalg.norm(a - b)

dist_matrix = np.array([[distance(a, b) for b in cities] for a in cities])
pheromone = np.ones((NUM_CITIES, NUM_CITIES))

# Algorytm mrówkowy
best_length = float("inf")
best_path = []

for iteration in range(NUM_ITER):
    all_paths = []
    all_lengths = []

    for _ in range(NUM_ANTS):
        path = [random.randint(0, NUM_CITIES-1)]
        visited = set(path)

        for _ in range(NUM_CITIES - 1):
            current = path[-1]
            probabilities = []
            for j in range(NUM_CITIES):
                if j not in visited:
                    tau = pheromone[current][j] ** ALPHA
                    eta = (1.0 / dist_matrix[current][j]) ** BETA
                    probabilities.append((j, tau * eta))
            total = sum(p for _, p in probabilities)
            r = random.uniform(0, total)
            cumulative = 0.0
            for city, prob in probabilities:
                cumulative += prob
                if r <= cumulative:
                    path.append(city)
                    visited.add(city)
                    break

        path.append(path[0])  # powrót do startu
        length = sum(dist_matrix[path[i]][path[i+1]] for i in range(NUM_CITIES))
        all_paths.append(path)
        all_lengths.append(length)

        if length < best_length:
            best_length = length
            best_path = path

    # Aktualizacja feromonów
    pheromone *= (1 - EVAPORATION)
    for path, length in zip(all_paths, all_lengths):
        for i in range(NUM_CITIES):
            a, b = path[i], path[i + 1]
            pheromone[a][b] += Q / length
            pheromone[b][a] += Q / length

    print(f"Iteracja {iteration+1}: Najlepsza długość = {best_length:.2f}")

# Wizualizacja
x = [cities[i][0] for i in best_path]
y = [cities[i][1] for i in best_path]
plt.plot(x, y, marker='o')
plt.title("Najlepsza znaleziona ścieżka (ACO)")
plt.show()
