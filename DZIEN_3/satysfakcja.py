import numpy as np
import tensorflow as tf
import random

# 1. Generujemy dane treningowe (symulacja)
# [features: [funkcje, cena, gwarancja]]
def generate_data(n=500):
    X = np.random.rand(n, 3)  # cechy w zakresie 0–1
    y = 5 * X[:, 0] - 3 * X[:, 1] + 2 * X[:, 2] + np.random.normal(0, 0.1, n)
    return X, y

X_train, y_train = generate_data()

# 2. Tworzymy prosty model regresyjny
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)  # regresja
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, verbose=0)

# 3. Heurystyczne przeszukiwanie (hill climbing) – szukamy najlepszej konfiguracji
def heuristic_search(model, steps=1000, step_size=0.05):
    current = np.random.rand(3)  # losowy punkt startowy
    current_score = model.predict(current.reshape(1, -1))[0][0]

    for _ in range(steps):
        # generuj sąsiadów w pobliżu
        neighbours = [
            np.clip(current + np.random.uniform(-step_size, step_size, size=3), 0, 1)
            for _ in range(10)
        ]
        neighbour_scores = [model.predict(n.reshape(1, -1))[0][0] for n in neighbours]

        # wybierz najlepszego sąsiada
        best_idx = np.argmax(neighbour_scores)
        if neighbour_scores[best_idx] > current_score:
            current = neighbours[best_idx]
            current_score = neighbour_scores[best_idx]
        else:
            break  # lokalne maksimum

    return current, current_score

# 4. Uruchamiamy poszukiwania
best_config, predicted_score = heuristic_search(model)
print(f"Najlepsza konfiguracja: {best_config}")
print(f"Oczekiwana satysfakcja klienta: {predicted_score:.2f}")
