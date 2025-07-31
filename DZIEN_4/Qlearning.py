import numpy as np
import tensorflow as tf
import random
from collections import deque
import matplotlib.pyplot as plt

# Parametry środowiska
GRID_SIZE = 5
STATE_SIZE = 2
ACTION_SIZE = 4
ACTIONS = ['up', 'down', 'left', 'right']
TARGET_STATE = (4, 4)

# Parametry treningu
NUM_EPISODES = 500
MAX_STEPS = 100
ALPHA = 0.001
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 2000

# Funkcja nagrody
def get_reward(state):
    return 1.0 if state == TARGET_STATE else 0.0

# Przejście do nowego stanu
def take_action(state, action):
    x, y = state
    if action == 0: x = max(0, x - 1)
    elif action == 1: x = min(GRID_SIZE-1, x + 1)
    elif action == 2: y = max(0, y - 1)
    elif action == 3: y = min(GRID_SIZE-1, y + 1)
    return (x, y)

# Sieć neuronowa Q
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(STATE_SIZE,)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(ACTION_SIZE, activation='linear')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=ALPHA),
              loss='mse')

# Bufor pamięci
memory = deque(maxlen=MEMORY_SIZE)
def state_to_input(state):
    return np.array(state).reshape(1, -1)

rewards_per_episode = []
epsilon = EPSILON

# Trening
for episode in range(NUM_EPISODES):
    state = (0, 0)
    total_reward = 0

    for step in range(MAX_STEPS):
        if np.random.rand() < epsilon:
            action = np.random.randint(ACTION_SIZE)
        else:
            q_values = model.predict(state_to_input(state), verbose=0)
            action = np.argmax(q_values[0])

        next_state = take_action(state, action)
        reward = get_reward(next_state)
        done = reward > 0

        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if done:
            break

        # Trening na próbce mini-batch
        if len(memory) >= BATCH_SIZE:
            minibatch = random.sample(memory, BATCH_SIZE)
            states = np.array([s for (s, a, r, ns, d) in minibatch])
            next_states = np.array([ns for (s, a, r, ns, d) in minibatch])
            q_targets = model.predict(states, verbose=0)
            q_next = model.predict(next_states, verbose=0)

            for i, (s, a, r, ns, d) in enumerate(minibatch):
                target = r if d else r + GAMMA * np.max(q_next[i])
                q_targets[i][a] = target

            model.fit(states, q_targets, epochs=1, verbose=0)

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    rewards_per_episode.append(total_reward)

# Wykres
plt.plot(rewards_per_episode)
plt.title("Nagroda w kolejnych epizodach (DQN z TensorFlow)")
plt.xlabel("Epizod")
plt.ylabel("Nagroda")
plt.grid()
plt.show()
