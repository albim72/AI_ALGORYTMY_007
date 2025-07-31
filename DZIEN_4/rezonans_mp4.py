import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parametry systemu
GRID_SIZE = 50
NUM_AGENTS = 200
STEPS = 100

# Inicjalizacja agentów
agents = []
for _ in range(NUM_AGENTS):
    x, y = np.random.randint(0, GRID_SIZE, 2)
    phase = np.random.rand() * 2 * np.pi
    agents.append({'x': x, 'y': y, 'phase': phase})

# Funkcja aktualizacji
def update_agents():
    for agent in agents:
        neighbors = [a for a in agents if abs(a['x'] - agent['x']) <= 3 and abs(a['y'] - agent['y']) <= 3 and a != agent]
        if neighbors:
            avg_phase = np.mean([n['phase'] for n in neighbors])
            agent['phase'] += 0.1 * (avg_phase - agent['phase'])

        dx = np.cos(agent['phase']) * 0.5
        dy = np.sin(agent['phase']) * 0.5
        agent['x'] = (agent['x'] + dx) % GRID_SIZE
        agent['y'] = (agent['y'] + dy) % GRID_SIZE

# Konwersja pozycji na float i aktualizacja rysunku
fig, ax = plt.subplots()
sc = ax.scatter([], [], c=[], cmap='hsv', s=20, vmin=0, vmax=2*np.pi)
ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)
ax.set_title("Emergencja i rezonans w układzie agentów")


def animate(frame):
    update_agents()
    x = [agent['x'] for agent in agents]
    y = [agent['y'] for agent in agents]
    phases = [agent['phase'] for agent in agents]
    sc.set_offsets(np.column_stack((x, y)))
    sc.set_array(np.array(phases))
    return sc,

ani = animation.FuncAnimation(fig, animate, frames=STEPS, interval=100, blit=True)
plt.show()
