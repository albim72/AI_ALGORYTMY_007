
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parametry systemu
GRID_SIZE = 50
NUM_AGENTS = 200
STEPS = 100

# Inicjalizacja agentów (każdy agent ma pozycję i stan wewnętrzny)
agents = []
for _ in range(NUM_AGENTS):
    x, y = np.random.randint(0, GRID_SIZE, 2)
    phase = np.random.rand() * 2 * np.pi  # faza (rezonans)
    agents.append({'x': x, 'y': y, 'phase': phase})

# Funkcja aktualizacji stanu agentów
def update_agents():
    for agent in agents:
        # Rezonans: dostrajanie fazy do sąsiadów (w zasięgu 3)
        neighbors = [a for a in agents if abs(a['x'] - agent['x']) <= 3 and abs(a['y'] - agent['y']) <= 3 and a != agent]
        if neighbors:
            avg_phase = np.mean([n['phase'] for n in neighbors])
            agent['phase'] += 0.1 * (avg_phase - agent['phase'])  # powolne dostrajanie

        # Emergentne zachowanie: przesuwanie się w kierunku lokalnej synchronizacji
        dx = np.cos(agent['phase']) * 0.5
        dy = np.sin(agent['phase']) * 0.5
        agent['x'] = int((agent['x'] + dx) % GRID_SIZE)
        agent['y'] = int((agent['y'] + dy) % GRID_SIZE)

# Wizualizacja
fig, ax = plt.subplots()
sc = ax.scatter([], [], c=[], cmap='hsv', s=10, vmin=0, vmax=2*np.pi)
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
