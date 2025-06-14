import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Ситуация 1: Избор на работа
strategies = ['S1', 'S2', 'S3', 'S4', 'S5'] 
A = np.array([0.85, 0.43, 0.67, 0.85, 0.67])  # Ентропия на Шанън за стратегиите
x = A / A.sum()

a_job = np.array([
    [0.9, 0.1, 0.4, 0.6, 0.2],  # S1
    [0.1, 0.9, 0.1, 0.2, 0.4],  # S2
    [0.4, 0.1, 0.9, 0.5, 0.2],  # S3
    [0.6, 0.2, 0.5, 0.9, 0.3],  # S4
    [0.2, 0.4, 0.2, 0.3, 0.9],  # S5
])

def calculate_fitness(a_matrix, x):
    return a_matrix @ x

def calculate_average_fitness(fitness, x):
    return np.dot(x, fitness)

def replicator_dynamics(fitness, average_fitness, x):
    return x * (fitness - average_fitness)

iterations = 20
history = [x.copy()]
for _ in range(iterations):
    fitness = calculate_fitness(a_job, x)
    avg_fitness = calculate_average_fitness(fitness, x)
    dx = replicator_dynamics(fitness, avg_fitness, x)
    x = x + dx
    x = x / x.sum()
    history.append(x.copy())

df = pd.DataFrame(history, columns=strategies)
df['iteration'] = range(len(history))

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
df.to_csv(output_dir / "job_results.csv", index=False, encoding='utf-8-sig')

plt.figure(figsize=(10, 6))
for strategy in strategies:
    plt.plot(df['iteration'], df[strategy], label=strategy)

plt.xlabel("Итерация")
plt.ylabel("Дял от популацията")
plt.title("Ситуация 1: Развитие на стратегии при избор на работа")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / "job_situation_graph.png")
plt.show()
