import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Стратегии и начални ентропийни стойности
strategies = ['S1', 'S2', 'S3', 'S4', 'S5']
A = np.array([0.85, 0.43, 0.67, 0.85, 0.67])  # Ентропийни стойности
x = A / A.sum()  # Начални дялове, нормализирани

# Примерна платежна матрица (a_ij) — можеш да я замениш с твоя
a_matrix = np.array([
    [1.0, 0.3, 0.4, 0.6, 0.5],
    [0.3, 1.0, 0.2, 0.4, 0.6],
    [0.4, 0.2, 1.0, 0.5, 0.5],
    [0.6, 0.4, 0.5, 1.0, 0.7],
    [0.5, 0.6, 0.5, 0.7, 1.0]
])

# Функция за пресмятане на фитнеси
def calculate_fitness(a_matrix, x):
    return a_matrix @ x

# Среден фитнес на популацията
def calculate_average_fitness(fitness, x):
    return np.dot(x, fitness)

# Репликаторна динамика: изчисляване на dx_i/dt
def replicator_dynamics(fitness, average_fitness, x):
    return x * (fitness - average_fitness)

# Симулация през определен брой стъпки
iterations = 20
history = [x.copy()]

for _ in range(iterations):
    fitness = calculate_fitness(a_matrix, x)
    avg_fitness = calculate_average_fitness(fitness, x)
    dx = replicator_dynamics(fitness, avg_fitness, x)
    x = x + dx
    x = x / x.sum()  # нормализиране, за да е сума = 1
    history.append(x.copy())

# Показване на резултатите
df = pd.DataFrame(history, columns=strategies)
df['iteration'] = range(len(history))

# Визуализация
plt.figure(figsize=(10, 6))
for strategy in strategies:
    plt.plot(df['iteration'], df[strategy], label=strategy)

plt.xlabel('Итерация')
plt.ylabel('Дял от популацията')
plt.title('Развитие на стратегиите във времето')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#print(df)
df.to_csv("evolution_history.csv", index=False, encoding='utf-8-sig')
