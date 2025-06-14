import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Дефиниране на стратегии и начално разпределение (на база ентропия)
strategies = ['S1', 'S2', 'S3', 'S4', 'S5']
A = np.array([0.85, 0.43, 0.67, 0.85, 0.67])
x_init = A / A.sum()

# Универсална функция за симулация
def simulate_scenario(a_matrix, name, x0, iterations=20):
    x = x0.copy()
    history = [x.copy()]
    for _ in range(iterations):
        fitness = a_matrix @ x
        avg_fitness = np.dot(x, fitness)
        dx = x * (fitness - avg_fitness)
        x = x + dx
        x = x / x.sum()
        history.append(x.copy())

    df = pd.DataFrame(history, columns=strategies)
    df['iteration'] = range(len(history))

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / f"{name}_results.csv", index=False, encoding='utf-8-sig')

    plt.figure(figsize=(10, 6))
    for strategy in strategies:
        plt.plot(df['iteration'], df[strategy], label=strategy)

    plt.xlabel("Итерация")
    plt.ylabel("Дял от популацията")
    plt.title(f"Ситуация 4: Кариерен избор → Развитие на стратегии")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_graph.png")
    plt.close()

# Платежна матрица за Ситуация 4
a_path = np.array([
    [0.7, 0.3, 0.5, 0.6, 0.5],  # S1 – Адаптивност
    [0.3, 0.6, 0.2, 0.3, 0.4],  # S2 – Агресивност
    [0.5, 0.2, 0.7, 0.5, 0.3],  # S3 – Консервативност
    [0.6, 0.3, 0.5, 0.8, 0.6],  # S4 – Аналитичност
    [0.5, 0.4, 0.3, 0.6, 0.9],  # S5 – Интуитивност
])

# Стартиране на симулацията
simulate_scenario(a_matrix=a_path, name="path", x0=x_init)
