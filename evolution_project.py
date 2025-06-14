# Поправям синтактичната грешка в реда за запис на CSV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Общи параметри
strategies = ['S1', 'S2', 'S3', 'S4', 'S5']
A = np.array([0.85, 0.43, 0.67, 0.85, 0.67])  # Шанън ентропия за стратегиите
x_init = A / A.sum()  # Начално разпределение

# 🔧 Универсална функция за симулация на сценарий
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
    plt.title(f"Ситуация 2: Групов проект – Развитие на стратегии")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_graph.png")
    plt.close()

# ▶️ Симулация за Ситуация 2: Краткосрочен екипен проект
a_project = np.array([
    [0.7, 0.4, 0.3, 0.5, 0.6],  # S1 – Адаптивност
    [0.4, 0.8, 0.2, 0.3, 0.6],  # S2 – Агресивност
    [0.3, 0.2, 0.7, 0.4, 0.3],  # S3 – Консервативност
    [0.5, 0.3, 0.4, 0.7, 0.5],  # S4 – Аналитичност
    [0.6, 0.6, 0.3, 0.5, 0.8],  # S5 – Интуитивност
])

# Стартираме симулацията
simulate_scenario(a_matrix=a_project, name="project", x0=x_init)
