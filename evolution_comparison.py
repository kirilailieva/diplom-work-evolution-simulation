import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

scenarios = {
    "job": "Избор на работа",
    "project": "Групов проект",
    "finance": "Финансова криза",
    "path": "Кариерен избор"
}

output_dir = Path("output")
dataframes = {}

# Зареждаме всяка таблица по име
for key, label in scenarios.items():
    filename = f"{key}_results.csv"
    df = pd.read_csv(output_dir / filename)
    dataframes[key] = df

# Визуализираме всички стратегии в 4 ситуации
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 18), sharex=True)

for i, (key, label) in enumerate(scenarios.items()):
    df = dataframes[key]
    for strategy in df.columns[:-1]:  # без 'iteration'
        axs[i].plot(df['iteration'], df[strategy], label=strategy)
    axs[i].set_title(f"Ситуация: {label}")
    axs[i].set_ylabel("Дял")
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel("Итерация")
plt.tight_layout()
plt.savefig(output_dir / "combined_scenarios_graph.png")
plt.show()
