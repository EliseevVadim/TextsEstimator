import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def plot_pie_chart(data: pd.DataFrame, column: str, title: str) -> None:
    value_counts = data[column].value_counts()
    total = sum(value_counts)
    plt.figure(figsize=(10, 10))
    wedges, texts, auto_texts = plt.pie(
        value_counts,
        labels=value_counts.index,
        autopct=lambda p: f'{p:.1f}%\n({int(p * sum(value_counts) / 100)})' if p > 2 else '',
        startangle=90,
        pctdistance=0.85,
        labeldistance=1.1,
        textprops={'fontsize': 11},
        shadow=True
    )
    for autotext in auto_texts:
        autotext.set_fontsize(12)
        autotext.set_color('white')
    legend_labels = [
        f'{label} ({value_counts[label]}, {value_counts[label] / total * 100:.1f}%)'
        for label in value_counts.index
    ]
    plt.legend(wedges, legend_labels, title=column, loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
    plt.title(title)
    plt.show()


def plot_evaluations_histograms(data: pd.DataFrame, models: list[str], integers_only=False) -> None:
    bins = None
    if integers_only:
        unique_counts = [data[model].nunique() for model in models]
        bins_number = max(unique_counts) + 1
        bins = np.linspace(1, bins_number, bins_number)
        labels = np.int16(np.linspace(1, bins_number - 1, bins_number - 1))
        ticks = labels + 0.5
    for model in models:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[model], bins=bins, color="blue", edgecolor="black", alpha=0.7)
        plt.title(f"Оценки модели: {model}", fontsize=16, fontweight='bold')
        plt.xlabel("Оценка", fontsize=14)
        plt.ylabel("Частота", fontsize=14)
        if integers_only:
            plt.xticks(ticks, labels)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
