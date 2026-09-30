import os
import pandas as pd
import matplotlib.pyplot as plt

data_path = "palm-sEMG-doorknob-filtered-main/data"

subjects = ["A", "B", "C"]

fig, axes = plt.subplots(3, 1, figsize=(12, 9))

for i, subject in enumerate(subjects):
    file_path = os.path.join(data_path, subject, f"{subject.lower()} (1).csv")
    df = pd.read_csv(file_path)

    time = [j / 1000 for j in range(len(df))]

    axes[i].plot(time, df["Comp Ch 3"])
    axes[i].set_title(f"Subject {subject}")
    axes[i].set_ylabel("sEMG")
    axes[i].axvline(1, linestyle="--")
    axes[i].axvline(2, linestyle="--")
    axes[i].grid()

    axes[i].text(0.5, 0.95, "파지", transform=axes[i].transAxes, ha="center")
    axes[i].text(1.5, 0.95, "회전", transform=axes[i].transAxes, ha="center")
    axes[i].text(2.5, 0.95, "정지", transform=axes[i].transAxes, ha="center")

axes[-1].set_xlabel("시간 (초)")

plt.tight_layout()
plt.show()