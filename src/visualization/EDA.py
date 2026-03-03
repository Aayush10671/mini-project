import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("./reports/figures", exist_ok=True)


df = pd.read_csv("./data/processed/train_engineered.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())



plt.figure(figsize=(12, 8))
correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=False,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("./reports/figures/correlation_heatmap.png")
plt.close()



plt.figure(figsize=(8, 5))
sns.histplot(df["Final_Score"], bins=20, kde=True)

plt.title("Distribution of Final Score")
plt.xlabel("Final Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("./reports/figures/final_score_distribution.png")
plt.close()



plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=df["Study_Hours_per_Week"],
    y=df["Final_Score"]
)

plt.title("Study Hours vs Final Score")
plt.tight_layout()
plt.savefig("./reports/figures/study_hours_vs_final_score.png")
plt.close()



plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=df["Attendance (%)"],
    y=df["Final_Score"]
)

plt.title("Attendance vs Final Score")
plt.tight_layout()
plt.savefig("./reports/figures/attendance_vs_final_score.png")
plt.close()


plt.figure(figsize=(8, 5))
sns.boxplot(
    x=df["Stress_Level"],
    y=df["Final_Score"]
)

plt.title("Stress Level vs Final Score")
plt.tight_layout()
plt.savefig("./reports/figures/stress_vs_final_score.png")
plt.close()

print("\nEDA Completed. Plots saved in ./reports/figures/")