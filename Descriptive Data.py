import pandas as pd
import numpy as np
import statistics as stats

#read datasets
df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")


print("Dataset 1 Variables")
for col in df1.columns:
#contain only numeric columns for mean, median, mode, std deviation
    if pd.api.types.is_numeric_dtype(df1[col]):
        data = df1[col].dropna()
        mean = stats.mean(data)
        median = stats.median(data)
        try:
            mode = stats.mode(data)
        except stats.StatisticsError:
            mode = "No unique mode"
        std_dev = np.std(data, ddof=1)
        print(f"\nVariable: {col}")
        print(f"  Mean: {mean:.2f}")
        print(f"  Median: {median:.2f}")
        print(f"  Mode: {mode}")
        print(f"  Standard Deviation: {std_dev:.2f}")

print("\n\nDataset 2 Variables")
for col in df2.columns:
#contain only numeric columns for mean, median, mode, std deviation (dataset 2)
    if pd.api.types.is_numeric_dtype(df2[col]):
        data = df2[col].dropna()
        mean = stats.mean(data)
        median = stats.median(data)
        try:
            mode = stats.mode(data)
        except stats.StatisticsError:
            mode = "No unique mode"
        std_dev = np.std(data, ddof=1)
        print(f"\nVariable: {col}")
        print(f"  Mean: {mean:.2f}")
        print(f"  Median: {median:.2f}")
        print(f"  Mode: {mode}")
        print(f"  Standard Deviation: {std_dev:.2f}")

