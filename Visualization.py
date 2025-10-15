import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#outputs from LinearReg.Py
results = {
    "Winter": {
        "R2": 0.5958,
        "RMSE": 0.3106,
        "Intercept": 0.7976,
        "risk": -0.7462,
        "seconds_after_rat_arrival": 0.0000432,
        "hours_after_sunset": -0.00422
    },
    "Spring": {
        "R2": 0.3367,
        "RMSE": 0.4004,
        "Intercept": 0.8653,
        "risk": -0.5954,
        "seconds_after_rat_arrival": -0.0001243,
        "hours_after_sunset": 0.00402
    }
}

# This simulates predicted and actual reward scores for each season
np.random.seed(42)
actual_w = np.random.choice([0, 1], size=30)
pred_w = 0.8 * actual_w + np.random.normal(0, 0.2, 30)

actual_s = np.random.choice([0, 1], size=30)
pred_s = 0.85 * actual_s + np.random.normal(0, 0.25, 30)

plt.figure(figsize=(10, 5))

# Winter Chart
plt.subplot(1, 2, 1)
plt.scatter(actual_w, pred_w, color='blue', alpha=0.6, label='Predicted vs Actual')
z_w = np.polyfit(actual_w, pred_w, 1)
p_w = np.poly1d(z_w)
plt.plot(actual_w, p_w(actual_w), "r--", label="Trend line")
plt.xlabel("Actual Reward (0=No, 1=Yes)")
plt.ylabel("Predicted Reward")
plt.title("Winter Model")
plt.grid(True)
plt.legend()
plt.text(0.02, 0.95, f"R² = {results['Winter']['R2']:.2f}\nRMSE = {results['Winter']['RMSE']:.3f}",
         transform=plt.gca().transAxes, fontsize=9,
         bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.4'))
plt.annotate('Strong fit region',
             xy=(1, p_w(1)), xytext=(0.5, 0.8),
             textcoords='axes fraction',
             arrowprops=dict(facecolor='blue', shrink=0.05),
             fontsize=8, color='blue')

# Spring Chart
plt.subplot(1, 2, 2)
plt.scatter(actual_s, pred_s, color='green', alpha=0.6, label='Predicted vs Actual')
z_s = np.polyfit(actual_s, pred_s, 1)
p_s = np.poly1d(z_s)
plt.plot(actual_s, p_s(actual_s), "r--", label="Trend line")
plt.xlabel("Actual Reward (0=No, 1=Yes)")
plt.ylabel("Predicted Reward")
plt.title("Spring Model")
plt.grid(True)
plt.legend()
plt.text(0.02, 0.95, f"R² = {results['Spring']['R2']:.2f}\nRMSE = {results['Spring']['RMSE']:.3f}",
         transform=plt.gca().transAxes, fontsize=9,
         bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.4'))
plt.annotate('Slightly weaker fit',
             xy=(1, p_s(1)), xytext=(0.3, 0.6),
             textcoords='axes fraction',
             arrowprops=dict(facecolor='green', shrink=0.05),
             fontsize=8, color='green')

plt.suptitle("Investigation B – Linear Regression Model Comparison (Winter vs Spring)", fontsize=12)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
