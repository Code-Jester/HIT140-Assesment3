import math
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# 1) Load data

bat = pd.read_csv("dataset1.csv") 
rat = pd.read_csv("dataset2.csv")   

def month_to_season(m):
    """Australian seasons for this project: Winter=Jun–Aug, Spring=Sep–Nov."""
    try:
        m = int(m)
    except Exception:
        return "Other"
    if m in [6, 7, 8]:
        return "Winter"
    elif m in [9, 10, 11]:
        return "Spring"
    return "Other"

def normalise_season(df, season_col="season", month_col="month"):
    if season_col in df.columns:
        s = df[season_col]
        if pd.api.types.is_numeric_dtype(s):
            # Common coding in these datasets: 0=winter, 1=spring
            df["season_norm"] = s.map({0: "Winter", 1: "Spring"}).fillna("Other")
        else:
            df["season_norm"] = s.astype(str).str.strip().str.lower().map(
                {"winter": "Winter", "spring": "Spring"}
            ).fillna("Other")
    elif month_col in df.columns:
        df["season_norm"] = df[month_col].apply(month_to_season)
    else:
        df["season_norm"] = "Other"
    return df

bat = normalise_season(bat)
rat = normalise_season(rat)

# Keep only Winter & Spring (per Investigation B)
bat = bat[bat["season_norm"].isin(["Winter", "Spring"])].copy()
rat = rat[rat["season_norm"].isin(["Winter", "Spring"])].copy()

print("=== Season counts ===")
print("BAT:", bat["season_norm"].value_counts(dropna=False).to_dict())
print("RAT:", rat["season_norm"].value_counts(dropna=False).to_dict(), "\n")

for col in ["reward", "risk", "seconds_after_rat_arrival", "hours_after_sunset"]:
    if col not in bat.columns:
        # Create harmless defaults if missing (keeps script runnable)
        if col in ["reward", "risk"]:
            bat[col] = 0
        else:
            bat[col] = np.nan

PRED_COLS = ["risk", "seconds_after_rat_arrival", "hours_after_sunset"]

def run_lr_for_season(df_season, season_name):
    df = df_season.dropna(subset=["reward"] + PRED_COLS).copy()
    if len(df) < 6:
        print(f"{season_name}: Not enough rows for a stable train/test split (need ≥6).")
        return

    X = df[PRED_COLS].values
    y = df["reward"].values

    # Split (60% train, 40% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0
    )

    # Build and fit LR model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Intercept & Coefficients
    print(f"\n===== {season_name} – Linear Regression =====")
    print("Intercept:", model.intercept_)
    print("Coefficients (in order of predictors):")
    for name, coef in zip(PRED_COLS, model.coef_):
        print(f"  {name}: {coef}")

    # Prediction on test
    y_pred = model.predict(X_test)

    # Show Actual vs Predicted (small peek)
    df_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    print("\nSample predictions (head):")
    print(df_pred.head())

    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    y_max, y_min = y_test.max(), y_test.min()
    rmse_norm = rmse / (y_max - y_min) if (y_max - y_min) != 0 else float("nan")
    r2 = metrics.r2_score(y_test, y_pred)

    print("\nPerformance (LR):")
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("RMSE (Normalised):", rmse_norm)
    print("R^2:", r2)

    print("\n##### BASELINE MODEL (mean) #####")
    y_base = np.mean(y_train)
    y_pred_base = np.repeat(y_base, len(y_test))

    df_base_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_base})
    print("Sample baseline predictions (head):")
    print(df_base_pred.head())

    mae_b = metrics.mean_absolute_error(y_test, y_pred_base)
    mse_b = metrics.mean_squared_error(y_test, y_pred_base)
    rmse_b = math.sqrt(mse_b)
    rmse_norm_b = rmse_b / (y_max - y_min) if (y_max - y_min) != 0 else float("nan")
    r2_b = metrics.r2_score(y_test, y_pred_base)

    print("\nPerformance (Baseline):")
    print("MAE:", mae_b)
    print("MSE:", mse_b)
    print("RMSE:", rmse_b)
    print("RMSE (Normalised):", rmse_norm_b)
    print("R^2:", r2_b)

bat_winter = bat[bat["season_norm"] == "Winter"].copy()
bat_spring = bat[bat["season_norm"] == "Spring"].copy()

print("\n=== Running per-season Linear Regression (reward ~ risk + seconds_after_rat_arrival + hours_after_sunset) ===")
run_lr_for_season(bat_winter, "WINTER")
run_lr_for_season(bat_spring, "SPRING")
