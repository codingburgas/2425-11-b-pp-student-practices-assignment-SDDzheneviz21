import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load and merge datasets
exercise_df = pd.read_csv("exercise.csv")
calories_df = pd.read_csv("calories.csv")
df = exercise_df.merge(calories_df, on='User_ID', how='inner')

# Features to analyze
features = ['Duration', 'Heart_Rate', 'Body_Temp', 'Age', 'Weight', 'Height']

print("=== LINEAR REGRESSION FEATURE ANALYSIS ===")
print(f"Dataset size: {len(df)} records")
print(f"Features analyzed: {features}")
print(f"Target: Calories burned\n")

# Create subplots for all features
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, feature in enumerate(features):
    x = df[feature].to_numpy().reshape(-1, 1)
    y = df['Calories'].to_numpy()
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    
    # Calculate R-squared
    r_squared = model.score(x, y)
    
    # Plot
    axes[i].scatter(x, y, alpha=0.5, color='blue', s=20)
    axes[i].plot(x, y_pred, color='red', linewidth=2, label=f'R² = {r_squared:.3f}')
    
    # Set axis limits based on data range
    x_min, x_max = x.min(), x.max()
    x_padding = (x_max - x_min) * 0.05
    axes[i].set_xlim(x_min - x_padding, x_max + x_padding)
    
    # Set y-axis to start at 0 or just below min actual value
    y_min, y_max = y.min(), y.max()
    y_padding = (y_max - y_min) * 0.05
    axes[i].set_ylim(max(0, y_min - y_padding), y_max + y_padding)
    
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Calories Burned')
    axes[i].set_title(f'{feature} vs Calories Burned')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print("LINEAR REGRESSION RESULTS:")
print("-" * 50)
for feature in features:
    x = df[feature].to_numpy().reshape(-1, 1)
    y = df['Calories'].to_numpy()
    
    model = LinearRegression()
    model.fit(x, y)
    r_squared = model.score(x, y)
    
    print(f"{feature:12} | R² = {r_squared:.4f} | Slope = {model.coef_[0]:.4f}")

print("\n=== ANALYSIS COMPLETE ===") 