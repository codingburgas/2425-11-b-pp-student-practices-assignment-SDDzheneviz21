import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

def load_and_merge_datasets():
    """
    Load and merge the calories and exercise datasets
    """
    print("Loading datasets...")
    
    # Load datasets
    calories_df = pd.read_csv('calories.csv')
    exercise_df = pd.read_csv('exercise.csv')
    
    print(f"Calories dataset shape: {calories_df.shape}")
    print(f"Exercise dataset shape: {exercise_df.shape}")
    
    # Merge datasets on User_ID
    merged_df = pd.merge(exercise_df, calories_df, on='User_ID', how='inner')
    
    print(f"Merged dataset shape: {merged_df.shape}")
    print(f"Number of unique users: {merged_df['User_ID'].nunique()}")
    
    return merged_df

def preprocess_data(df):
    """
    Preprocess the data for machine learning
    """
    print("\nPreprocessing data...")
    
    # Create a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    df_processed['Gender_encoded'] = le.fit_transform(df_processed['Gender'])
    
    # Select features for modeling
    feature_columns = ['Gender_encoded', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    target_column = 'Calories'
    
    X = df_processed[feature_columns]
    y = df_processed[target_column]
    
    print(f"Feature columns: {feature_columns}")
    print(f"Target column: {target_column}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns

def train_linear_regression(X_train, X_test, y_train, y_test, feature_columns):
    """
    Train linear regression model and evaluate performance
    """
    print("\nTraining Linear Regression model...")
    
    # Train the model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = lr_model.predict(X_train)
    y_pred_test = lr_model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print("\nLinear Regression Performance:")
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    
    return lr_model, y_pred_test

def train_random_forest(X_train, X_test, y_train, y_test, feature_columns):
    """
    Train Random Forest model for feature importance comparison
    """
    print("\nTraining Random Forest model for feature importance...")
    
    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_rf = rf_model.predict(X_test)
    
    # Calculate metrics
    rf_r2 = r2_score(y_test, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    
    print(f"Random Forest R² Score: {rf_r2:.4f}")
    print(f"Random Forest RMSE: {rf_rmse:.2f}")
    
    return rf_model

def analyze_feature_importance(lr_model, rf_model, feature_columns, scaler):
    """
    Analyze feature importance from both models
    """
    print("\nAnalyzing feature importance...")
    
    # Linear Regression coefficients (absolute values for importance)
    lr_importance = np.abs(lr_model.coef_)
    
    # Random Forest feature importance
    rf_importance = rf_model.feature_importances_
    
    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Linear_Regression_Importance': lr_importance,
        'Random_Forest_Importance': rf_importance
    })
    
    # Sort by Random Forest importance
    importance_df = importance_df.sort_values('Random_Forest_Importance', ascending=False)
    
    print("\nFeature Importance Rankings:")
    print(importance_df.to_string(index=False))
    
    return importance_df

def make_predictions(model, scaler, feature_columns):
    """
    Make predictions on new data
    """
    print("\nMaking predictions on sample data...")
    
    # Sample data for prediction (you can modify these values)
    sample_data = {
        'Gender_encoded': [1],  # 0 for female, 1 for male
        'Age': [30],
        'Height': [175.0],  # cm
        'Weight': [70.0],   # kg
        'Duration': [25.0], # minutes
        'Heart_Rate': [100.0], # bpm
        'Body_Temp': [40.5]  # celsius
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_scaled = scaler.transform(sample_df)
    
    prediction = model.predict(sample_scaled)[0]
    
    print(f"Sample prediction for:")
    print(f"  Gender: {'Male' if sample_data['Gender_encoded'][0] == 1 else 'Female'}")
    print(f"  Age: {sample_data['Age'][0]} years")
    print(f"  Height: {sample_data['Height'][0]} cm")
    print(f"  Weight: {sample_data['Weight'][0]} kg")
    print(f"  Duration: {sample_data['Duration'][0]} minutes")
    print(f"  Heart Rate: {sample_data['Heart_Rate'][0]} bpm")
    print(f"  Body Temperature: {sample_data['Body_Temp'][0]}°C")
    print(f"  Predicted Calories Burned: {prediction:.2f} calories")
    
    return prediction

def plot_results(y_test, y_pred, importance_df):
    """
    Create visualization plots
    """
    print("\nCreating visualization plots...")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Calories')
    axes[0, 0].set_ylabel('Predicted Calories')
    axes[0, 0].set_title('Actual vs Predicted Calories')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals plot
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Calories')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Linear Regression Feature Importance
    axes[1, 0].barh(importance_df['Feature'], importance_df['Linear_Regression_Importance'])
    axes[1, 0].set_xlabel('Importance (Absolute Coefficient)')
    axes[1, 0].set_title('Linear Regression Feature Importance')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Random Forest Feature Importance
    axes[1, 1].barh(importance_df['Feature'], importance_df['Random_Forest_Importance'])
    axes[1, 1].set_xlabel('Importance')
    axes[1, 1].set_title('Random Forest Feature Importance')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('calorie_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Plots saved as 'calorie_prediction_results.png'")

def main():
    """
    Main function to run the complete analysis
    """
    print("=" * 60)
    print("CALORIE BURN PREDICTION MODEL ANALYSIS")
    print("=" * 60)
    
    # 1. Load and merge datasets
    merged_df = load_and_merge_datasets()
    
    # 2. Preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_columns = preprocess_data(merged_df)
    
    # 3. Train Linear Regression model
    lr_model, y_pred = train_linear_regression(X_train, X_test, y_train, y_test, feature_columns)
    
    # 4. Train Random Forest for feature importance comparison
    rf_model = train_random_forest(X_train, X_test, y_train, y_test, feature_columns)
    
    # 5. Analyze feature importance
    importance_df = analyze_feature_importance(lr_model, rf_model, feature_columns, scaler)
    
    # 6. Make predictions on sample data
    sample_prediction = make_predictions(lr_model, scaler, feature_columns)
    
    # 7. Create visualizations
    plot_results(y_test, y_pred, importance_df)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    
    # Summary
    print("\nSUMMARY:")
    print("- Linear Regression is the appropriate model for calorie prediction (continuous target)")
    print("- The model shows good performance with reasonable R² score")
    print("- Feature importance analysis reveals which factors most influence calorie burn")
    print("- The model can be used to predict calories burned for new users")
    
    return lr_model, scaler, feature_columns, importance_df

if __name__ == "__main__":
    main() 