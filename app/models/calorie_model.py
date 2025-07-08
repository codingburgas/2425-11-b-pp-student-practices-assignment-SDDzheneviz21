import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class CaloriePredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = ['Gender_encoded', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
        self.model_path = 'app/models/calorie_model.pkl'
        self.scaler_path = 'app/models/calorie_scaler.pkl'
        self.encoder_path = 'app/models/calorie_encoder.pkl'
        
    def train_and_save_model(self):
        """
        Train the model on the provided dataset and save it
        """
        print("Training calorie prediction model...")
        
        # Load datasets
        calories_df = pd.read_csv('calories.csv')
        exercise_df = pd.read_csv('exercise.csv')
        
        # Merge datasets
        merged_df = pd.merge(exercise_df, calories_df, on='User_ID', how='inner')
        
        # Preprocess data
        df_processed = merged_df.copy()
        
        # Encode gender
        self.label_encoder = LabelEncoder()
        df_processed['Gender_encoded'] = self.label_encoder.fit_transform(df_processed['Gender'])
        
        # Prepare features and target
        X = df_processed[self.feature_columns]
        y = df_processed['Calories']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Model trained successfully!")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.2f}")
        
        # Save model and preprocessors
        self.save_model()
        
        return r2, rmse
    
    def save_model(self):
        """
        Save the trained model and preprocessors
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save label encoder
        with open(self.encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print("Model saved successfully!")
    
    def load_model(self):
        """
        Load the trained model and preprocessors
        """
        try:
            # Load model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load label encoder
            with open(self.encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            print("Model loaded successfully!")
            return True
        except FileNotFoundError:
            print("Model files not found. Training new model...")
            return False
    
    def predict_calories(self, gender, age, height, weight, duration, heart_rate, body_temp):
        """
        Make calorie prediction using the trained model
        
        Args:
            gender (str): 'male' or 'female'
            age (int): age in years
            height (float): height in cm
            weight (float): weight in kg
            duration (float): exercise duration in minutes
            heart_rate (float): heart rate in bpm
            body_temp (float): body temperature in celsius
        
        Returns:
            float: predicted calories burned
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not available. Please train the model first.")
        
        # Prepare input data
        gender_encoded = self.label_encoder.transform([gender])[0]
        
        input_data = np.array([[
            gender_encoded,
            age,
            height,
            weight,
            duration,
            heart_rate,
            body_temp
        ]])
        
        # Scale input data
        input_scaled = self.scaler.transform(input_data)
        
        # Make prediction
        prediction = self.model.predict(input_scaled)[0]
        
        return max(prediction, 0)  # Ensure non-negative calories
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not available. Please train the model first.")
        
        importance = np.abs(self.model.coef_)
        feature_importance = dict(zip(self.feature_columns, importance))
        
        # Sort by importance
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_importance

# Global model instance
calorie_model = CaloriePredictionModel()

def initialize_calorie_model():
    """
    Initialize the calorie prediction model
    """
    # Try to load existing model, if not available, train new one
    if not calorie_model.load_model():
        calorie_model.train_and_save_model()

def predict_calories_burned(gender, age, height, weight, duration, heart_rate, body_temp):
    """
    Convenience function to predict calories burned
    """
    return calorie_model.predict_calories(gender, age, height, weight, duration, heart_rate, body_temp)

def get_model_feature_importance():
    """
    Convenience function to get feature importance
    """
    return calorie_model.get_feature_importance() 