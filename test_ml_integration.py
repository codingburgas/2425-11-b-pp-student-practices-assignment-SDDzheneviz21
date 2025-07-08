#!/usr/bin/env python3
"""
Test script to verify ML model integration
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_loading():
    """Test if the model can be loaded and used"""
    try:
        from app.models.calorie_model import predict_calories_burned, get_model_feature_importance
        
        print("Testing ML model integration...")
        
        # Test prediction
        prediction = predict_calories_burned(
            gender='male',
            age=30,
            height=175.0,
            weight=70.0,
            duration=25.0,
            heart_rate=100.0,
            body_temp=40.5
        )
        
        print(f"✓ Prediction successful: {prediction:.2f} calories")
        
        # Test feature importance
        importance = get_model_feature_importance()
        print("✓ Feature importance retrieved:")
        for feature, imp in importance:
            print(f"  - {feature}: {imp:.3f}")
        
        print("\n🎉 ML model integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ ML model integration test failed: {e}")
        return False

def test_flask_integration():
    """Test if Flask app can start with ML model"""
    try:
        from app import create_app
        
        print("\nTesting Flask app with ML model...")
        app = create_app()
        
        with app.app_context():
            from app.models.calorie_model import predict_calories_burned
            prediction = predict_calories_burned(
                gender='female',
                age=25,
                height=165.0,
                weight=60.0,
                duration=30.0,
                heart_rate=110.0,
                body_temp=40.0
            )
            print(f"✓ Flask integration successful: {prediction:.2f} calories")
        
        print("🎉 Flask integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Flask integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ML MODEL INTEGRATION TEST")
    print("=" * 60)
    
    # Test 1: Model loading
    model_ok = test_model_loading()
    
    # Test 2: Flask integration
    flask_ok = test_flask_integration()
    
    print("\n" + "=" * 60)
    if model_ok and flask_ok:
        print("✅ ALL TESTS PASSED!")
        print("\nThe ML model is ready to use in your Flask application.")
        print("Users can now make predictions using the pre-trained model.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above.")
    print("=" * 60) 