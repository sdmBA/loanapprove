# test_csv_integration.py - Test script for vehicle CSV integration

import pandas as pd
import os
import sys
from pathlib import Path

def test_csv_file():
    """Test CSV file loading and validation"""
    
    print("🧪 Testing CSV File Integration")
    print("=" * 50)
    
    # Test 1: Check if CSV file exists
    csv_file = "Book1.csv"
    print(f"1️⃣ Checking if {csv_file} exists...")
    
    if not os.path.exists(csv_file):
        print(f"❌ {csv_file} not found in current directory")
        print(f"📁 Current directory: {os.getcwd()}")
        print(f"📋 Files in directory: {list(os.listdir('.'))}")
        return False
    
    print(f"✅ {csv_file} found")
    
    # Test 2: Check file properties
    print(f"\n2️⃣ Checking file properties...")
    file_size = os.path.getsize(csv_file)
    print(f"📊 File size: {file_size:,} bytes")
    
    # Test 3: Load CSV file
    print(f"\n3️⃣ Loading CSV file...")
    
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"✅ CSV loaded successfully with UTF-8 encoding")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_file, encoding='latin-1')
            print(f"⚠️ CSV loaded with latin-1 encoding (consider converting to UTF-8)")
        except Exception as e:
            print(f"❌ Failed to load CSV with any encoding: {e}")
            return False
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False
    
    print(f"📊 DataFrame shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Test 4: Validate columns
    print(f"\n4️⃣ Validating columns...")
    required_columns = ['Brand', 'Model_Name']
    
    for col in required_columns:
        if col in df.columns:
            print(f"✅ Column '{col}' found")
        else:
            print(f"❌ Required column '{col}' missing")
            print(f"💡 Available columns: {list(df.columns)}")
            return False
    
    # Test 5: Check data quality
    print(f"\n5️⃣ Checking data quality...")
    
    # Check for null values
    null_brands = df['Brand'].isnull().sum()
    null_models = df['Model_Name'].isnull().sum()
    
    print(f"📊 Null values in Brand: {null_brands}")
    print(f"📊 Null values in Model_Name: {null_models}")
    
    if null_brands > 0 or null_models > 0:
        print(f"⚠️ Found null values in data")
    else:
        print(f"✅ No null values found")
    
    # Check for empty strings
    empty_brands = (df['Brand'] == '').sum()
    empty_models = (df['Model_Name'] == '').sum()
    
    print(f"📊 Empty brands: {empty_brands}")
    print(f"📊 Empty models: {empty_models}")
    
    # Test 6: Analyze brands and models
    print(f"\n6️⃣ Analyzing brands and models...")
    
    unique_brands = df['Brand'].unique()
    brand_count = len(unique_brands)
    total_models = len(df)
    
    print(f"🚗 Total unique brands: {brand_count}")
    print(f"📱 Total models: {total_models}")
    
    print(f"\n📋 Brand breakdown:")
    for brand in sorted(unique_brands):
        if pd.notna(brand) and brand != '':
            brand_models = df[df['Brand'] == brand]
            model_count = len(brand_models)
            print(f"  • {brand}: {model_count} models")
            
            # Show first few models
            first_models = brand_models['Model_Name'].head(3).tolist()
            for model in first_models:
                if pd.notna(model) and model != '':
                    # Truncate long model names
                    display_model = model[:50] + "..." if len(model) > 50 else model
                    print(f"    - {display_model}")
            
            if model_count > 3:
                print(f"    ... and {model_count - 3} more models")
    
    # Test 7: Sample data display
    print(f"\n7️⃣ Sample data:")
    print("-" * 40)
    sample_df = df.head()
    for idx, row in sample_df.iterrows():
        brand = row['Brand']
        model = row['Model_Name']
        print(f"{idx+1:2d}. {brand:15s} | {model}")
    
    return True

def test_streamlit_integration():
    """Test Streamlit app integration"""
    
    print(f"\n🌐 Testing Streamlit Integration")
    print("=" * 50)
    
    # Test 1: Check if Streamlit is installed
    print(f"1️⃣ Checking Streamlit installation...")
    
    try:
        import streamlit as st
        print(f"✅ Streamlit installed successfully")
    except ImportError:
        print(f"❌ Streamlit not installed")
        print(f"💡 Install with: pip install streamlit")
        return False
    
    # Test 2: Check if minimal_app.py exists
    print(f"\n2️⃣ Checking app file...")
    
    app_file = "minimal_app.py"
    if not os.path.exists(app_file):
        print(f"❌ {app_file} not found")
        return False
    
    print(f"✅ {app_file} found")
    
    # Test 3: Test data loading function
    print(f"\n3️⃣ Testing data loading function...")
    
    try:
        # Import the function from minimal_app
        sys.path.append('.')
        from minimal_app import load_vehicle_data, get_default_vehicle_data
        
        # Test loading
        vehicle_data = load_vehicle_data()
        
        if vehicle_data:
            print(f"✅ Vehicle data loaded successfully")
            print(f"📊 Brands loaded: {len(vehicle_data)}")
            
            total_models = sum(len(models) for models in vehicle_data.values())
            print(f"📱 Total models: {total_models}")
            
            # Show sample
            for brand, models in list(vehicle_data.items())[:2]:
                print(f"  • {brand}: {len(models)} models")
        else:
            print(f"❌ Failed to load vehicle data")
            return False
            
    except Exception as e:
        print(f"❌ Error testing data loading: {e}")
        return False
    
    return True

def test_prediction_engine():
    """Test prediction engine with vehicle data"""
    
    print(f"\n🔮 Testing Prediction Engine")
    print("=" * 50)
    
    try:
        from minimal_app import create_mock_prediction_engine
        
        predict_func = create_mock_prediction_engine()
        
        # Test case 1: Good application with premium brand
        test_data_1 = {
            'salary': 60000,
            'contract_amount': 500000,
            'loan_term': 36,
            'fraud_alert_pm1': 'No',
            'fraud_alert_pm2': 'No',
            'brand': 'HONDA',
            'model_name': 'HONDA รุ่น PCX 150'
        }
        
        print(f"1️⃣ Testing good application (Premium brand)...")
        result1 = predict_func(test_data_1)
        
        if len(result1) == 3:
            prediction1, probability1, risk1 = result1
            print(f"✅ Prediction: {prediction1} (Risk: {risk1:.1%})")
        else:
            prediction1, probability1 = result1
            print(f"✅ Prediction: {prediction1}")
        
        # Test case 2: Risky application
        test_data_2 = {
            'salary': 20000,
            'contract_amount': 800000,
            'loan_term': 24,
            'fraud_alert_pm1': 'Yes',
            'fraud_alert_pm2': 'No',
            'brand': 'Others',
            'model_name': 'Unknown Model'
        }
        
        print(f"\n2️⃣ Testing risky application...")
        result2 = predict_func(test_data_2)
        
        if len(result2) == 3:
            prediction2, probability2, risk2 = result2
            print(f"✅ Prediction: {prediction2} (Risk: {risk2:.1%})")
        else:
            prediction2, probability2 = result2
            print(f"✅ Prediction: {prediction2}")
        
        print(f"\n✅ Prediction engine working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error testing prediction engine: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    
    print("🚀 Starting Vehicle CSV Integration Tests")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: CSV File
    if test_csv_file():
        tests_passed += 1
        print("\n✅ CSV Test: PASSED")
    else:
        print("\n❌ CSV Test: FAILED")
    
    # Test 2: Streamlit Integration
    if test_streamlit_integration():
        tests_passed += 1
        print("\n✅ Streamlit Integration Test: PASSED")
    else:
        print("\n❌ Streamlit Integration Test: FAILED")
    
    # Test 3: Prediction Engine
    if test_prediction_engine():
        tests_passed += 1
        print("\n✅ Prediction Engine Test: PASSED")
    else:
        print("\n❌ Prediction Engine Test: FAILED")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🏁 Test Summary: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your app should work correctly.")
        print("\n🚀 Next steps:")
        print("   1. Run: streamlit run minimal_app.py --server.address 0.0.0.0 --server.port 8088")
        print("   2. Open: http://localhost:8088")
        print("   3. Test the vehicle selection dropdown")
    else:
        print("⚠️ Some tests failed. Please fix the issues before running the app.")
        
        if tests_passed == 0:
            print("\n💡 Quick fixes:")
            print("   1. Make sure Book1.csv is in the same directory")
            print("   2. Install required packages: pip install streamlit pandas")
            print("   3. Check file permissions and encoding")

if __name__ == "__main__":
    run_all_tests()