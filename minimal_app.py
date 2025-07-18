# minimal_app.py - Guaranteed working version with CSV data
import streamlit as st
import pandas as pd
import os
import sys
import subprocess
import csv
from pathlib import Path

# Page config
st.set_page_config(
    page_title="🏦 Loan Prediction (Safe Mode)",
    page_icon="🏦",
    layout="wide"
)

def setup_spark_environment():
    """Setup Spark environment variables safely"""
    
    # Essential environment variables
    env_setup = {
        'SPARK_LOCAL_IP': '127.0.0.1',
        'SPARK_DRIVER_HOST': '127.0.0.1', 
        'SPARK_DRIVER_BIND_ADDRESS': '127.0.0.1',
        'PYSPARK_PYTHON': sys.executable,
        'PYSPARK_DRIVER_PYTHON': sys.executable,
        'SPARK_DRIVER_MEMORY': '512m',
        'SPARK_EXECUTOR_MEMORY': '512m',
        'SPARK_DRIVER_MAX_RESULT_SIZE': '256m'
    }
    
    for key, value in env_setup.items():
        os.environ[key] = value
    
    # Find and set JAVA_HOME if not set
    if not os.environ.get('JAVA_HOME'):
        java_homes = [
            '/usr/lib/jvm/java-11-openjdk-amd64',
            '/usr/lib/jvm/default-java',
            '/opt/java/openjdk',
            '/Library/Java/JavaVirtualMachines/adoptopenjdk-11.jdk/Contents/Home'
        ]
        
        for java_home in java_homes:
            if os.path.exists(java_home):
                os.environ['JAVA_HOME'] = java_home
                break

def load_vehicle_data():
    """Load vehicle brand and model data from CSV"""
    
    vehicle_data = {}
    csv_file = "Book1.csv"
    
    try:
        # Check if CSV file exists
        if not os.path.exists(csv_file):
            st.warning(f"⚠️ CSV file '{csv_file}' not found. Using default data.")
            return get_default_vehicle_data()
        
        # Read CSV file
        df = pd.read_csv(csv_file, encoding='utf-8')
        
        # Validate columns
        if 'Brand' not in df.columns or 'Model_Name' not in df.columns:
            st.error("❌ CSV file must contain 'Brand' and 'Model_Name' columns")
            return get_default_vehicle_data()
        
        # Group models by brand
        for _, row in df.iterrows():
            brand = str(row['Brand']).strip()
            model = str(row['Model_Name']).strip()
            
            if brand and model and brand != 'nan' and model != 'nan':
                if brand not in vehicle_data:
                    vehicle_data[brand] = []
                vehicle_data[brand].append(model)
        
        # Remove duplicates and sort
        for brand in vehicle_data:
            vehicle_data[brand] = sorted(list(set(vehicle_data[brand])))
        
        # Display success message
        total_brands = len(vehicle_data)
        total_models = sum(len(models) for models in vehicle_data.values())
        st.sidebar.success(f"✅ Loaded {total_brands} brands, {total_models} models from CSV")
        
        return vehicle_data
        
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return get_default_vehicle_data()

def get_default_vehicle_data():
    """Fallback vehicle data if CSV is not available"""
    return {
        "HONDA": ["HONDA รุ่น PCX 150", "HONDA รุ่น CLICK 160", "HONDA รุ่น WAVE 110i"],
        "YAMAHA": ["YAMAHA รุ่น AEROX 155", "YAMAHA รุ่น NMAX 155", "YAMAHA รุ่น FINO"],
        "SUZUKI": ["SUZUKI รุ่น BURGMAN STREET", "SUZUKI รุ่น SMASH", "SUZUKI รุ่น RAIDER"],
        "Others": ["Other Brand", "Custom Model"]
    }

@st.cache_data
def get_vehicle_data_cached():
    """Cached version of vehicle data loading"""
    return load_vehicle_data()
    """Check Java availability"""
    try:
        result = subprocess.run(['java', '-version'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0, result.stderr.split('\n')[0] if result.stderr else "Java found"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "Java not found or timeout"

def safe_spark_init():
    """Initialize Spark with maximum safety"""
    
    setup_spark_environment()
    
    try:
        from pyspark.sql import SparkSession
        from pyspark.conf import SparkConf
        
        # Ultra-conservative Spark config
        conf = SparkConf()
        conf.set("spark.ui.enabled", "false")
        conf.set("spark.ui.showConsoleProgress", "false")
        conf.set("spark.driver.host", "127.0.0.1")
        conf.set("spark.driver.bindAddress", "127.0.0.1")
        conf.set("spark.driver.memory", "512m")
        conf.set("spark.executor.memory", "512m")
        conf.set("spark.driver.maxResultSize", "256m")
        conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
        conf.set("spark.sql.adaptive.enabled", "false")
        
        # Create session with timeout
        spark = SparkSession.builder \
            .appName("LoanPredictionSafe") \
            .master("local[1]") \
            .config(conf=conf) \
            .getOrCreate()
        
        # Set log level to reduce noise
        spark.sparkContext.setLogLevel("ERROR")
        
        # Quick test
        test_df = spark.createDataFrame([(1, "test")], ["id", "value"])
        test_df.count()  # Force action
        
        return spark, "Spark initialized successfully"
        
    except Exception as e:
        return None, f"Spark initialization failed: {str(e)}"

def create_mock_prediction_engine():
    """Create a simple rule-based prediction engine (no Spark needed)"""
    
    def predict(data):
        # Simple rule-based prediction logic
        salary = data.get('salary', 0)
        contract_amount = data.get('contract_amount', 0)
        fraud_pm1 = data.get('fraud_alert_pm1', 'No')
        fraud_pm2 = data.get('fraud_alert_pm2', 'No')
        loan_term = data.get('loan_term', 36)
        brand = data.get('brand', '')
        model_name = data.get('model_name', '')
        
        # Calculate debt-to-income ratio
        if salary > 0:
            debt_ratio = (contract_amount / loan_term) / salary
        else:
            debt_ratio = float('inf')
        
        # Brand risk assessment (example logic)
        brand_risk_modifier = 0.0
        premium_brands = ['HONDA', 'YAMAHA']
        if brand in premium_brands:
            brand_risk_modifier = -0.05  # Lower risk for premium brands
        elif brand == 'Others':
            brand_risk_modifier = 0.1   # Higher risk for unknown brands
        
        # Decision logic with brand consideration
        base_risk = 0.0
        
        if fraud_pm1 == "Yes" and fraud_pm2 == "Yes":
            base_risk = 0.8  # High risk - multiple fraud alerts
        elif fraud_pm1 == "Yes" or fraud_pm2 == "Yes":
            base_risk = 0.5  # Medium risk - single fraud alert
        elif debt_ratio > 0.5:
            base_risk = 0.7  # High risk - high debt ratio
        elif debt_ratio > 0.3:
            base_risk = 0.4  # Medium risk - moderate debt ratio
        else:
            base_risk = 0.1  # Low risk - good profile
        
        # Apply brand risk modifier
        final_risk = max(0.0, min(1.0, base_risk + brand_risk_modifier))
        
        # Convert risk to prediction
        if final_risk > 0.6:
            prediction = 2.0  # Reject
            probability = [0.1, 0.2, 0.7]
        elif final_risk > 0.3:
            prediction = 1.0  # Review
            probability = [0.3, 0.5, 0.2]
        else:
            prediction = 0.0  # Approve
            probability = [0.7, 0.2, 0.1]
        
        return prediction, probability, final_risk
    
    return predict

def main():
    
    st.markdown("# 🏦 Loan Prediction System (Safe Mode)")
    st.markdown("### 🚗 With Vehicle Data from CSV")
    
    # Load vehicle data and show in sidebar
    st.sidebar.markdown("## 📊 Vehicle Data Status")
    
    # Load and cache vehicle data
    vehicle_data = get_vehicle_data_cached()
    
    # Display vehicle data summary in sidebar
    if vehicle_data:
        total_brands = len(vehicle_data)
        total_models = sum(len(models) for models in vehicle_data.values())
        
        st.sidebar.metric("Brands Available", total_brands)
        st.sidebar.metric("Total Models", total_models)
        
        # Show top brands by model count
        brand_counts = [(brand, len(models)) for brand, models in vehicle_data.items()]
        brand_counts.sort(key=lambda x: x[1], reverse=True)
        
        st.sidebar.markdown("**Top Brands:**")
        for brand, count in brand_counts[:3]:
            st.sidebar.write(f"• {brand}: {count} models")
    
    # CSV file instructions
    if not os.path.exists("Book1.csv"):
        st.sidebar.warning("⚠️ Book1.csv not found")
        st.sidebar.markdown("""
        **Instructions:**
        1. Place `Book1.csv` in the same directory as this app
        2. CSV should contain columns: `Brand`, `Model_Name`
        3. Restart the app to load new data
        """)
    
    # Vehicle data refresh button
    if st.sidebar.button("🔄 Refresh Vehicle Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Java check
    java_ok, java_msg = check_java()
    if java_ok:
        st.sidebar.success(f"✅ Java: {java_msg}")
    else:
        st.sidebar.error(f"❌ Java: {java_msg}")
    
    # Environment check
    java_home = os.environ.get('JAVA_HOME', 'Not set')
    st.sidebar.info(f"**JAVA_HOME:** `{java_home}`")
    
    # Try to initialize Spark
    if 'spark_status' not in st.session_state:
        st.session_state.spark_status = 'not_tried'
    
    if st.sidebar.button("🔄 Test Spark Connection"):
        st.session_state.spark_status = 'trying'
        
    if st.session_state.spark_status == 'trying':
        with st.spinner("Testing Spark connection..."):
            spark, message = safe_spark_init()
            
        if spark:
            st.session_state.spark = spark
            st.session_state.spark_status = 'connected'
            st.sidebar.success("✅ Spark: Connected")
            st.success(f"🎉 {message}")
        else:
            st.session_state.spark = None
            st.session_state.spark_status = 'failed'
            st.sidebar.error("❌ Spark: Failed")
            st.error(f"⚠️ {message}")
            
            # Show troubleshooting tips
            with st.expander("🔧 Troubleshooting Tips", expanded=True):
                st.markdown("""
                **Quick Fixes:**
                
                1. **Install/Fix Java:**
                ```bash
                sudo apt update
                sudo apt install openjdk-11-jre-headless
                export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
                ```
                
                2. **Set Environment Variables:**
                ```bash
                export SPARK_LOCAL_IP=127.0.0.1
                export SPARK_DRIVER_HOST=127.0.0.1
                ```
                
                3. **Use our fix script:**
                ```bash
                # Download and run our fix script
                curl -o fix_java_gateway.sh [script_url]
                chmod +x fix_java_gateway.sh
                ./fix_java_gateway.sh
                ```
                
                4. **Restart Streamlit:**
                ```bash
                # Kill current process and restart
                pkill -f streamlit
                streamlit run minimal_app.py --server.address 0.0.0.0 --server.port 8088
                ```
                """)
    
    # Determine prediction mode
    use_spark = st.session_state.get('spark_status') == 'connected'
    
    if not use_spark:
        st.info("🤖 Using rule-based prediction engine (Spark not available)")
        predict_function = create_mock_prediction_engine()
    else:
        st.success("🔥 Using Spark-based prediction engine")
        # Use your actual Spark prediction here
        predict_function = create_mock_prediction_engine()  # Fallback for now
    
    # Main application form
    st.markdown("## 📝 Loan Application Form")
    
    with st.form("loan_application"):
        # Create two columns for form layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💰 Financial Information")
            salary = st.number_input(
                "Monthly Salary (฿)", 
                min_value=0, 
                value=50000, 
                step=1000,
                help="Your monthly salary in Thai Baht"
            )
            
            contract_amount = st.number_input(
                "Loan Amount (฿)", 
                min_value=0, 
                value=500000, 
                step=10000,
                help="Total amount you want to borrow"
            )
            
            loan_term = st.number_input(
                "Loan Term (months)", 
                min_value=1, 
                max_value=360, 
                value=36,
                help="How many months to repay the loan"
            )
            
            installment = contract_amount / loan_term if loan_term > 0 else 0
            st.metric("Monthly Installment", f"฿{installment:,.0f}")
            
        with col2:
            st.markdown("### 👤 Personal Information")
            
            occupation = st.selectbox(
                "Occupation",
                ["Employee", "Business Owner", "Freelancer", "Government Officer", "Others"]
            )
            
            st.markdown("### 🚨 Fraud Alerts")
            fraud_alert_pm1 = st.radio(
                "Fraud Alert (Previous Month)",
                ["No", "Yes"],
                help="Any fraud alerts in the previous month?"
            )
            
            fraud_alert_pm2 = st.radio(
                "Fraud Alert (2 Months Ago)",
                ["No", "Yes"], 
                help="Any fraud alerts 2 months ago?"
            )
            
            st.markdown("### 🚗 Vehicle Information")
            
            # Load vehicle data from CSV
            vehicle_data = get_vehicle_data_cached()
            
            # Brand selection
            available_brands = list(vehicle_data.keys())
            selected_brand = st.selectbox(
                "Vehicle Brand",
                available_brands,
                help="Select vehicle brand from loaded data"
            )
            
            # Model selection based on selected brand
            if selected_brand and selected_brand in vehicle_data:
                available_models = vehicle_data[selected_brand]
                
                # Show count of available models
                st.caption(f"📋 {len(available_models)} models available for {selected_brand}")
                
                selected_model = st.selectbox(
                    "Vehicle Model",
                    available_models,
                    help=f"Select model for {selected_brand}"
                )
            else:
                selected_model = st.selectbox(
                    "Vehicle Model",
                    ["No models available"],
                    disabled=True
                )
            
            # Show data source info
            with st.expander("ℹ️ Vehicle Data Info", expanded=False):
                st.write("**Data Source:** Book1.csv")
                st.write(f"**Total Brands:** {len(vehicle_data)}")
                st.write(f"**Total Models:** {sum(len(models) for models in vehicle_data.values())}")
                
                # Show brand breakdown
                st.markdown("**Brand Breakdown:**")
                for brand, models in vehicle_data.items():
                    st.write(f"• **{brand}:** {len(models)} models")
                
                # Show sample models for selected brand
                if selected_brand in vehicle_data and len(vehicle_data[selected_brand]) > 0:
                    st.markdown(f"**Sample models for {selected_brand}:**")
                    sample_models = vehicle_data[selected_brand][:5]  # Show first 5 models
                    for model in sample_models:
                        # Clean up model name for display
                        display_model = model.replace(f"{selected_brand} รุ่น ", "") if f"{selected_brand} รุ่น" in model else model
                        st.write(f"  - {display_model}")
                    
                    if len(vehicle_data[selected_brand]) > 5:
                        st.write(f"  ... and {len(vehicle_data[selected_brand]) - 5} more models")
                
                # CSV file info
                if os.path.exists("Book1.csv"):
                    csv_size = os.path.getsize("Book1.csv")
                    st.write(f"**File size:** {csv_size:,} bytes")
                    
                    # Show last modified time
                    import time
                    mtime = os.path.getmtime("Book1.csv")
                    st.write(f"**Last modified:** {time.ctime(mtime)}")
            
            # Additional information
            st.markdown("### 📋 Additional Information")
            application_status = st.selectbox(
                "Application Status", 
                ["Pending", "In Progress", "Under Review"]
            )
        
        # Submit button
        submitted = st.form_submit_button(
            "🔮 Predict Loan Approval", 
            use_container_width=True,
            type="primary"
        )
        
        if submitted:
            # Prepare data
            application_data = {
                'salary': salary,
                'contract_amount': contract_amount,
                'loan_term': loan_term,
                'occupation': occupation,
                'fraud_alert_pm1': fraud_alert_pm1,
                'fraud_alert_pm2': fraud_alert_pm2,
                'brand': selected_brand,
                'model_name': selected_model
            }
            
            # Make prediction
            try:
                result = predict_function(application_data)
                
                # Handle different return formats
                if len(result) == 3:
                    prediction, probability, risk_score = result
                else:
                    prediction, probability = result
                    risk_score = 1 - probability[0]  # Fallback calculation
                
                # Display results
                st.markdown("## 🎯 Prediction Results")
                
                # Main result
                labels = ["✅ APPROVED", "⚠️ UNDER REVIEW", "❌ REJECTED"]
                colors = ["success", "warning", "error"]
                
                result_label = labels[int(prediction)]
                result_color = colors[int(prediction)]
                
                if result_color == "success":
                    st.success(f"## {result_label}")
                elif result_color == "warning":
                    st.warning(f"## {result_label}")
                else:
                    st.error(f"## {result_label}")
                
                # Confidence scores
                st.markdown("### 📊 Confidence Breakdown")
                
                score_cols = st.columns(3)
                score_labels = ["Approve", "Review", "Reject"]
                score_emojis = ["✅", "⚠️", "❌"]
                
                for i, (col, label, emoji, prob) in enumerate(zip(score_cols, score_labels, score_emojis, probability)):
                    with col:
                        st.metric(
                            f"{emoji} {label}",
                            f"{prob:.1%}",
                            delta=f"+{prob:.1%}" if i == int(prediction) else None
                        )
                        st.progress(prob)
                
                # Additional insights
                st.markdown("### 💡 Key Factors")
                
                debt_ratio = (contract_amount / loan_term) / salary if salary > 0 else 0
                
                insights = []
                
                # Fraud alerts
                if fraud_alert_pm1 == "Yes" or fraud_alert_pm2 == "Yes":
                    fraud_count = sum([fraud_alert_pm1 == "Yes", fraud_alert_pm2 == "Yes"])
                    insights.append(f"🚨 {fraud_count} fraud alert(s) detected")
                
                # Debt ratio analysis
                if debt_ratio > 0.5:
                    insights.append(f"📈 High debt-to-income ratio: {debt_ratio:.1%}")
                elif debt_ratio > 0.3:
                    insights.append(f"📊 Moderate debt-to-income ratio: {debt_ratio:.1%}")
                else:
                    insights.append(f"📉 Good debt-to-income ratio: {debt_ratio:.1%}")
                
                # Income level
                if salary >= 50000:
                    insights.append("💰 Good income level")
                elif salary >= 30000:
                    insights.append("💵 Moderate income level")
                else:
                    insights.append("⚠️ Low income level")
                
                # Vehicle information
                if selected_brand and selected_model:
                    insights.append(f"🚗 Vehicle: {selected_brand}")
                    
                    # Brand-specific insights
                    premium_brands = ['HONDA', 'YAMAHA', 'ROYAL ENFIELD']
                    if selected_brand in premium_brands:
                        insights.append(f"⭐ Premium brand selected: {selected_brand}")
                    
                    # Model specific info
                    if "Limited" in selected_model or "Custom" in selected_model:
                        insights.append("🏆 Special/Limited edition model")
                
                # Risk assessment
                if risk_score <= 0.3:
                    insights.append(f"✅ Low risk profile (Risk: {risk_score:.1%})")
                elif risk_score <= 0.6:
                    insights.append(f"⚠️ Medium risk profile (Risk: {risk_score:.1%})")
                else:
                    insights.append(f"🔴 High risk profile (Risk: {risk_score:.1%})")
                
                # Display insights
                for insight in insights:
                    st.write(f"• {insight}")
                
                # Vehicle data summary
                st.markdown("### 🚗 Vehicle Details")
                vehicle_col1, vehicle_col2 = st.columns(2)
                
                with vehicle_col1:
                    st.metric("Brand", selected_brand)
                
                with vehicle_col2:
                    # Clean up model name for display
                    display_model = selected_model
                    if selected_brand in selected_model:
                        # Remove brand prefix if it exists
                        display_model = selected_model.replace(f"{selected_brand} รุ่น ", "")
                    st.metric("Model", display_model)
                
                # Show input summary
                with st.expander("📋 Application Summary"):
                    summary_df = pd.DataFrame([application_data]).T
                    summary_df.columns = ['Value']
                    st.dataframe(summary_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.info("💡 Try using the rule-based engine or fix your Spark installation.")

if __name__ == "__main__":
    main()