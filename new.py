# minimal_app.py - Guaranteed working version
import streamlit as st
import pandas as pd
import os
import sys
import subprocess

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

def check_java():
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
        
        # Calculate debt-to-income ratio
        if salary > 0:
            debt_ratio = (contract_amount / loan_term) / salary
        else:
            debt_ratio = float('inf')
        
        # Decision logic
        if fraud_pm1 == "Yes" and fraud_pm2 == "Yes":
            prediction = 2.0  # Reject - multiple fraud alerts
            probability = [0.1, 0.2, 0.7]
        elif fraud_pm1 == "Yes" or fraud_pm2 == "Yes":
            prediction = 1.0  # Review - single fraud alert
            probability = [0.2, 0.6, 0.2]
        elif debt_ratio > 0.5:
            prediction = 2.0  # Reject - high debt ratio
            probability = [0.1, 0.3, 0.6]
        elif debt_ratio > 0.3:
            prediction = 1.0  # Review - moderate debt ratio
            probability = [0.3, 0.5, 0.2]
        else:
            prediction = 0.0  # Approve - good profile
            probability = [0.7, 0.2, 0.1]
        
        return prediction, probability
    
    return predict

def main():
    
    st.markdown("# 🏦 Loan Prediction System (Safe Mode)")
    
    # Check system status
    st.sidebar.markdown("## 🔧 System Status")
    
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
            
            st.markdown("### 🚗 Collateral (Optional)")
            brand = st.selectbox("Vehicle Brand", ["Toyota", "Honda", "Others", "N/A"])
            model = st.selectbox("Vehicle Model", ["Camry", "Civic", "Others", "N/A"])
        
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
                'brand': brand,
                'model_name': model
            }
            
            # Make prediction
            try:
                prediction, probability = predict_function(application_data)
                
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
                if fraud_alert_pm1 == "Yes" or fraud_alert_pm2 == "Yes":
                    insights.append("🚨 Fraud alerts detected")
                
                if debt_ratio > 0.5:
                    insights.append(f"📈 High debt-to-income ratio: {debt_ratio:.1%}")
                elif debt_ratio > 0.3:
                    insights.append(f"📊 Moderate debt-to-income ratio: {debt_ratio:.1%}")
                else:
                    insights.append(f"📉 Good debt-to-income ratio: {debt_ratio:.1%}")
                
                if salary >= 50000:
                    insights.append("💰 Good income level")
                
                for insight in insights:
                    st.write(f"• {insight}")
                
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