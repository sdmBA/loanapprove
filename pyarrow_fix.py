# pyarrow_fix.py - Fixed version that avoids PyArrow issues

import streamlit as st
import pandas as pd
import os
import sys
import subprocess
import time
from datetime import datetime

# Page config
st.set_page_config(
    page_title="🏦 Loan Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disable PyArrow usage in Streamlit
import streamlit as st
# To avoid PyArrow issues, set the config via environment variable or config.toml.
# Direct assignment to st.config is not valid Python syntax and will cause errors.
# You can set the following environment variable before running Streamlit:
# os.environ["STREAMLIT_DATAFRAME_SERIALIZATION"] = "legacy"

# CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .prediction-result {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .approved {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        color: #155724;
        border: 2px solid #28a745;
    }
    .suspicious {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        color: #856404;
        border: 2px solid #ffc107;
    }
    .rejected {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        color: #721c24;
        border: 2px solid #dc3545;
    }
    .summary-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        padding: 0.25rem 0;
        border-bottom: 1px solid #dee2e6;
    }
    .metric-key {
        font-weight: bold;
        color: #495057;
    }
    .metric-value {
        color: #212529;
    }
</style>
""", unsafe_allow_html=True)

def load_vehicle_data_safe():
    """Load vehicle data with PyArrow-safe processing"""
    
    vehicle_data = {}
    csv_file = "Book1.csv"
    
    try:
        if not os.path.exists(csv_file):
            st.sidebar.warning(f"⚠️ CSV file '{csv_file}' not found. Using default data.")
            return get_default_vehicle_data()
        
        # Read CSV with explicit dtype to avoid PyArrow issues
        df = pd.read_csv(csv_file, encoding='utf-8', dtype=str)
        
        # Validate columns
        if 'Brand' not in df.columns or 'Model_Name' not in df.columns:
            st.sidebar.error("❌ CSV file must contain 'Brand' and 'Model_Name' columns")
            return get_default_vehicle_data()
        
        # Process data safely
        for _, row in df.iterrows():
            brand = str(row['Brand']).strip() if pd.notna(row['Brand']) else ''
            model = str(row['Model_Name']).strip() if pd.notna(row['Model_Name']) else ''
            
            if brand and model and len(brand) > 0 and len(model) > 0:
                if brand not in vehicle_data:
                    vehicle_data[brand] = []
                if model not in vehicle_data[brand]:
                    vehicle_data[brand].append(model)
        
        # Sort models
        for brand in vehicle_data:
            vehicle_data[brand].sort()
        
        if not vehicle_data:
            return get_default_vehicle_data()
        
        return vehicle_data
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading CSV: {e}")
        return get_default_vehicle_data()

def get_default_vehicle_data():
    """Fallback vehicle data"""
    return {
        "HONDA": [
            "HONDA รุ่น PCX 150", "HONDA รุ่น CLICK 160", "HONDA รุ่น WAVE 110i"
        ],
        "YAMAHA": [
            "YAMAHA รุ่น AEROX 155", "YAMAHA รุ่น NMAX 155", "YAMAHA รุ่น FINO"
        ],
        "SUZUKI": [
            "SUZUKI รุ่น BURGMAN STREET", "SUZUKI รุ่น SMASH", "SUZUKI รุ่น RAIDER"
        ],
        "Others": ["Other Model", "Custom Model"]
    }

def check_java_safe():
    """Check Java availability safely"""
    try:
        result = subprocess.run(['java', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version_line = result.stderr.split('\n')[0] if result.stderr else "Java found"
            return True, version_line
        else:
            return False, "Java command failed"
    except:
        return False, "Java not available"

def create_prediction_engine():
    """Enhanced prediction engine"""
    
    def predict(data):
        salary = data.get('salary', 0)
        contract_amount = data.get('contract_amount', 0)
        fraud_pm1 = data.get('fraud_alert_pm1', 'No')
        fraud_pm2 = data.get('fraud_alert_pm2', 'No')
        loan_term = data.get('loan_term', 36)
        brand = data.get('brand', '')
        occupation = data.get('occupation', '')
        
        # Calculate metrics
        monthly_payment = contract_amount / loan_term if loan_term > 0 else 0
        debt_ratio = monthly_payment / salary if salary > 0 else float('inf')
        
        # Risk calculation
        base_risk = 0.0
        
        # Fraud impact
        if fraud_pm1 == "Yes" and fraud_pm2 == "Yes":
            base_risk += 0.8
        elif fraud_pm1 == "Yes" or fraud_pm2 == "Yes":
            base_risk += 0.5
        
        # Debt ratio impact
        if debt_ratio > 0.6:
            base_risk += 0.7
        elif debt_ratio > 0.4:
            base_risk += 0.4
        elif debt_ratio > 0.3:
            base_risk += 0.2
        
        # Brand impact
        premium_brands = ['HONDA', 'YAMAHA', 'ROYAL ENFIELD']
        if brand in premium_brands:
            base_risk -= 0.1
        elif brand == 'Others':
            base_risk += 0.1
        
        # Occupation impact
        stable_jobs = ['Government Officer', 'Employee']
        if occupation in stable_jobs:
            base_risk -= 0.05
        elif occupation in ['Freelancer', 'Others']:
            base_risk += 0.05
        
        # Income impact
        if salary >= 60000:
            base_risk -= 0.1
        elif salary < 25000:
            base_risk += 0.1
        
        # Normalize risk
        final_risk = max(0.0, min(1.0, base_risk))
        
        # Convert to prediction
        if final_risk > 0.65:
            prediction = 2.0  # Reject
            probability = [0.1, 0.2, 0.7]
        elif final_risk > 0.35:
            prediction = 1.0  # Review
            probability = [0.25, 0.55, 0.2]
        else:
            prediction = 0.0  # Approve
            probability = [0.75, 0.2, 0.05]
        
        return prediction, probability, final_risk
    
    return predict

def display_summary_safe(data, prediction_info):
    """Display summary without PyArrow issues"""
    
    st.markdown("### 📋 Application Summary")
    
    # Financial section
    st.markdown('<div class="summary-section">', unsafe_allow_html=True)
    st.markdown("**💰 Financial Information**")
    
    financial_items = [
        ("Monthly Salary", f"฿{data['salary']:,}"),
        ("Loan Amount", f"฿{data['contract_amount']:,}"),
        ("Loan Term", f"{data['loan_term']} months"),
        ("Monthly Payment", f"฿{data['contract_amount']/data['loan_term']:,.0f}"),
        ("Debt Ratio", f"{(data['contract_amount']/data['loan_term'])/data['salary']*100:.1f}%")
    ]
    
    for key, value in financial_items:
        st.markdown(f'<div class="metric-row"><span class="metric-key">{key}:</span><span class="metric-value">{value}</span></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Personal section
    st.markdown('<div class="summary-section">', unsafe_allow_html=True)
    st.markdown("**👤 Personal Information**")
    
    personal_items = [
        ("Occupation", data['occupation']),
        ("Fraud Alert PM1", data['fraud_alert_pm1']),
        ("Fraud Alert PM2", data['fraud_alert_pm2'])
    ]
    
    for key, value in personal_items:
        st.markdown(f'<div class="metric-row"><span class="metric-key">{key}:</span><span class="metric-value">{value}</span></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Vehicle section
    st.markdown('<div class="summary-section">', unsafe_allow_html=True)
    st.markdown("**🚗 Vehicle Information**")
    
    vehicle_items = [
        ("Brand", data['brand']),
        ("Model", data['model_name'])
    ]
    
    for key, value in vehicle_items:
        st.markdown(f'<div class="metric-row"><span class="metric-key">{key}:</span><span class="metric-value">{value}</span></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Assessment section
    st.markdown('<div class="summary-section">', unsafe_allow_html=True)
    st.markdown("**🎯 Assessment Results**")
    
    prediction, probability, risk_score = prediction_info
    labels = ["APPROVED", "UNDER REVIEW", "REJECTED"]
    result_label = labels[int(prediction)]
    
    assessment_items = [
        ("Prediction", result_label),
        ("Risk Score", f"{risk_score:.1%}"),
        ("Confidence", f"{probability[int(prediction)]:.1%}")
    ]
    
    for key, value in assessment_items:
        st.markdown(f'<div class="metric-row"><span class="metric-key">{key}:</span><span class="metric-value">{value}</span></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    
    # Header
    st.markdown('<h1 class="main-header">🏦 Loan Application Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("### 🚗 PyArrow Compatible Version")
    
    # Load vehicle data
    vehicle_data = load_vehicle_data_safe()
    
    # Sidebar status
    st.sidebar.markdown("## 🔧 System Status")
    
    # Java status
    java_ok, java_msg = check_java_safe()
    if java_ok:
        st.sidebar.success("✅ Java: Available")
    else:
        st.sidebar.warning("⚠️ Java: Not available")
    
    # Vehicle data status
    st.sidebar.markdown("## 📊 Vehicle Data")
    if vehicle_data:
        total_brands = len(vehicle_data)
        total_models = sum(len(models) for models in vehicle_data.values())
        st.sidebar.metric("Brands", total_brands)
        st.sidebar.metric("Models", total_models)
        
        st.sidebar.markdown("**Available Brands:**")
        for brand, models in vehicle_data.items():
            st.sidebar.write(f"• **{brand}:** {len(models)} models")
    
    # Refresh button
    if st.sidebar.button("🔄 Reload Data"):
        st.rerun()
    
    # Prediction mode
    st.info("🤖 Using enhanced rule-based prediction engine")
    predict_function = create_prediction_engine()
    
    # Main form
    st.markdown("## 📝 Loan Application Form")
    
    with st.form("loan_application"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💰 Financial Information")
            
            salary = st.number_input(
                "Monthly Salary (฿)", 
                min_value=0, 
                value=50000, 
                step=1000
            )
            
            contract_amount = st.number_input(
                "Loan Amount (฿)", 
                min_value=0, 
                value=500000, 
                step=10000
            )
            
            loan_term = st.number_input(
                "Loan Term (months)", 
                min_value=1, 
                max_value=360, 
                value=36
            )
            
            # Show calculations
            monthly_payment = contract_amount / loan_term if loan_term > 0 else 0
            debt_ratio = (monthly_payment / salary * 100) if salary > 0 else 0
            
            col1a, col1b = st.columns(2)
            with col1a:
                st.metric("Monthly Payment", f"฿{monthly_payment:,.0f}")
            with col1b:
                st.metric("Debt Ratio", f"{debt_ratio:.1f}%")
            
        with col2:
            st.markdown("### 👤 Personal Information")
            
            occupation = st.selectbox(
                "Occupation",
                ["ธุรกิจส่วนตัว","ข้าราชการ","พนักงานบริษัทเอกชน","พนักงานรัฐวิสาหกิจ","อื่นๆ"]
            )
            
            st.markdown("### 🚨 Fraud Alert History")
            
            fraud_alert_pm1 = st.radio(
                "Fraud Alert (Previous Month)",
                ["No", "Yes"],
                key="fraud1"
            )
            
            fraud_alert_pm2 = st.radio(
                "Fraud Alert (2 Months Ago)",
                ["No", "Yes"],
                key="fraud2"
            )
            
            st.markdown("### 🚗 Vehicle Information")
            
            # Vehicle selection
            brands = list(vehicle_data.keys()) if vehicle_data else ["No brands"]
            
            selected_brand = st.selectbox(
                "Vehicle Brand",
                brands,
                key="brand_select"
            )
            
            # Model selection
            if vehicle_data and selected_brand in vehicle_data:
                models = vehicle_data[selected_brand]
                if models:
                    st.caption(f"📋 {len(models)} models available")
                    selected_model = st.selectbox(
                        "Vehicle Model",
                        models,
                        key=f"model_{selected_brand}"
                    )
                else:
                    selected_model = "No models available"
                    st.warning("No models found")
            else:
                selected_model = "No models available"
        
        # Submit
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
                with st.spinner("🔄 Processing..."):
                    prediction_result = predict_function(application_data)
                    prediction, probability, risk_score = prediction_result
                
                # Display results
                st.markdown("## 🎯 Prediction Results")
                
                # Main result
                labels = ["✅ APPROVED", "⚠️ UNDER REVIEW", "❌ REJECTED"]
                colors = ["approved", "suspicious", "rejected"]
                
                result_label = labels[int(prediction)]
                result_color = colors[int(prediction)]
                
                st.markdown(f"""
                <div class="prediction-result {result_color}">
                    <strong>{result_label}</strong><br>
                    <small>Confidence: {probability[int(prediction)]:.1%}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence breakdown
                st.markdown("### 📊 Confidence Scores")
                
                score_cols = st.columns(3)
                score_labels = ["Approve", "Review", "Reject"]
                score_emojis = ["✅", "⚠️", "❌"]
                
                for i, (col, label, emoji, prob) in enumerate(zip(score_cols, score_labels, score_emojis, probability)):
                    with col:
                        st.metric(f"{emoji} {label}", f"{prob:.1%}")
                        st.progress(prob)
                
                # Risk analysis
                st.markdown("### 🎯 Risk Assessment")
                
                if risk_score <= 0.3:
                    risk_level = "🟢 Low Risk"
                elif risk_score <= 0.6:
                    risk_level = "🟡 Medium Risk"
                else:
                    risk_level = "🔴 High Risk"
                
                risk_col1, risk_col2 = st.columns(2)
                with risk_col1:
                    st.metric("Risk Level", risk_level)
                with risk_col2:
                    st.metric("Risk Score", f"{risk_score:.1%}")
                
                # Key factors
                st.markdown("### 💡 Assessment Factors")
                
                factors = []
                
                # Debt ratio
                if debt_ratio > 60:
                    factors.append("🔴 High debt-to-income ratio")
                elif debt_ratio > 40:
                    factors.append("🟡 Moderate debt-to-income ratio")
                else:
                    factors.append("🟢 Good debt-to-income ratio")
                
                # Fraud alerts
                fraud_count = sum([fraud_alert_pm1 == "Yes", fraud_alert_pm2 == "Yes"])
                if fraud_count > 0:
                    factors.append(f"🔴 {fraud_count} fraud alert(s) detected")
                else:
                    factors.append("🟢 No fraud alerts")
                
                # Income
                if salary >= 60000:
                    factors.append("🟢 High income level")
                elif salary >= 25000:
                    factors.append("🟡 Moderate income level")
                else:
                    factors.append("🔴 Low income level")
                
                # Vehicle brand
                premium_brands = ['HONDA', 'YAMAHA', 'ROYAL ENFIELD']
                if selected_brand in premium_brands:
                    factors.append(f"🟢 Premium brand: {selected_brand}")
                else:
                    factors.append(f"🟡 Vehicle brand: {selected_brand}")
                
                for factor in factors:
                    st.write(f"• {factor}")
                
                # Summary (safe display)
                with st.expander("📋 Application Summary"):
                    display_summary_safe(application_data, prediction_result)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.info("💡 Please check your input and try again.")

if __name__ == "__main__":
    main()