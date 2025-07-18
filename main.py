import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType
import numpy as np

# ตั้งค่า page config
st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS สำหรับปรับแต่ง UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .approved {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .suspicious {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .rejected {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_spark():
    """Initialize Spark Session"""
    spark = SparkSession.builder \
        .appName("LoanPredictionApp") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()\
       
       
        master("spark://spark-master:7077").\
       
        getOrCreate()
    return spark

@st.cache_resource
def load_model(model_path):
    """Load trained PipelineModel"""
    try:
        model = PipelineModel.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def create_spark_dataframe(data, spark):
    """Create Spark DataFrame from input data"""
    
    # Define schema ตาม raw_df.printSchema()
    schema = StructType([
        StructField("salary", IntegerType(), True),
        StructField("contract_amount", IntegerType(), True),
        StructField("loan_term", IntegerType(), True),
        StructField("installment_amount", DoubleType(), True),
        StructField("interest_rate", DoubleType(), True),
        StructField("occupation", StringType(), True),
        StructField("fraud_alert_pm1", StringType(), True),
        StructField("fraud_alert_pm2", StringType(), True),
        StructField("brand", StringType(), True),
        StructField("model_name", StringType(), True),
        StructField("application_status", StringType(), True),
        StructField("request_id", IntegerType(), True),
        StructField("contract_id", StringType(), True)
    ])
    
    # สร้าง DataFrame
    df = spark.createDataFrame([data], schema)
    return df

def get_prediction_label(prediction_value):
    """แปลง prediction value เป็น label ที่อ่านง่าย"""
    label_mapping = {
        0.0: "อนุมัติ",
        1.0: "อนุมัติ แต่ต้องตรวจสอบเพิ่มเติม",
        2.0: "ปฏิเสธ"
    }
    return label_mapping.get(prediction_value, f"Unknown ({prediction_value})")

def get_prediction_class(prediction_value):
    """ได้ CSS class สำหรับแสดงผล"""
    class_mapping = {
        0.0: "approved",
        1.0: "approved with conditions", 
        2.0: "rejected"
    }
    return class_mapping.get(prediction_value, "suspicious")

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 Loan Application Prediction System</h1>', unsafe_allow_html=True)
    
    # Sidebar สำหรับ model settings
    st.sidebar.markdown("## ⚙️ Model Settings")
    
    # Model path input
    model_path = st.sidebar.text_input(
        "Model Path",
        value="/path/to/your/saved/model",
        help="Path to your saved PipelineModel"
    )
    
    # Initialize Spark
    spark = init_spark()
    
    # Load model
    if st.sidebar.button("🔄 Load Model"):
        model = load_model(model_path)
        if model:
            st.session_state.model = model
            st.sidebar.success("✅ Model loaded successfully!")
        else:
            st.sidebar.error("❌ Failed to load model")
    
    # Check if model is loaded
    if 'model' not in st.session_state:
        st.warning("⚠️ Please load the model first from the sidebar.")
        st.stop()
    
    # Main form
    st.markdown("## 📝 Loan Application Form")
    
    with st.form("prediction_form"):
        # สร้าง 3 columns สำหรับจัดเรียง input
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 💰 Financial Information")
            salary = st.number_input(
                "Salary (฿)",
                min_value=0,
                value=50000,
                step=1000,
                help="Monthly salary in Thai Baht"
            )
            
            contract_amount = st.number_input(
                "Contract Amount (฿)",
                min_value=0,
                value=500000,
                step=10000,
                help="Total loan amount"
            )
            
            loan_term = st.number_input(
                "Loan Term (months)",
                min_value=1,
                max_value=360,
                value=36,
                step=1,
                help="Loan duration in months"
            )
            
            installment_amount = st.number_input(
                "Installment Amount (฿)",
                min_value=0.0,
                value=15000.0,
                step=100.0,
                help="Monthly installment amount"
            )
            
            interest_rate = st.number_input(
                "Interest Rate (%)",
                min_value=0.0,
                max_value=50.0,
                value=7.5,
                step=0.1,
                help="Annual interest rate percentage"
            )
        
        with col2:
            st.markdown("### 👤 Personal Information")
            
            # Occupation dropdown
            occupation_options = [
                "Employee", "Business Owner", "Freelancer", 
                "Government Officer", "Teacher", "Doctor",
                "Engineer", "Lawyer", "Others"
            ]
            occupation = st.selectbox(
                "Occupation",
                occupation_options,
                help="Select your occupation"
            )
            
            # Fraud Alert Radio Buttons
            fraud_alert_pm1 = st.radio(
                "Fraud Alert PM1",
                ["No", "Yes"],
                help="Previous month fraud alert status"
            )
            
            fraud_alert_pm2 = st.radio(
                "Fraud Alert PM2",
                ["No", "Yes"],
                help="Two months ago fraud alert status"
            )
        
        with col3:
            st.markdown("### 🚗 Vehicle Information")
            
            # Brand dropdown
            brand_options = [
                "Toyota", "Honda", "Mazda", "Nissan", "Ford",
                "Chevrolet", "BMW", "Mercedes-Benz", "Audi",
                "Lexus", "Isuzu", "Mitsubishi", "Others"
            ]
            brand = st.selectbox(
                "Vehicle Brand",
                brand_options,
                help="Select vehicle brand"
            )
            
            # Model dropdown (ปรับตาม brand ที่เลือก)
            model_options = {
                "Toyota": ["Camry", "Corolla", "Vios", "Yaris", "Hilux", "Fortuner"],
                "Honda": ["Civic", "Accord", "City", "HR-V", "CR-V", "BR-V"],
                "Mazda": ["Mazda2", "Mazda3", "CX-3", "CX-5", "BT-50"],
                "Others": ["Model A", "Model B", "Model C"]
            }
            
            available_models = model_options.get(brand, ["Model A", "Model B", "Model C"])
            model_name = st.selectbox(
                "Vehicle Model",
                available_models,
                help="Select vehicle model"
            )
            
            st.markdown("### 📋 Additional Information")
            
            application_status = st.selectbox(
                "Application Status",
                ["Pending", "In Progress", "Under Review"],
                help="Current application status"
            )
            
            request_id = st.number_input(
                "Request ID",
                min_value=1,
                value=12345,
                help="Unique request identifier"
            )
            
            contract_id = st.text_input(
                "Contract ID",
                value="CON-001",
                help="Contract identifier"
            )
        
        # Submit button
        submitted = st.form_submit_button("🔮 Predict Application Result", use_container_width=True)
        
        if submitted:
            # เตรียมข้อมูลสำหรับ prediction
            input_data = {
                "salary": salary,
                "contract_amount": contract_amount,
                "loan_term": loan_term,
                "installment_amount": installment_amount,
                "interest_rate": interest_rate,
                "occupation": occupation,
                "fraud_alert_pm1": fraud_alert_pm1,
                "fraud_alert_pm2": fraud_alert_pm2,
                "brand": brand,
                "model_name": model_name,
                "application_status": application_status,
                "request_id": request_id,
                "contract_id": contract_id
            }
            
            try:
                # สร้าง Spark DataFrame
                df = create_spark_dataframe(input_data, spark)
                
                # ทำ prediction
                with st.spinner("🔄 Processing prediction..."):
                    predictions = st.session_state.model.transform(df)
                    
                    # ดึงผลลัพธ์
                    result = predictions.select("prediction", "probability").collect()[0]
                    prediction_value = result["prediction"]
                    probability = result["probability"].toArray()
                
                # แสดงผลลัพธ์
                st.markdown("## 🎯 Prediction Results")
                
                # Main result
                prediction_label = get_prediction_label(prediction_value)
                prediction_class = get_prediction_class(prediction_value)
                
                st.markdown(f"""
                <div class="prediction-result {prediction_class}">
                    📊 Prediction: {prediction_label}
                </div>
                """, unsafe_allow_html=True)
                
                # Probability breakdown
                st.markdown("### 📈 Confidence Scores")
                col1, col2, col3 = st.columns(3)
                
                labels = ["อนุมัติ", "รอตรวจสอบ", "ปฏิเสธ"]
                colors = ["#28a745", "#ffc107", "#dc3545"]
                
                for i, (label, prob, color) in enumerate(zip(labels, probability, colors)):
                    with [col1, col2, col3][i]:
                        st.metric(
                            label=label,
                            value=f"{prob:.1%}",
                            delta=None
                        )
                        st.progress(prob)
                
                # Input summary
                with st.expander("📋 Input Summary"):
                    st.json(input_data)
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
                st.info("💡 Please check your model path and ensure all required features are properly indexed.")

if __name__ == "__main__":
    main()