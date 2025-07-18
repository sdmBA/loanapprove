# Dockerfile
FROM openjdk:11-jre-slim

# ติดตั้ง Python และ dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# สร้าง symbolic link สำหรับ python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# ตั้งค่า environment variables สำหรับ Spark
ENV JAVA_HOME=/usr/local/openjdk-11
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.7-src.zip:$PYTHONPATH

# Download และติดตั้ง Spark
RUN curl -O https://archive.apache.org/dist/spark/spark-3.4.1/spark-3.4.1-bin-hadoop3.tgz \
    && tar -xzf spark-3.4.1-bin-hadoop3.tgz \
    && mv spark-3.4.1-bin-hadoop3 /opt/spark \
    && rm spark-3.4.1-bin-hadoop3.tgz

# Copy requirements.txt และติดตั้ง Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# สร้าง directory สำหรับ model (ถ้าจำเป็น)
RUN mkdir -p /app/models

# Set permissions
RUN chmod +x /app

# Expose port สำหรับ Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command สำหรับ run application
CMD ["streamlit", "run", "loan_prediction_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]

# ===== Alternative Dockerfile (Lighter version) =====
# ถ้าต้องการขนาดเล็กกว่า ใช้ version นี้แทน

# FROM python:3.9-slim

# # ติดตั้ง Java
# RUN apt-get update && apt-get install -y \
#     openjdk-11-jre-headless \
#     curl \
#     procps \
#     && rm -rf /var/lib/apt/lists/*

# # Set environment variables
# ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
# ENV PATH=$PATH:$JAVA_HOME/bin

# WORKDIR /app

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# EXPOSE 8501

# CMD ["streamlit", "run", "loan_prediction_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]