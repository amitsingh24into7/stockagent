## Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
#CMD ["streamlit", "run", "stock_streamlit_sql.py", "--server.enableCORS", "false"]
CMD ["streamlit", "run", "stock_streamlit_sql.py", "--server.enableCORS", "false", "--server.port", "8501", "--server.headless", "true"]
