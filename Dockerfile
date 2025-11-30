# ---------------------------
# 1. Base Image
# ---------------------------
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---------------------------
# 2. Install dependencies
# ---------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# 3. Set workdir
# ---------------------------
WORKDIR /app

# ---------------------------
# 4. Install Python deps
# ---------------------------
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . /app/

# ---------------------------
# 5. Streamlit config
# ---------------------------
EXPOSE 8501
ENV STREAMLIT_SERVER_HEADLESS=true

# ---------------------------
# 6. Start App
# ---------------------------
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
