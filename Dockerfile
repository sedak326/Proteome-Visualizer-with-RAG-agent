FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for opencv
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_dash.txt .
RUN pip install --no-cache-dir -r requirements_dash.txt

# Copy app code (not large data files)
COPY proteome_explorer_ux2.py .
COPY download_data.py .

# Download data assets and start app
CMD python download_data.py && python proteome_explorer_ux2.py
