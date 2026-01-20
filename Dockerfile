FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for opencv
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create a non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Run the Dash app
CMD ["python", "proteome_explorer_ux2.py"]
