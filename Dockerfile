FROM python:3.11-slim 

# Set working directory
WORKDIR /app

# Copy requirements first (for cache efficiency)
COPY requirements.txt .

# Install dependencies
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Copy the app code
COPY blog_summarizer.py .

# Expose port (optional for web)
EXPOSE 8000

# Default command
CMD ["python", "blog_summarizer.py"]