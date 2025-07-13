FROM python:3.11-slim 

# Working directory
WORKDIR /app

# Requirements
COPY requirements.txt .

# Dependencies
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Copy the app code
COPY blog_summarizer.py .
COPY app.py .

# Expose port
EXPOSE 7777

# Commands
CMD ["python", "app.py"]