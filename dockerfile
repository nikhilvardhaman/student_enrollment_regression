# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy application files
COPY app.py .
COPY templates/index.html templates/
COPY ridge_model.pkl .
COPY scaler.pkl .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for Flask
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]