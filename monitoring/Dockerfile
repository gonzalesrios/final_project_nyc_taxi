FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install dependencies/libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files to the container
COPY app.py .

# Expose the port that the application will run on (Streamlit port)
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]