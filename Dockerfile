# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set environment variables to avoid buffering and ensure that output is sent straight to the terminal
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8501

# Define the command to run the application
CMD ["streamlit", "run", "LungCancerDetection.py"]
