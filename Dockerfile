# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the Docker container
WORKDIR /app

# Copy the current directory contents into the Docker container at /app
COPY . /app

# Install the necessary Python packages
RUN pip install fastapi uvicorn redis sentence-transformers torch requests

# Expose port 8000 to be accessible from outside the container
EXPOSE 8000

# Run the command to start the FastAPI application when the Docker container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
