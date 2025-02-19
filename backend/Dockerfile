# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first (for better caching)
COPY requirements.txt .

# Upgrade pip and install dependencies with --no-cache-dir
RUN pip install --upgrade pip && \
    pip install --default-timeout=300 --no-cache-dir -r requirements.txt
# Copy the entire backend directory contents into /app
COPY . /app

WORKDIR /app

# Expose FastAPI port
EXPOSE 8000


# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
