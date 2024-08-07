# Use the official Python image with Python 3.11
FROM python:3.11-slim

# Install system dependencies and TensorFlow
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pipenv
RUN pip install pipenv tensorflow

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in Pipfile.lock
RUN pipenv install --deploy --ignore-pipfile

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py

# Run app.py when the container launches
CMD ["pipenv", "run", "flask", "run", "--host=0.0.0.0"]
