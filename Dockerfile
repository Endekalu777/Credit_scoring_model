# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements.txt first to leverage Docker caching
COPY ../app/requirements.txt ./

# Install setuptools to avoid pkg_resources error
RUN apt-get update && apt-get install -y python3-setuptools

# Create a virtual environment
RUN python -m venv venv

# Install the dependencies
RUN ./venv/bin/pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your application code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["./venv/bin/python", "app.py"]
