FROM python:3.10-slim

# Set the working directory to the app's parent folder
WORKDIR /app

# Install dependencies
COPY startup_env/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire startup_env package
COPY startup_env/ /app/startup_env/

# Set the Python path to include /app so it can find the startup_env package
ENV PYTHONPATH=/app

# Define the command to run the inference script
CMD ["python", "-m", "startup_env.inference"]
