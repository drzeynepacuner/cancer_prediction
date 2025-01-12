# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Expose the Flask app port
EXPOSE 5000

# Set an environment variable for dynamic path handling
ENV BASE_DIR /app

# Run the main.py script in 'app' mode
CMD ["python", "main.py", "--mode", "app"]
