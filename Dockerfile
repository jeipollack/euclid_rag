# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install curl
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy necessary source to container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install .

# Expose the port that the app will run on
EXPOSE 8501

# Run the application when the container launches
CMD ["streamlit", "run", "python/euclid/rag/app.py"]
