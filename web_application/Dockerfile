# Use the official Python image as the base image
FROM python:3.10.10

# Set the working directory inside the container
WORKDIR /web_application/

# Upgrade pip
RUN pip install --upgrade pip

# Install system dependencies
RUN apt-get update && apt-get install -y gcc python3-dev git

# Copy the requirements file to the working directory
COPY requirements.txt /web_application/

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the project files to the working directory
COPY . /web_application/

# Expose the port that your application listens on
EXPOSE 8501

# Set the command to run your application
CMD ["streamlit", "run", "main.py"]